use std::{
    fmt::Display,
    sync::{Arc, Mutex},
};

// use crate::half_edge::drawing::Decoration;
use frostfire::prelude::*;

// #[cfg(feature = "drawing")]
use cgmath::{Angle, InnerSpace, Rad, Vector2, Zero};
use rand::{rngs::{SmallRng, StdRng}, SeedableRng, Rng};

// #[cfg(feature = "drawing")]
use super::{
    drawing::{CetzEdge, CetzString, Decoration, EdgeGeometry},
    involution::EdgeVec,
};
use super::{
    involution::{EdgeIndex, HedgePair},
    nodestore::{NodeStorageOps, NodeStorageVec},
    Flow, HedgeGraph, NodeIndex, Orientation,
};

/// Represents a graph vertex (node) with associated layout information, specifically its 2D position.
///
/// # Type Parameters
/// - `V`: The type of the custom data associated with the vertex.
pub struct LayoutVertex<V> {
    /// The custom data associated with the vertex.
    pub data: V,
    /// The 2D position (x, y coordinates) of the vertex in the layout.
    pos: Vector2<f64>,
}

impl<V> LayoutVertex<V> {
    pub fn new(data: V, x: f64, y: f64) -> Self {
        LayoutVertex {
            data,
            pos: Vector2::new(x, y),
        }
    }

    pub fn pos(&self) -> &Vector2<f64> {
        &self.pos
    }
}

/// Represents a graph edge with associated layout information, including its geometry
/// for drawing (e.g., straight line, curve, control points for splines).
///
/// # Type Parameters
/// - `E`: The type of the custom data associated with the edge.
pub struct LayoutEdge<E> {
    /// The custom data associated with the edge.
    pub data: E,
    /// The geometric representation of the edge, defining how it should be drawn.
    /// This could include control points for curves, arrow information, etc.
    geometry: EdgeGeometry,
}

impl<E> LayoutEdge<E> {
    pub fn map<E2>(self, f: impl FnOnce(E) -> E2) -> LayoutEdge<E2> {
        LayoutEdge {
            data: f(self.data),
            geometry: self.geometry,
        }
    }

    pub fn new_internal(
        data: E,
        source: &Vector2<f64>,
        sink: &Vector2<f64>,
        x: f64,
        y: f64,
    ) -> Self {
        let pos = Vector2::new(x, y);
        let (center, _) = EdgeGeometry::circumcircle([source, sink, &pos]);

        let center_to_pos = pos - center;
        let center_to_source = source - center;
        let center_to_sink = sink - center;

        let sink_angle = center_to_source.angle(center_to_sink).normalize();
        let pos_angle = center_to_source.angle(center_to_pos).normalize();

        let angle = Vector2::unit_y().angle(center_to_pos)
            + if sink_angle > pos_angle {
                Rad::turn_div_2()
            } else {
                Rad::zero()
            };

        LayoutEdge {
            data,
            geometry: EdgeGeometry::Simple {
                pos: Vector2::new(x, y),
                angle,
            },
        }
    }

    pub fn new_external(data: E, source: &Vector2<f64>, x: f64, y: f64, flow: Flow) -> Self {
        let pos = Vector2::new(x, y);
        let mut pos_angle = Vector2::unit_x().angle(pos - source);

        if let Flow::Source = flow {
        } else {
            pos_angle += Rad::turn_div_2()
        }
        LayoutEdge {
            data,
            geometry: EdgeGeometry::Simple {
                pos: Vector2::new(x, y),
                angle: pos_angle,
            },
        }
    }

    pub fn pos(&self) -> &Vector2<f64> {
        self.geometry.pos()
    }

    pub fn to_fancy(
        &mut self,
        source: Vector2<f64>,
        sink: Option<Vector2<f64>>,
        flow: Flow,
        settings: &FancySettings,
    ) {
        match self.geometry {
            EdgeGeometry::Simple { .. } => {}
            _ => {
                return;
            }
        };
        self.geometry = self.geometry.clone().to_fancy(source, sink, flow, settings);
    }
}

impl HedgePair {
    pub fn cetz<E: CetzEdge, V, H>(
        &self,
        graph: &PositionalHedgeGraph<E, V, H>,
        orientation: Orientation,
    ) -> String {
        self.cetz_impl(graph, &|e| e.label(), &|e| e.decoration(), orientation)
    }

    fn cetz_impl<E, V, H, L: Display>(
        &self,
        graph: &PositionalHedgeGraph<E, V, H>,
        label: &impl Fn(&E) -> L,
        decoration: &impl Fn(&E) -> Decoration,
        orientation: Orientation,
    ) -> String {
        match self {
            HedgePair::Unpaired { hedge, .. } => {
                let data = graph.get_edge_data(*hedge);
                data.geometry.cetz_identity(
                    graph.node_id(*hedge),
                    decoration(&data.data),
                    label(&data.data),
                    orientation,
                )
            }
            HedgePair::Paired { source, sink } => {
                let data = graph.get_edge_data(*source);
                data.geometry.cetz_pair(
                    graph.node_id(*source),
                    graph.node_id(*sink),
                    decoration(&data.data),
                    label(&data.data),
                    orientation,
                )
            }
            HedgePair::Split { source, sink, .. } => {
                let data = graph.get_edge_data(*source);
                data.geometry.cetz_pair(
                    graph.node_id(*source),
                    graph.node_id(*sink),
                    decoration(&data.data),
                    label(&data.data),
                    orientation,
                )
            }
        }
    }
}

impl<E> LayoutEdge<E> {}

pub type PositionalHedgeGraph<E, V, H> =
    HedgeGraph<LayoutEdge<E>, LayoutVertex<V>, H, NodeStorageVec<LayoutVertex<V>>>;

/// A constant string containing Cetz (a TikZ library for Typst) code preamble.
/// This preamble defines reusable drawing commands and styles (like node shapes,
/// colors, and edge decorations) for generating graph visualizations in Typst.
const CETZ_PREAMBLE: &str = r#"
let node(pos)=cetz.draw.circle(pos,radius:0.02,fill: black)
let stroke = 0.7pt
let amplitude = 0.051
let arrow-scale = 0.8
let segment-length = 0.0521
let edge(..points,decoration:"",angle:0deg)={
    if decoration == "coil"{
    cetz.decorations.coil(cetz.draw.hobby(..points),amplitude:amplitude,stroke:stroke,align:"MID")
    } else if decoration == "wave" {
        cetz.decorations.wave(cetz.draw.hobby(..points),amplitude:amplitude,stroke:stroke)
    }  else if decoration == "arrow"{
        let points = points.pos()
        if points.len()==2{
          let center = (0.5*(points.at(0).at(0)+points.at(1).at(0)),0.5*(points.at(0).at(1)+points.at(1).at(1)))
          cetz.draw.hobby(..points,stroke:stroke)
          cetz.draw.mark(center,angle,symbol: ">", fill: black,anchor: "center",scale:arrow-scale)
        } else {
          let (first,center,..other)=points
          cetz.draw.hobby(first,center,..other,stroke:stroke)
            cetz.draw.mark(center,angle,symbol: ">", fill: black,anchor: "center",scale:arrow-scale)
        }

    }else {
            cetz.draw.hobby(..points,stroke:stroke)
    }
}
"#;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// Settings for controlling the "fancy" rendering of edges, such as curved paths,
/// arrowheads, and label positioning.
pub struct FancySettings {
    /// The distance to shift edge labels away from the edge path.
    pub label_shift: f64,
    /// An optional percentage (0.0 to 1.0) determining the length of an arrowhead
    /// relative to the edge segment it's on. `None` might mean no arrow or a default style.
    pub arrow_angle_percentage: Option<f64>,
    /// The distance to shift arrowheads along or perpendicular to the edge path.
    pub arrow_shift: f64,
}

impl Default for FancySettings {
    fn default() -> Self {
        Self {
            label_shift: 0.1,
            arrow_angle_percentage: None,
            arrow_shift: 0.1,
        }
    }
}

impl FancySettings {
    pub fn label_shift(&self) -> f64 {
        self.label_shift
            + if self.arrow_angle_percentage.is_some() {
                self.arrow_shift
            } else {
                0.0
            }
    }
}

impl<E, V, H> PositionalHedgeGraph<E, V, H> {
    pub fn to_fancy(&mut self, settings: &FancySettings) {
        self.edge_store
            .data
            .iter_mut()
            .for_each(|(_, (e, p))| match p {
                HedgePair::Paired { source, sink } => {
                    let source_pos =
                        self.node_store.node_data[self.node_store.hedge_data[*source]].pos;
                    let sink_pos = self.node_store.node_data[self.node_store.hedge_data[*sink]].pos;
                    e.to_fancy(source_pos, Some(sink_pos), Flow::Source, settings);
                }
                HedgePair::Unpaired { hedge, flow } => {
                    let source_pos =
                        self.node_store.node_data[self.node_store.hedge_data[*hedge]].pos;
                    e.to_fancy(source_pos, None, *flow, settings);
                }
                _ => {}
            });
    }

    fn cetz_preamble() -> String {
        let mut out = String::new();
        out.push_str(
            r#"#import "@preview/cetz:0.3.1"
            "#,
        );
        out
    }

    #[allow(clippy::type_complexity)]
    pub fn cetz_impl_collection(
        graphs: &[(String, String, Vec<(String, Self)>)],
        edge_label: &impl Fn(&E) -> String,
        edge_decoration: &impl Fn(&E) -> Decoration,
        pagebreak: bool,
    ) -> String {
        let mut out = Self::cetz_preamble();
        out.push_str("#{\nlet cols = (30%,30%,30%)\n");

        out.push_str(CETZ_PREAMBLE);
        for (col_name, _, gs) in graphs.iter() {
            for (j, g) in gs.iter().enumerate() {
                out.push_str(&format!(
                    "let {col_name}{j}={}",
                    g.1.cetz_bare(edge_label, edge_decoration)
                ))
            }
        }
        for (col_name, label, gs) in graphs.iter() {
            out.push_str(&format!("[{label}]\n"));
            out.push_str("grid(columns: cols,gutter: 20pt,");
            for (i, (lab, _)) in gs.iter().enumerate() {
                out.push_str(&format!("box[#{col_name}{i} {lab}],"));
            }
            out.push_str(")\n");
            if pagebreak {
                out.push_str("pagebreak()\n");
            }
        }
        out.push('}');
        out
    }

    fn cetz_bare(
        &self,
        edge_label: &impl Fn(&E) -> String,
        edge_decoration: &impl Fn(&E) -> Decoration,
    ) -> String {
        let mut out = String::from("cetz.canvas(length:50%,{\n");
        for a in 0..self.n_nodes() {
            out.push_str(&format!(
                "let node{}= (pos:{})\n",
                a,
                self.node_store.node_data[NodeIndex(a)].pos.to_cetz()
            ));
            out.push_str(&format!("node(node{a}.pos)\n"));
        }

        for (eid, _, nid) in self.iter_edges() {
            out.push_str(&eid.cetz_impl(self, edge_label, edge_decoration, nid.orientation));
        }

        out.push_str("})\n");
        out
    }

    pub fn cetz_impl(
        &self,
        edge_label: &impl Fn(&E) -> String,
        edge_decoration: &impl Fn(&E) -> Decoration,
    ) -> String {
        let mut out = Self::cetz_preamble();
        out.push_str("#{\n");
        out.push_str(CETZ_PREAMBLE);
        out.push_str(&self.cetz_bare(edge_label, edge_decoration));
        out.push_str("}\n");
        out
    }
}

/// Represents the state and parameters for a graph layout optimization task,
/// typically used with a solver like `argmin`.
///
/// This struct holds references to the graph being laid out, the current (optimizable)
/// positions of its elements, fixed layout parameters (like spring constants), and
/// a random number generator for stochastic optimization algorithms.
///
/// # Type Parameters
/// - `'a`: Lifetime of the borrowed graph.
/// - `E`: The type of custom data associated with edges.
/// - `V`: The type of custom data associated with vertices.
/// - `H`: The type of custom data associated with half-edges.
pub struct GraphLayout<'a, E, V, H> {
    /// A reference to the graph whose layout is being computed.
    pub graph: &'a HedgeGraph<E, V, H, NodeStorageVec<V>>,
    /// The current set of optimizable and fixed positions for graph elements.
    pub positions: Positions,
    /// Parameters controlling the forces in a force-directed layout algorithm.
    pub params: LayoutParams,
    /// A random number generator, often used by stochastic optimization algorithms
    /// like simulated annealing.
    pub rng: Arc<Mutex<SmallRng>>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// Parameters controlling a force-directed graph layout algorithm.
///
/// These parameters define the strengths of various forces that act on nodes
/// and edge control points, such as spring forces between connected elements,
/// repulsive forces between all elements, and forces attracting elements
/// towards the center or influencing external edge placement.
pub struct LayoutParams {
    /// The spring constant (stiffness) for attractive forces along edges.
    pub spring_constant: f64,
    /// The ideal resting length for springs representing edges.
    pub spring_length: f64,
    /// The strength of a global repulsive force between all pairs of edges.
    pub global_edge_repulsion: f64,
    /// The strength of repulsive forces between edges and vertices.
    pub edge_vertex_repulsion: f64,
    /// The strength of electrostatic-like repulsive forces between edge control points.
    pub charge_constant_e: f64,
    /// The strength of electrostatic-like repulsive forces between vertices.
    pub charge_constant_v: f64,
    /// A constant influencing the placement of external (dangling) edges.
    pub external_constant: f64,
    /// The strength of a force attracting all elements towards the center of the layout.
    pub central_force_constant: f64,
}

impl Default for LayoutParams {
    fn default() -> Self {
        LayoutParams {
            spring_length: 1.,
            spring_constant: 0.5,
            global_edge_repulsion: 0.15,
            edge_vertex_repulsion: 1.5,
            charge_constant_e: 0.7,
            charge_constant_v: 13.2,
            external_constant: 0.0,
            central_force_constant: 0.0,
        }
    }
}

// LayoutParams {
//     spring_length: 1.,
//     spring_constant: 0.5,
//     global_edge_repulsion: 0.15,
//     edge_vertex_repulsion: 0.015,
//     charge_constant_e: 0.4,
//     charge_constant_v: 13.2,
//     external_constant: 0.0,
//     central_force_constant: 0.0,
// }

#[derive(Debug, Clone)]
#[allow(clippy::type_complexity)]
/// Manages the positional parameters for graph elements during layout optimization.
///
/// It distinguishes between positions that might be fixed (e.g., external edges pinned
/// to a circle) and those that are optimizable by the layout algorithm.
/// The actual coordinate values are stored in a flat `Vec<f64>` (the `param` argument
/// in `argmin`'s `CostFunction`), and this struct holds indices into that vector
/// for each vertex and edge control point.
///
/// Fields:
/// - `vertex_positions`: For each vertex, stores an optional fixed position `(x, y)`
///   or, if `None`, indices `(usize, usize)` into the optimizable parameter vector
///   for its x and y coordinates.
/// - `edge_positions`: For each edge (via `HedgeVec`), stores similar information
///   for its control point(s).
pub struct Positions {
    /// Stores position information for each vertex.
    /// Each element is a tuple: `(Option<(f64, f64)>, usize, usize)`.
    /// - `Option<(f64, f64)>`: If `Some`, this is a fixed (x, y) position for the vertex.
    /// - `usize`, `usize`: If the Option is `None`, these are indices into the flat
    ///   parameter vector (`Vec<f64>`) for the vertex's x and y coordinates, respectively.
    pub vertex_positions: Vec<(Option<(f64, f64)>, usize, usize)>,
    /// Stores position information for each edge's control point(s).
    /// Similar structure to `vertex_positions`, but wrapped in `HedgeVec` to be
    /// indexed by `EdgeIndex`.
    pub edge_positions: EdgeVec<(Option<(f64, f64)>, usize, usize)>,
}

impl Positions {
    pub fn scale(&mut self, _scale: f64) {
        // Don't scale pinned positions - they should remain at their absolute coordinates
        // Only the parameters for non-pinned positions get scaled in the layout function
    }
    pub fn max(&self, params: &[f64]) -> Option<f64> {
        // Only consider optimized parameters for scaling, not pinned positions
        // This keeps pinned positions at their absolute coordinates
        let vertex_max = self
            .vertex_positions
            .iter()
            .filter_map(|(v, i, j)| {
                if v.is_none() {
                    Some(params[*i].abs().max(params[*j].abs()))
                } else {
                    None // Skip pinned positions
                }
            })
            .reduce(f64::max);

        let mut max_edges = None;
        for (_, (a, i, j)) in self.edge_positions.iter() {
            if a.is_none() {
                let data = params[*i].abs().max(params[*j].abs());
                if let Some(max) = max_edges {
                    if max < data {
                        max_edges = Some(data);
                    }
                } else {
                    max_edges = Some(data);
                }
            } // Skip pinned edges
        }

        match (vertex_max, max_edges) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
    pub fn to_graph<E, V, H>(
        self,
        graph: HedgeGraph<E, V, H, NodeStorageVec<V>>,
        params: &[f64],
    ) -> PositionalHedgeGraph<E, V, H> {
        graph.map(
            |_, i, v| {
                let (x, y) = self.get_vertex_position(i, params);
                LayoutVertex::new(v, x, y)
            },
            |i, p, h, _, e| match h {
                HedgePair::Paired { source, sink } => {
                    let eid = i[h.any_hedge()];
                    let src = p.node_id_ref(source);
                    let sk = p.node_id_ref(sink);

                    let source_pos = self.get_vertex_position(src, params);
                    let source_pos = Vector2::new(source_pos.0, source_pos.1);
                    let sink_pos = self.get_vertex_position(sk, params);
                    let sink_pos = Vector2::new(sink_pos.0, sink_pos.1);
                    let pos = &self.get_edge_position(eid, params);

                    e.map(|d| LayoutEdge::new_internal(d, &source_pos, &sink_pos, pos.0, pos.1))
                }
                HedgePair::Split { source, sink, .. } => {
                    let eid = i[h.any_hedge()];
                    let src = p.node_id_ref(source);
                    let sk = p.node_id_ref(sink);

                    let source_pos = self.get_vertex_position(src, params);
                    let source_pos = Vector2::new(source_pos.0, source_pos.1);
                    let sink_pos = self.get_vertex_position(sk, params);
                    let sink_pos = Vector2::new(sink_pos.0, sink_pos.1);
                    let pos = &self.get_edge_position(eid, params);

                    e.map(|d| LayoutEdge::new_internal(d, &source_pos, &sink_pos, pos.0, pos.1))
                }
                HedgePair::Unpaired { hedge, flow } => {
                    let eid = i[h.any_hedge()];
                    let src = p.node_id_ref(hedge);

                    let source_pos = self.get_vertex_position(src, params);
                    let source_pos = Vector2::new(source_pos.0, source_pos.1);
                    let pos = &self.get_edge_position(eid, params);

                    e.map(|d| LayoutEdge::new_external(d, &source_pos, pos.0, pos.1, flow))
                }
            },
            |_, h| h,
        )
    }

    /// Create a Positions struct from pre-constructed components
    pub fn from_components(
        vertex_positions: Vec<(Option<(f64, f64)>, usize, usize)>,
        edge_positions: EdgeVec<(Option<(f64, f64)>, usize, usize)>,
    ) -> Self {
        Positions {
            vertex_positions,
            edge_positions,
        }
    }

    pub fn new<E, V, H>(
        graph: &HedgeGraph<E, V, H, NodeStorageVec<V>>,
        seed: u64,
    ) -> (Vec<f64>, Self) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut vertex_positions = Vec::new();
        let mut params = Vec::new();
        let ext_range = 2.0..4.0;
        let range = -1.0..1.0;

        let edge_positions = graph.new_edgevec(|_, _, pair| {
            let j = params.len();
            if pair.is_unpaired() {
                params.push(rng.gen_range(ext_range.clone()));
                params.push(rng.gen_range(ext_range.clone()));
            } else {
                params.push(rng.gen_range(range.clone()));
                params.push(rng.gen_range(range.clone()));
            }
            (None, j, j + 1)
        });

        for _ in graph.iter_node_ids() {
            let j = params.len();
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            vertex_positions.push((None, j, j + 1));
        }

        (
            params,
            Positions {
                vertex_positions,
                edge_positions,
            },
        )
    }
    pub fn circle_ext<E, V, H>(
        graph: &HedgeGraph<E, V, H, NodeStorageVec<V>>,
        seed: u64,
        radius: f64,
    ) -> (Vec<f64>, Self) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut vertex_positions = Vec::new();

        let range = -1.0..1.0;

        let mut params = Vec::new();

        let mut angle = 0.;
        let angle_step = 2. * std::f64::consts::PI / f64::from(graph.n_externals() as u32);

        let edge_positions = graph.new_edgevec(|_, _, pair| {
            let j = params.len();
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            if pair.is_unpaired() {
                let x = radius * f64::cos(angle);
                let y = radius * f64::sin(angle);
                angle += angle_step;
                (Some((x, y)), j, j + 1)
            } else {
                (None, j, j + 1)
            }
        });

        for _ in graph.iter_node_ids() {
            let j = params.len();
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            vertex_positions.push((None, j, j + 1));
        }

        (
            params,
            Positions {
                vertex_positions,
                edge_positions,
            },
        )
    }

    pub fn iter_vertex_positions<'a>(
        &'a self,
        params: &'a [f64],
    ) -> impl Iterator<Item = (NodeIndex, (f64, f64))> + 'a {
        self.vertex_positions
            .iter()
            .enumerate()
            .map(|(i, (p, x, y))| {
                if let Some(p) = p {
                    (NodeIndex(i), (p.0, p.1))
                } else {
                    (NodeIndex(i), (params[*x], params[*y]))
                }
            })
    }

    pub fn iter_edge_positions<'a>(
        &'a self,
        params: &'a [f64],
    ) -> impl Iterator<Item = (f64, f64)> + 'a {
        self.edge_positions.iter().map(|(_, (p, x, y))| {
            if let Some(p) = p {
                (p.0, p.1)
            } else {
                (params[*x], params[*y])
            }
        })
    }

    pub fn get_edge_position(&self, edge: EdgeIndex, params: &[f64]) -> (f64, f64) {
        let (p, ix, iy) = self.edge_positions[edge];
        if let Some(p) = p {
            (p.0, p.1)
        } else {
            (params[ix], params[iy])
        }
    }

    pub fn get_vertex_position(&self, vertex: NodeIndex, params: &[f64]) -> (f64, f64) {
        let (p, ix, iy) = self.vertex_positions[vertex.0];
        if let Some(p) = p {
            (p.0, p.1)
        } else {
            (params[ix], params[iy])
        }
    }
}

// State implementation for frostfire
#[derive(Clone)]
pub struct LayoutState {
    pub params: Vec<f64>,
    pub delta: f64,
}

impl State for LayoutState {
    fn neighbor(&self, rng: &mut impl Rng) -> Self {
        let mut new_params = self.params.clone();
        // Modify a random parameter
        let idx = rng.gen_range(0..new_params.len());
        let delta = rng.gen_range(-self.delta..self.delta);
        new_params[idx] += delta;
        LayoutState { params: new_params,delta:self.delta }
    }
}

// Energy implementation for frostfire
pub struct GraphLayoutEnergy<'a, E, V, H> {
    pub graph: &'a super::HedgeGraph<E, V, H, super::nodestore::NodeStorageVec<V>>,
    pub positions: &'a Positions,
    pub params: &'a LayoutParams,
}

impl<'a, E, V, H> Energy for GraphLayoutEnergy<'a, E, V, H> {
    type State = LayoutState;

    fn cost(&self, state: &Self::State) -> f64 {
        let param = &state.params;
        let mut cost = 0.0;

        // global edge repulsion:
        for (x, y) in self.positions.iter_edge_positions(param) {
            for (ex, ey) in self.positions.iter_edge_positions(param) {
                if ex == x && ey == y {
                    continue;
                }
                let dx = x - ex;
                let dy = y - ey;

                let dist_sq = dx * dx + dy * dy;
                let dist = (dist_sq + 1e-6).sqrt(); // Add epsilon for stability
                let repulsion = self.params.global_edge_repulsion / dist;
                cost += repulsion.min(1000.0); // Cap energy to prevent overflow
            }
        }

        for (node, (x, y)) in self.positions.iter_vertex_positions(param) {
            for (ex, ey) in self.positions.iter_edge_positions(param) {
                let dx = x - ex;
                let dy = y - ey;

                let dist_sq = dx * dx + dy * dy;
                let dist = (dist_sq + 1e-6).sqrt(); // Add epsilon for stability
                let repulsion = self.params.edge_vertex_repulsion / dist;
                cost += repulsion.min(1000.0); // Cap energy to prevent overflow
            }
            for e in self.graph.iter_crown(node) {
                let (ex, ey) = self.positions.get_edge_position(self.graph[&e], param);

                let dx = x - ex;
                let dy = y - ey;

                let dist_sq = dx * dx + dy * dy;
                let dist = (dist_sq + 1e-8).sqrt(); // Small epsilon for spring forces
                cost +=
                    0.5 * self.params.spring_constant * (self.params.spring_length - dist).powi(2);

                for othere in self.graph.iter_crown(node) {
                    let a = self.graph.inv(othere);
                    if e > othere && a != e {
                        let (ox, oy) = self.positions.get_edge_position(self.graph[&othere], param);
                        let dx = ex - ox;
                        let dy = ey - oy;
                        let dist_sq = dx * dx + dy * dy;
                        let dist = (dist_sq + 1e-6).sqrt(); // Add epsilon for stability
                        let charge = self.params.charge_constant_e / dist;
                        cost += charge.min(1000.0); // Cap energy to prevent overflow
                    }
                }
            }

            for (other, (ox, oy)) in self.positions.iter_vertex_positions(param) {
                if node != other {
                    let dx = x - ox;
                    let dy = y - oy;
                    let dist_sq = dx * dx + dy * dy;
                    let dist = (dist_sq + 1e-6).sqrt(); // Add epsilon for stability
                    let charge = 0.5 * self.params.charge_constant_v / dist;
                    cost += charge.min(1000.0); // Cap energy to prevent overflow

                    let dist_to_center = (ox * ox + oy * oy).sqrt();
                    if dist_to_center > 1.0 {
                        cost += 0.5 * self.params.central_force_constant / dist_to_center;
                    }
                }
            }
        }

        cost
    }
}

#[derive(Debug, Clone)]
/// Configuration settings for running a graph layout algorithm, typically
/// simulated annealing or another iterative optimization method.
///
/// This struct bundles the physical force parameters ([`LayoutParams`]),
/// the initial and optimizable positions ([`Positions`]), iteration controls
/// ([`LayoutIters`]), and the initial vector of parameters for the optimizer.
pub struct LayoutSettings {
    /// Parameters defining the forces in the layout model (e.g., spring stiffness, repulsion).
    params: LayoutParams,
    /// Initial and fixed positions for graph elements, and mapping to optimizable parameters.
    positions: Positions,
    /// Iteration control parameters for the layout algorithm (e.g., number of iterations, temperature).
    iters: LayoutIters,
    /// The initial flat vector of optimizable parameters (coordinates) passed to the solver.
    init_params: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// Specifies iteration parameters for a layout algorithm, particularly for
/// simulated annealing.
pub struct LayoutIters {
    /// The total number of iterations the layout algorithm should run.
    pub n_iters: u64,
    /// The initial "temperature" for simulated annealing, or a similar controlling
    /// parameter for other iterative solvers. It often influences the likelihood
    /// of accepting worse solutions, allowing escape from local minima.
    pub temp: f64,

    pub delta: f64,
    /// A seed for the random number generator used in stochastic layout algorithms,
    /// ensuring reproducibility.
    pub seed: u64,
}

impl LayoutSettings {
    pub fn new<E, V, H>(
        graph: &HedgeGraph<E, V, H, NodeStorageVec<V>>,
        params: LayoutParams,
        iters: LayoutIters,
    ) -> Self {
        let (init_params, positions) = Positions::new(graph, iters.seed);

        LayoutSettings {
            params,
            positions,
            iters,
            init_params,
        }
    }

    /// Create a LayoutSettings from pre-constructed components
    pub fn from_components(
        params: LayoutParams,
        positions: Positions,
        iters: LayoutIters,
        init_params: Vec<f64>,
    ) -> Self {
        LayoutSettings {
            params,
            positions,
            iters,
            init_params,
        }
    }

    /// Get the layout parameters
    pub fn params(&self) -> &LayoutParams {
        &self.params
    }

    /// Get the layout iterations settings
    pub fn iters(&self) -> &LayoutIters {
        &self.iters
    }

    /// Get the initialization parameters
    pub fn init_params(&self) -> &Vec<f64> {
        &self.init_params
    }

    pub fn left_right_square<E, V, H>(
        graph: &HedgeGraph<E, V, H, NodeStorageVec<V>>,
        params: LayoutParams,
        iters: LayoutIters,
        edge: f64,
        left: Vec<EdgeIndex>,
        right: Vec<EdgeIndex>,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(iters.seed);
        let mut vertex_positions = Vec::new();

        let range = -edge..edge;

        let mut init_params = Vec::new();

        let left_step = unsafe {
            if left.len() > 1 {
                edge / (left.len().unchecked_sub(1) as f64)
            } else {
                edge
            }
        };
        let mut left_bot_corner = if left.len() <= 1 {
            (-edge / 2., 0.)
        } else {
            (-edge / 2., -edge / 2.)
        };

        let right_step = unsafe {
            if right.len() > 1 {
                edge / (right.len().unchecked_sub(1) as f64)
            } else {
                edge
            }
        };
        let mut right_bot_corner = if right.len() <= 1 {
            (edge / 2., 0.)
        } else {
            (edge / 2., -edge / 2.)
        };

        let mut edge_positions = graph.new_edgevec(|_, _, _| {
            let j = init_params.len();
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            (None, j, j + 1)
        });

        for i in left {
            let (_, a, b) = edge_positions[i];
            edge_positions[i] = (Some(left_bot_corner), a, b);
            left_bot_corner.1 += left_step;
        }

        for i in right {
            let (_, a, b) = edge_positions[i];
            edge_positions[i] = (Some(right_bot_corner), a, b);
            right_bot_corner.1 += right_step;
        }

        for _ in graph.iter_node_ids() {
            let j = init_params.len();
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            vertex_positions.push((None, j, j + 1));
        }

        LayoutSettings {
            params,
            iters,
            positions: Positions {
                vertex_positions,
                edge_positions,
            },
            init_params,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn circle_ext<E, V>(
        graph: &HedgeGraph<E, V, NodeStorageVec<V>>,
        params: LayoutParams,
        iters: LayoutIters,
        angle_factors: Vec<u32>,
        _n_div: usize,
        shift: Rad<f64>,
        radius: f64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(iters.seed);
        let mut vertex_positions = Vec::new();

        let range = -1.0..1.0;

        let mut init_params = Vec::new();

        let angle_step = Rad(2. * std::f64::consts::PI / f64::from(graph.n_externals() as u32));

        let mut exti = 0;

        let edge_positions = graph.new_edgevec(|_, _, e| {
            let j = init_params.len();
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            if e.is_unpaired() {
                let angle = shift + angle_step * (angle_factors[exti] as f64);
                exti += 1;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                (Some((x, y)), j, j + 1)
            } else {
                (None, j, j + 1)
            }
        });

        for _ in graph.iter_node_ids() {
            let j = init_params.len();
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            vertex_positions.push((None, j, j + 1));
        }

        LayoutSettings {
            params,

            iters,

            positions: Positions {
                vertex_positions,
                edge_positions,
            },
            init_params,
        }
    }
}

impl<E, V, H> HedgeGraph<E, V, H, NodeStorageVec<V>> {
    pub fn layout(
        self,
        mut settings: LayoutSettings,
    ) -> HedgeGraph<LayoutEdge<E>, LayoutVertex<V>, H, NodeStorageVec<LayoutVertex<V>>> {

        let initial_state = LayoutState { params: settings.init_params,delta:settings.iters.delta };
        let energy = GraphLayoutEnergy {
            graph: &self,
            positions: &settings.positions,
            params: &settings.params,
        };
        let schedule = GeometricSchedule::new(settings.iters.temp, 0.95);
        let rng = StdRng::seed_from_u64(settings.iters.seed);




        let mut annealer = Annealer::new(
            initial_state,
            energy,
            schedule,
            rng,
            settings.iters.n_iters as usize,
        );

        let (best_state, _best_energy) = annealer.run();
        let best = best_state.params;

        // Check if we have any pinned positions
        let has_pins = settings.positions.vertex_positions.iter().any(|(pin, _, _)| pin.is_some()) ||
                      settings.positions.edge_positions.iter().any(|(_, (edge_data, _, _))| edge_data.is_some());

        if has_pins {
            // When pins are present, they establish the coordinate system scale
            // Don't rescale - keep pin coordinates at their absolute values
            settings.positions.to_graph(self, &best)
        } else {
            // When no pins are present, normalize to prevent coordinates from becoming too large
            let max = settings.positions.max(&best).unwrap_or(1.0);
            let best: Vec<_> = best.into_iter().map(|a| a / max).collect();
            settings.positions.scale(max);
            settings.positions.to_graph(self, &best)
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::str::FromStr;

    use crate::{
        dot,
        half_edge::drawing::Decoration,
        parser::{DotEdgeData, DotHedgeData, DotVertexData},
    };

    use super::{LayoutIters, LayoutParams, LayoutSettings, PositionalHedgeGraph};

    #[test]
    fn layout_simple() {
        let boxdiag = dot!(
            digraph{
                a->b->c->d->a
            }
        )
        .unwrap();

        let layout_settings = LayoutSettings::new(
            &boxdiag,
            LayoutParams::default(),
            LayoutIters {
                n_iters: 100,
                temp: 1.,
                seed: 1,
                delta: 0.1,
            },
        );
        boxdiag.graph.layout(layout_settings);
    }

    #[test]
    fn layout_sunshine() {
        let sunshine = dot!(
            digraph{
                0 [flow= source]
                1 [flow= sink]
                2 [flow= sink]
                a->0
                a->1
                2->a [dir=back]
                a->3
                a->4
                5->a[dir=back]
            }
        )
        .unwrap();

        println!("{}", sunshine.base_dot());

        let layout_settings = LayoutSettings::new(
            &sunshine,
            LayoutParams::default(),
            LayoutIters {
                n_iters: 30000,
                temp: 1.,
                seed: 1,
                delta: 0.1,
            },
        );
        let mut layout = sunshine.graph.layout(layout_settings);
        layout.to_fancy(&super::FancySettings {
            label_shift: 0.1,
            arrow_angle_percentage: Some(0.6),
            arrow_shift: 0.1,
        });

        let out = String::from_str("#set page(width: 35cm, height: auto)\n").unwrap()
            + PositionalHedgeGraph::<DotEdgeData, DotVertexData,DotHedgeData>::cetz_impl_collection(
                &[(
                    "".to_string(),
                    "".to_string(),
                    vec![("sunshine".to_string(), layout)],
                )],
                &|_| "a".to_string(),
                &|_| Decoration::Arrow,
                true,
            )
            .as_str();

        println!("{out}");
    }
}
