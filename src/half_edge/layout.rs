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

/// Represents different constraint types for a single coordinate (x or y).
#[derive(Debug, Clone, Copy)]
pub enum CoordinateConstraint {
    /// The coordinate is fixed to a specific value and won't be optimized.
    Fixed(f64),
    /// The coordinate is free to be optimized, with the given index in the parameter vector.
    Free(usize),
    /// The coordinate is linked to another coordinate at the same parameter index.
    /// This allows multiple elements to share the same coordinate value.
    Linked(usize),
}

/// Represents position constraints for a graph element (vertex or edge).
/// Allows independent control over x and y coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PositionConstraints {
    /// Constraint for the x coordinate.
    pub x: CoordinateConstraint,
    /// Constraint for the y coordinate.
    pub y: CoordinateConstraint,
}

impl PositionConstraints {
    /// Create constraints with both coordinates fixed.
    pub fn fixed(x: f64, y: f64) -> Self {
        Self {
            x: CoordinateConstraint::Fixed(x),
            y: CoordinateConstraint::Fixed(y),
        }
    }

    /// Create constraints with both coordinates free.
    pub fn free(x_index: usize, y_index: usize) -> Self {
        Self {
            x: CoordinateConstraint::Free(x_index),
            y: CoordinateConstraint::Free(y_index),
        }
    }

    /// Create constraints with x fixed and y free.
    pub fn fix_x(x: f64, y_index: usize) -> Self {
        Self {
            x: CoordinateConstraint::Fixed(x),
            y: CoordinateConstraint::Free(y_index),
        }
    }

    /// Create constraints with y fixed and x free.
    pub fn fix_y(x_index: usize, y: f64) -> Self {
        Self {
            x: CoordinateConstraint::Free(x_index),
            y: CoordinateConstraint::Fixed(y),
        }
    }

    /// Create constraints with both coordinates linked to other elements.
    pub fn linked(x_index: usize, y_index: usize) -> Self {
        Self {
            x: CoordinateConstraint::Linked(x_index),
            y: CoordinateConstraint::Linked(y_index),
        }
    }

    /// Create constraints with x linked and y free.
    pub fn link_x(x_index: usize, y_index: usize) -> Self {
        Self {
            x: CoordinateConstraint::Linked(x_index),
            y: CoordinateConstraint::Free(y_index),
        }
    }

    /// Create constraints with y linked and x free.
    pub fn link_y(x_index: usize, y_index: usize) -> Self {
        Self {
            x: CoordinateConstraint::Free(x_index),
            y: CoordinateConstraint::Linked(y_index),
        }
    }

    /// Get the coordinate value from the constraints and parameter vector.
    pub fn get_position(&self, params: &[f64]) -> (f64, f64) {
        let x = match self.x {
            CoordinateConstraint::Fixed(val) => val,
            CoordinateConstraint::Free(idx) | CoordinateConstraint::Linked(idx) => params[idx],
        };
        let y = match self.y {
            CoordinateConstraint::Fixed(val) => val,
            CoordinateConstraint::Free(idx) | CoordinateConstraint::Linked(idx) => params[idx],
        };
        (x, y)
    }
}

#[derive(Debug, Clone)]
/// Manages the positional parameters for graph elements during layout optimization.
///
/// It distinguishes between coordinates that might be fixed, free to optimize, or linked
/// to other coordinates. The actual coordinate values are stored in a flat `Vec<f64>`
/// (the `param` argument in optimization), and this struct holds constraints for each
/// vertex and edge control point.
pub struct Positions {
    /// Stores position constraints for each vertex.
    pub vertex_positions: Vec<PositionConstraints>,
    /// Stores position constraints for each edge's control point(s).
    pub edge_positions: EdgeVec<PositionConstraints>,
}

impl Positions {
    pub fn scale(&mut self, _scale: f64) {
        // Don't scale fixed positions - they should remain at their absolute coordinates
        // Only the parameters for free/linked positions get scaled in the layout function
    }

    pub fn max(&self, params: &[f64]) -> Option<f64> {
        // Only consider free parameters for scaling, not fixed positions
        let mut max_val = None;

        // Check vertex positions
        for constraints in &self.vertex_positions {
            for coord_constraint in [&constraints.x, &constraints.y] {
                if let CoordinateConstraint::Free(idx) | CoordinateConstraint::Linked(idx) = coord_constraint {
                    let val = params[*idx].abs();
                    max_val = Some(max_val.map_or(val, |current: f64| current.max(val)));
                }
            }
        }

        // Check edge positions
        for (_, constraints) in self.edge_positions.iter() {
            for coord_constraint in [&constraints.x, &constraints.y] {
                if let CoordinateConstraint::Free(idx) | CoordinateConstraint::Linked(idx) = coord_constraint {
                    let val = params[*idx].abs();
                    max_val = Some(max_val.map_or(val, |current: f64| current.max(val)));
                }
            }
        }

        max_val
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
                    let pos = self.get_edge_position(eid, params);

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
                    let pos = self.get_edge_position(eid, params);

                    e.map(|d| LayoutEdge::new_internal(d, &source_pos, &sink_pos, pos.0, pos.1))
                }
                HedgePair::Unpaired { hedge, flow } => {
                    let eid = i[h.any_hedge()];
                    let src = p.node_id_ref(hedge);

                    let source_pos = self.get_vertex_position(src, params);
                    let source_pos = Vector2::new(source_pos.0, source_pos.1);
                    let pos = self.get_edge_position(eid, params);

                    e.map(|d| LayoutEdge::new_external(d, &source_pos, pos.0, pos.1, flow))
                }
            },
            |_, h| h,
        )
    }

    /// Create a Positions struct from pre-constructed components
    pub fn from_components(
        vertex_positions: Vec<PositionConstraints>,
        edge_positions: EdgeVec<PositionConstraints>,
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
            let x_idx = params.len();
            let y_idx = params.len() + 1;
            if pair.is_unpaired() {
                params.push(rng.gen_range(ext_range.clone()));
                params.push(rng.gen_range(ext_range.clone()));
            } else {
                params.push(rng.gen_range(range.clone()));
                params.push(rng.gen_range(range.clone()));
            }
            PositionConstraints::free(x_idx, y_idx)
        });

        for _ in graph.iter_node_ids() {
            let x_idx = params.len();
            let y_idx = params.len() + 1;
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            vertex_positions.push(PositionConstraints::free(x_idx, y_idx));
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
            let x_idx = params.len();
            let y_idx = params.len() + 1;
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            if pair.is_unpaired() {
                let x = radius * f64::cos(angle);
                let y = radius * f64::sin(angle);
                angle += angle_step;
                PositionConstraints::fixed(x, y)
            } else {
                PositionConstraints::free(x_idx, y_idx)
            }
        });

        for _ in graph.iter_node_ids() {
            let x_idx = params.len();
            let y_idx = params.len() + 1;
            params.push(rng.gen_range(range.clone()));
            params.push(rng.gen_range(range.clone()));
            vertex_positions.push(PositionConstraints::free(x_idx, y_idx));
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
            .map(|(i, constraints)| {
                (NodeIndex(i), constraints.get_position(params))
            })
    }

    pub fn iter_edge_positions<'a>(
        &'a self,
        params: &'a [f64],
    ) -> impl Iterator<Item = (f64, f64)> + 'a {
        self.edge_positions.iter().map(|(_, constraints)| {
            constraints.get_position(params)
        })
    }

    pub fn get_edge_position(&self, edge: EdgeIndex, params: &[f64]) -> (f64, f64) {
        self.edge_positions[edge].get_position(params)
    }

    pub fn get_vertex_position(&self, vertex: NodeIndex, params: &[f64]) -> (f64, f64) {
        self.vertex_positions[vertex.0].get_position(params)
    }

    /// Set a vertex to have a fixed x coordinate while keeping y free to optimize
    pub fn fix_vertex_x(&mut self, vertex: NodeIndex, x: f64) {
        let constraints = &mut self.vertex_positions[vertex.0];
        constraints.x = CoordinateConstraint::Fixed(x);
    }

    /// Set a vertex to have a fixed y coordinate while keeping x free to optimize
    pub fn fix_vertex_y(&mut self, vertex: NodeIndex, y: f64) {
        let constraints = &mut self.vertex_positions[vertex.0];
        constraints.y = CoordinateConstraint::Fixed(y);
    }

    /// Link two vertices to share the same x coordinate
    pub fn link_vertices_x(&mut self, vertex1: NodeIndex, vertex2: NodeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared x coordinate
        let shared_x_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.vertex_positions[vertex1.0].x = CoordinateConstraint::Linked(shared_x_idx);
        self.vertex_positions[vertex2.0].x = CoordinateConstraint::Linked(shared_x_idx);
    }

    /// Link two vertices to share the same y coordinate
    pub fn link_vertices_y(&mut self, vertex1: NodeIndex, vertex2: NodeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared y coordinate
        let shared_y_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.vertex_positions[vertex1.0].y = CoordinateConstraint::Linked(shared_y_idx);
        self.vertex_positions[vertex2.0].y = CoordinateConstraint::Linked(shared_y_idx);
    }

    /// Set an edge to have a fixed x coordinate while keeping y free to optimize
    pub fn fix_edge_x(&mut self, edge: EdgeIndex, x: f64) {
        self.edge_positions[edge].x = CoordinateConstraint::Fixed(x);
    }

    /// Set an edge to have a fixed y coordinate while keeping x free to optimize
    pub fn fix_edge_y(&mut self, edge: EdgeIndex, y: f64) {
        self.edge_positions[edge].y = CoordinateConstraint::Fixed(y);
    }

    /// Link two edges to share the same x coordinate
    pub fn link_edges_x(&mut self, edge1: EdgeIndex, edge2: EdgeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared x coordinate
        let shared_x_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.edge_positions[edge1].x = CoordinateConstraint::Linked(shared_x_idx);
        self.edge_positions[edge2].x = CoordinateConstraint::Linked(shared_x_idx);
    }

    /// Link two edges to share the same y coordinate
    pub fn link_edges_y(&mut self, edge1: EdgeIndex, edge2: EdgeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared y coordinate
        let shared_y_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.edge_positions[edge1].y = CoordinateConstraint::Linked(shared_y_idx);
        self.edge_positions[edge2].y = CoordinateConstraint::Linked(shared_y_idx);
    }

    /// Link a vertex and an edge to share the same x coordinate
    pub fn link_vertex_edge_x(&mut self, vertex: NodeIndex, edge: EdgeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared x coordinate
        let shared_x_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.vertex_positions[vertex.0].x = CoordinateConstraint::Linked(shared_x_idx);
        self.edge_positions[edge].x = CoordinateConstraint::Linked(shared_x_idx);
    }

    /// Link a vertex and an edge to share the same y coordinate
    pub fn link_vertex_edge_y(&mut self, vertex: NodeIndex, edge: EdgeIndex, params: &mut Vec<f64>) {
        // Create a new parameter index for the shared y coordinate
        let shared_y_idx = params.len();
        params.push(0.0); // Initialize with default value

        self.vertex_positions[vertex.0].y = CoordinateConstraint::Linked(shared_y_idx);
        self.edge_positions[edge].y = CoordinateConstraint::Linked(shared_y_idx);
    }

    /// Create a horizontal rail constraint for multiple vertices at the same y coordinate
    pub fn create_horizontal_rail(&mut self, vertices: &[NodeIndex], y: f64, params: &mut Vec<f64>) {
        if vertices.is_empty() { return; }

        if vertices.len() == 1 {
            // Single vertex - just fix its y coordinate
            self.fix_vertex_y(vertices[0], y);
        } else {
            // Multiple vertices - create a shared parameter for y coordinate
            let shared_y_idx = params.len();
            params.push(y); // Initialize with the rail position

            for &vertex in vertices {
                self.vertex_positions[vertex.0].y = CoordinateConstraint::Linked(shared_y_idx);
            }
        }
    }

    /// Create a vertical rail constraint for multiple vertices at the same x coordinate
    pub fn create_vertical_rail(&mut self, vertices: &[NodeIndex], x: f64, params: &mut Vec<f64>) {
        if vertices.is_empty() { return; }

        if vertices.len() == 1 {
            // Single vertex - just fix its x coordinate
            self.fix_vertex_x(vertices[0], x);
        } else {
            // Multiple vertices - create a shared parameter for x coordinate
            let shared_x_idx = params.len();
            params.push(x); // Initialize with the rail position

            for &vertex in vertices {
                self.vertex_positions[vertex.0].x = CoordinateConstraint::Linked(shared_x_idx);
            }
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
                cost += repulsion;//.min(1000.0); // Cap energy to prevent overflow
            }
        }

        for (node, (x, y)) in self.positions.iter_vertex_positions(param) {
            for (ex, ey) in self.positions.iter_edge_positions(param) {
                let dx = x - ex;
                let dy = y - ey;

                let dist_sq = dx * dx + dy * dy;
                let dist = (dist_sq + 1e-6).sqrt(); // Add epsilon for stability
                let repulsion = self.params.edge_vertex_repulsion / dist;
                cost += repulsion;//.min(1000.0); // Cap energy to prevent overflow
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
                    cost += charge;//.min(1000.0); // Cap energy to prevent overflow

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

    /// Get mutable access to positions for adding constraints
    pub fn positions_mut(&mut self) -> &mut Positions {
        &mut self.positions
    }

    /// Get mutable access to init_params for constraint modifications
    pub fn init_params_mut(&mut self) -> &mut Vec<f64> {
        &mut self.init_params
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
            let x_idx = init_params.len();
            let y_idx = init_params.len() + 1;
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            PositionConstraints::free(x_idx, y_idx)
        });

        for i in left {
            edge_positions[i] = PositionConstraints::fixed(left_bot_corner.0, left_bot_corner.1);
            left_bot_corner.1 += left_step;
        }

        for i in right {
            edge_positions[i] = PositionConstraints::fixed(right_bot_corner.0, right_bot_corner.1);
            right_bot_corner.1 += right_step;
        }

        for _ in graph.iter_node_ids() {
            let x_idx = init_params.len();
            let y_idx = init_params.len() + 1;
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            vertex_positions.push(PositionConstraints::free(x_idx, y_idx));
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
            let x_idx = init_params.len();
            let y_idx = init_params.len() + 1;
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            if e.is_unpaired() {
                let angle = shift + angle_step * (angle_factors[exti] as f64);
                exti += 1;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                PositionConstraints::fixed(x, y)
            } else {
                PositionConstraints::free(x_idx, y_idx)
            }
        });

        for _ in graph.iter_node_ids() {
            let x_idx = init_params.len();
            let y_idx = init_params.len() + 1;
            init_params.push(rng.gen_range(range.clone()));
            init_params.push(rng.gen_range(range.clone()));
            vertex_positions.push(PositionConstraints::free(x_idx, y_idx));
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

        // Check if we have any fixed positions
        let has_pins = settings.positions.vertex_positions.iter().any(|constraints| {
            matches!(constraints.x, CoordinateConstraint::Fixed(_)) ||
            matches!(constraints.y, CoordinateConstraint::Fixed(_))
        }) || settings.positions.edge_positions.iter().any(|(_, constraints)| {
            matches!(constraints.x, CoordinateConstraint::Fixed(_)) ||
            matches!(constraints.y, CoordinateConstraint::Fixed(_))
        });

        if has_pins {
            // When fixed coordinates are present, they establish the coordinate system scale
            // Don't rescale - keep fixed coordinates at their absolute values
            settings.positions.to_graph(self, &best)
        } else {
            // When no fixed coordinates are present, normalize to prevent coordinates from becoming too large
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

    #[test]
    fn test_coordinate_constraints() {
        let simple_graph = dot!(
            digraph{
                a->b->c->a
            }
        )
        .unwrap();

        // Create layout settings with constraints
        let mut layout_settings = LayoutSettings::new(
            &simple_graph,
            LayoutParams::default(),
            LayoutIters {
                n_iters: 1000,
                temp: 1.,
                seed: 42,
                delta: 0.1,
            },
        );

        // Example 1: Fix vertex 'a' to have x = 0.0 (acts like on a vertical rail)
        layout_settings.positions_mut().fix_vertex_x(NodeIndex(0), 0.0);

        // Example 2: Fix vertex 'b' to have y = 1.0 (acts like on a horizontal rail)
        layout_settings.positions_mut().fix_vertex_y(NodeIndex(1), 1.0);

        // Example 3: Link vertices 'b' and 'c' to share the same x coordinate
        layout_settings.positions_mut().link_vertices_x(
            NodeIndex(1),
            NodeIndex(2),
            layout_settings.init_params_mut()
        );

        let layout = simple_graph.graph.layout(layout_settings);

        // Verify constraints are satisfied
        let vertex_a_pos = layout.node_store.node_data[NodeIndex(0)].pos();
        let vertex_b_pos = layout.node_store.node_data[NodeIndex(1)].pos();
        let vertex_c_pos = layout.node_store.node_data[NodeIndex(2)].pos();

        // Check that vertex 'a' has x = 0.0
        assert!((vertex_a_pos.x - 0.0).abs() < 1e-10, "Vertex 'a' x should be fixed at 0.0");

        // Check that vertex 'b' has y = 1.0
        assert!((vertex_b_pos.y - 1.0).abs() < 1e-10, "Vertex 'b' y should be fixed at 1.0");

        // Check that vertices 'b' and 'c' share the same x coordinate
        assert!((vertex_b_pos.x - vertex_c_pos.x).abs() < 1e-10,
                "Vertices 'b' and 'c' should share the same x coordinate");

        println!("Constraint test passed!");
        println!("Vertex A: ({}, {})", vertex_a_pos.x, vertex_a_pos.y);
        println!("Vertex B: ({}, {})", vertex_b_pos.x, vertex_b_pos.y);
        println!("Vertex C: ({}, {})", vertex_c_pos.x, vertex_c_pos.y);
    }

    #[test]
    fn test_edge_constraints() {
        let graph_with_externals = dot!(
            digraph{
                0 [flow= source]
                1 [flow= sink]
                a->0
                a->1
                a->b
            }
        )
        .unwrap();

        let mut layout_settings = LayoutSettings::new(
            &graph_with_externals,
            LayoutParams::default(),
            LayoutIters {
                n_iters: 1000,
                temp: 1.,
                seed: 42,
                delta: 0.1,
            },
        );

        // Find edge indices (this would typically be done based on your graph structure)
        let mut external_edges = Vec::new();
        let mut internal_edges = Vec::new();

        for (edge_idx, _, _) in graph_with_externals.iter_edges() {
            if edge_idx.is_unpaired() {
                external_edges.push(graph_with_externals[&edge_idx.any_hedge()]);
            } else {
                internal_edges.push(graph_with_externals[&edge_idx.any_hedge()]);
            }
        }

        // Fix an external edge to have y = 2.0 (horizontal rail)
        if let Some(&first_external) = external_edges.first() {
            layout_settings.positions_mut().fix_edge_y(first_external, 2.0);
        }

        // Link two edges to share the same x coordinate if we have enough edges
        if external_edges.len() >= 2 {
            layout_settings.positions_mut().link_edges_x(
                external_edges[0],
                external_edges[1],
                layout_settings.init_params_mut()
            );
        }

        let layout = graph_with_externals.graph.layout(layout_settings);

        println!("Edge constraint test completed!");

        // Print edge positions for verification
        for (edge_idx, _, _) in layout.iter_edges() {
            let edge_data = layout.get_edge_data(edge_idx.any_hedge());
            println!("Edge position: ({}, {})", edge_data.pos().x, edge_data.pos().y);
        }
    }
}
