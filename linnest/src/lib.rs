pub mod geom;
use bitvec::vec::BitVec;
use cgmath::{EuclideanSpace, Point2, Rad, Vector2, Zero};
use linnet::{
    half_edge::{
        involution::{EdgeData, EdgeIndex, EdgeVec, Flow, Hedge, HedgePair, Involution},
        layout::{
            simulatedanneale::{anneal, GeoSchedule, SAConfig},
            spring::{
                Constraints, LayoutNeighbor, ParamTuning, Shiftable, SpringChargeEnergy, XYorBOTH,
            },
            PositionConstraints,
        },
        nodestore::NodeStorageOps,
        subgraph::{Inclusion, SubGraph, SubGraphOps},
        tree::SimpleTraversalTree,
        EdgeAccessors, HedgeGraph, NodeIndex, NodeVec,
    },
    parser::{
        set::DotGraphSet, DotEdgeData, DotGraph, DotHedgeData, DotVertexData, GlobalData,
        HedgeParseError,
    },
    tree::child_pointer::ParentChildStore,
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum PinConstraint {
    /// Pin both coordinates to fixed values: pin="1.0,2.0"
    Fixed(f64, f64),
    /// Fix only x coordinate: pin="x:1.0"
    FixX(f64),
    /// Fix only y coordinate: pin="y:2.0"
    FixY(f64),
    /// Link x coordinate to a named group: pin="x:@group1"
    LinkX(String),
    /// Link y coordinate to a named group: pin="y:@group1"
    LinkY(String),
    /// Link both coordinates to a named group: pin="@group1"
    LinkBoth(String),
    /// Combine constraints: pin="x:1.0,y:@group1"
    Combined(Box<PinConstraint>, Box<PinConstraint>),
}

impl PinConstraint {
    pub fn point_constraint(
        &self,
        index: usize,
        map: &mut HashMap<String, usize>,
    ) -> (Point2<f64>, Constraints) {
        match self {
            PinConstraint::Fixed(x, y) => (Point2::new(*x, *y), Constraints::Fixed(XYorBOTH::Both)),
            PinConstraint::FixX(x) => (Point2::new(*x, 0.0), Constraints::Fixed(XYorBOTH::X)),
            PinConstraint::FixY(y) => (Point2::new(0.0, *y), Constraints::Fixed(XYorBOTH::Y)),
            PinConstraint::LinkX(group) => {
                let reference = *map
                    .entry(format!("link_x_{}", group))
                    .or_insert_with(|| index);
                (
                    Point2::new(0.0, 0.0),
                    if reference == index {
                        Constraints::Free
                    } else {
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::X,
                        }
                    },
                )
            }
            PinConstraint::LinkY(group) => {
                let reference = *map
                    .entry(format!("link_y_{}", group))
                    .or_insert_with(|| index);
                (
                    Point2::new(0.0, 0.0),
                    if reference == index {
                        Constraints::Free
                    } else {
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::Y,
                        }
                    },
                )
            }
            PinConstraint::LinkBoth(group) => {
                let reference = *map
                    .entry(format!("link_{}", group))
                    .or_insert_with(|| index);
                (
                    Point2::new(0.0, 0.0),
                    if reference == index {
                        Constraints::Free
                    } else {
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::Both,
                        }
                    },
                )
            }
            PinConstraint::Combined(x_constraint, y_constraint) => {
                let (mut pos, constraintx) = x_constraint.point_constraint(index, map);

                let (pos_y, constrainty) = y_constraint.point_constraint(index, map);

                pos.y = pos_y.y;
                match (constraintx, constrainty) {
                    (Constraints::Free, a) => (pos, a),
                    (a, Constraints::Free) => (pos, a),

                    (Constraints::Fixed(XYorBOTH::X), Constraints::Fixed(XYorBOTH::Y)) => {
                        (pos, Constraints::Fixed(XYorBOTH::Both))
                    }
                    (
                        Constraints::Fixed(XYorBOTH::X),
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::Y,
                        },
                    ) => (
                        pos,
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::Y,
                        },
                    ),
                    (
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::X,
                        },
                        Constraints::Fixed(XYorBOTH::Y),
                    ) => (
                        pos,
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::X,
                        },
                    ),
                    (
                        Constraints::Grouped {
                            reference,
                            axis: XYorBOTH::X,
                        },
                        Constraints::Grouped {
                            reference: referencey,
                            axis: XYorBOTH::Y,
                        },
                    ) => {
                        if reference == referencey {
                            (
                                pos,
                                Constraints::Grouped {
                                    reference,
                                    axis: XYorBOTH::Both,
                                },
                            )
                        } else {
                            (pos, Constraints::Free)
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn parse(input: &str) -> Option<Self> {
        let input = input
            .trim()
            .trim_matches('"')
            .trim_matches(|c| c == '(' || c == ')');

        // Handle @group syntax for linking both coordinates
        if input.starts_with('@') {
            return Some(PinConstraint::LinkBoth(input[1..].to_string()));
        }

        // Handle x:value,y:value syntax or fixed coordinates (comma or space separated)
        if input.contains(',') || input.split_whitespace().count() == 2 {
            let parts: Vec<&str> = if input.contains(',') {
                input.split(',').map(|s| s.trim()).collect()
            } else {
                input.split_whitespace().collect()
            };

            if parts.len() == 2 {
                // Try to parse as coordinate constraints
                let x_constraint = Self::parse_single_constraint(parts[0]);
                let y_constraint = Self::parse_single_constraint(parts[1]);

                match (x_constraint, y_constraint) {
                    (Some(x), Some(y)) => {
                        return Some(PinConstraint::Combined(Box::new(x), Box::new(y)))
                    }
                    _ => {
                        // Fall back to parsing as fixed coordinates
                        if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                            return Some(PinConstraint::Fixed(x, y));
                        }
                    }
                }
            }
        }

        // Handle single constraint
        Self::parse_single_constraint(input)
    }

    fn parse_single_constraint(input: &str) -> Option<Self> {
        let input = input.trim();

        if input.starts_with("x:") {
            let value = &input[2..];
            if value.starts_with('@') {
                Some(PinConstraint::LinkX(value[1..].to_string()))
            } else if let Ok(x) = value.parse::<f64>() {
                Some(PinConstraint::FixX(x))
            } else {
                None
            }
        } else if input.starts_with("y:") {
            let value = &input[2..];
            if value.starts_with('@') {
                Some(PinConstraint::LinkY(value[1..].to_string()))
            } else if let Ok(y) = value.parse::<f64>() {
                Some(PinConstraint::FixY(y))
            } else {
                None
            }
        } else if input.starts_with('@') {
            Some(PinConstraint::LinkBoth(input[1..].to_string()))
        } else {
            None
        }
    }

    pub fn to_position_constraints<F, R>(
        &self,
        params: &mut Vec<f64>,
        get_link_index: &mut F,
        rng: &mut R,
        is_external: bool,
    ) -> PositionConstraints
    where
        F: FnMut(&str, &mut Vec<f64>, &mut R) -> usize,
        R: Rng,
    {
        let range = if is_external { 2.0..4.0 } else { -1.0..1.0 };

        match self {
            PinConstraint::Fixed(x, y) => PositionConstraints::fixed(*x, *y),
            PinConstraint::FixX(x) => {
                let y_idx = params.len();
                params.push(rng.gen_range(range));
                PositionConstraints::fix_x(*x, y_idx)
            }
            PinConstraint::FixY(y) => {
                let x_idx = params.len();
                params.push(rng.gen_range(range));
                PositionConstraints::fix_y(x_idx, *y)
            }
            PinConstraint::LinkX(group) => {
                let x_idx = get_link_index(group, params, rng);
                let y_idx = params.len();
                params.push(rng.gen_range(range));
                PositionConstraints::link_x(x_idx, y_idx)
            }
            PinConstraint::LinkY(group) => {
                let x_idx = params.len();
                params.push(rng.gen_range(range));
                let y_idx = get_link_index(group, params, rng);
                PositionConstraints::link_y(x_idx, y_idx)
            }
            PinConstraint::LinkBoth(group) => {
                let x_idx = get_link_index(&format!("{}_x", group), params, rng);
                let y_idx = get_link_index(&format!("{}_y", group), params, rng);
                PositionConstraints::linked(x_idx, y_idx)
            }
            PinConstraint::Combined(x_constraint, y_constraint) => {
                let x_pos =
                    x_constraint.to_position_constraints(params, get_link_index, rng, is_external);
                let y_pos =
                    y_constraint.to_position_constraints(params, get_link_index, rng, is_external);
                PositionConstraints {
                    x: x_pos.x,
                    y: y_pos.y,
                }
            }
        }
    }
}

impl std::fmt::Display for PinConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinConstraint::Fixed(x, y) => write!(f, "{},{}", x, y),
            PinConstraint::FixX(x) => write!(f, "x:{}", x),
            PinConstraint::FixY(y) => write!(f, "y:{}", y),
            PinConstraint::LinkX(group) => write!(f, "x:@{}", group),
            PinConstraint::LinkY(group) => write!(f, "y:@{}", group),
            PinConstraint::LinkBoth(group) => write!(f, "@{}", group),
            PinConstraint::Combined(x, y) => write!(f, "{},{}", x, y),
        }
    }
}

use rand::{rngs::SmallRng, Rng};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{collections::BTreeMap, io::BufWriter};
use std::{collections::HashMap, ops::Deref};
use std::{f64, fs::File};

use wasm_minimal_protocol::*;

initiate_protocol!();

// Custom getrandom implementation for WASM
#[cfg(feature = "custom")]
use getrandom::register_custom_getrandom;

use crate::geom::{tangent_angle_toward_c_side, GeomError};

/// Expand template placeholders in a string using values from statements
/// This function supports {key} placeholders that get replaced with values from the statements map
pub fn expand_template(
    template: &str,
    statements: &std::collections::BTreeMap<String, String>,
) -> String {
    let mut result = template.to_string();

    // Find all placeholders in format {key}
    let mut chars = template.chars().peekable();
    let mut placeholders = Vec::new();
    let mut current_pos = 0;

    while let Some(ch) = chars.next() {
        if ch == '{' {
            let start = current_pos;
            let mut key = String::new();
            let mut found_closing = false;

            while let Some(inner_ch) = chars.next() {
                current_pos += inner_ch.len_utf8();
                if inner_ch == '}' {
                    found_closing = true;
                    break;
                } else if inner_ch == '{' {
                    // Nested braces, ignore this placeholder
                    break;
                } else {
                    key.push(inner_ch);
                }
            }

            if found_closing && !key.is_empty() {
                placeholders.push((start, current_pos + 1, key));
            }
        }
        current_pos += ch.len_utf8();
    }

    // Replace placeholders in reverse order to maintain positions
    for (start, end, key) in placeholders.iter().rev() {
        if let Some(value) = statements.get(key) {
            // Remove quotes from the replacement value
            let clean_value = value.trim().trim_matches('"');
            result.replace_range(*start..*end, clean_value);
        }
    }

    result
}

#[cfg(feature = "custom")]
fn custom_getrandom(buf: &mut [u8]) -> Result<(), getrandom::Error> {
    // Simple deterministic implementation for WASM
    // In production, you might want a better source of entropy
    static mut COUNTER: u64 = 42;
    unsafe {
        for chunk in buf.chunks_mut(8) {
            let bytes = COUNTER.to_le_bytes();
            for (i, &byte) in bytes.iter().enumerate() {
                if i < chunk.len() {
                    chunk[i] = byte;
                }
            }
            COUNTER = COUNTER.wrapping_add(1);
        }
    }
    Ok(())
}

#[cfg(feature = "custom")]
register_custom_getrandom!(custom_getrandom);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TypstNode {
    pos: Point2<f64>,
    constraints: Constraints,
    shift: Option<Vector2<f64>>,
    eval: Option<String>,
}

impl Default for TypstNode {
    fn default() -> Self {
        TypstNode {
            pos: Point2::origin(),
            constraints: Constraints::Free,
            shift: None,
            eval: None,
        }
    }
}

impl Shiftable for TypstNode {
    fn shift<I: From<usize> + PartialEq + Copy, R: std::ops::IndexMut<I, Output = Point2<f64>>>(
        &self,
        shift: Vector2<f64>,
        index: I,
        values: &mut R,
    ) -> bool {
        self.constraints.shift(shift, index, values)
    }
}

impl TypstNode {
    /// Convert back to DotVertexData
    fn to_dot(&self) -> DotVertexData {
        let mut statements = std::collections::BTreeMap::new();

        // Add position as pos attribute
        statements.insert("pos".to_string(), format!("{},{}", self.pos.x, self.pos.y));

        if let Some(s) = self.shift {
            statements.insert("shift".to_string(), format!("{},{}", s.x, s.y));
        }

        DotVertexData {
            name: None,
            index: None,
            statements,
        }
    }

    fn parse(
        _inv: &Involution,
        nid: NodeIndex,
        data: DotVertexData,
        init_points: &NodeVec<(Point2<f64>, Constraints)>,
    ) -> Self {
        let shift =
            Self::parse_position(&data.statements, "shift").map(|(x, y)| Vector2::new(x, y));

        let mut eval: Option<String> = data.get("eval").transpose().unwrap();

        // Apply template expansion and clean quotes
        eval = eval.map(|template| {
            let clean_template = template
                .strip_prefix('"')
                .unwrap_or(&template)
                .strip_suffix('"')
                .unwrap_or(&template);
            expand_template(clean_template, &data.statements)
        });

        let (pos, constraints) = init_points[nid];

        Self {
            pos,
            constraints,
            shift,
            eval,
        }
    }

    fn parse_position(
        statements: &std::collections::BTreeMap<String, String>,
        attr: &str,
    ) -> Option<(f64, f64)> {
        if let Some(value) = statements.get(attr) {
            // Remove outer quotes from DOT parsing: "\"1.0,2.0\"" -> "1.0,2.0"
            let unquoted = value.trim().trim_matches('"');
            // Parse formats like "1.0,2.0" or "1.0 2.0" or "(1.0,2.0)"
            let cleaned = unquoted.trim().trim_matches(|c| c == '(' || c == ')');
            let parts: Vec<&str> = cleaned
                .split(|c| c == ',' || c == ' ')
                .filter(|s| !s.is_empty())
                .collect();

            if parts.len() == 2 {
                if let (Ok(x), Ok(y)) = (
                    parts[0].trim().parse::<f64>(),
                    parts[1].trim().parse::<f64>(),
                ) {
                    return Some((x, y));
                }
            }
        }
        None
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TypstEdge {
    from: Option<NodeIndex>,
    to: Option<NodeIndex>,
    bend: Result<Rad<f64>, GeomError>,
    pos: Point2<f64>,
    eval: Option<String>,
    mom_eval: Option<String>,
    shift: Option<Vector2<f64>>,
    pub constraints: Constraints,
}
impl Shiftable for TypstEdge {
    fn shift<I: From<usize> + PartialEq + Copy, R: std::ops::IndexMut<I, Output = Point2<f64>>>(
        &self,
        shift: Vector2<f64>,
        index: I,
        values: &mut R,
    ) -> bool {
        self.constraints.shift(shift, index, values)
    }
}
impl Default for TypstEdge {
    fn default() -> Self {
        Self {
            from: None,
            to: None,
            bend: Err(GeomError::NotComputed),
            pos: Point2::origin(),
            shift: None,
            eval: None,
            mom_eval: None,
            constraints: Constraints::Free,
        }
    }
}

impl TypstEdge {
    fn parse<N: NodeStorageOps>(
        _inv: &Involution,
        node_store: &N,
        p: HedgePair,
        eid: EdgeIndex,
        data: EdgeData<DotEdgeData>,
        pin_constaints: &EdgeVec<(Point2<f64>, Constraints)>,
    ) -> EdgeData<Self> {
        data.map(|d| {
            let shift =
                TypstNode::parse_position(&d.statements, "shift").map(|(x, y)| Vector2::new(x, y));

            let mut eval: Option<String> = d.get("eval").transpose().unwrap();

            // Apply template expansion and clean quotes for eval
            eval = eval.map(|template| {
                let clean_template = template
                    .strip_prefix('"')
                    .unwrap_or(&template)
                    .strip_suffix('"')
                    .unwrap_or(&template);
                expand_template(clean_template, &d.statements)
            });

            let mut mom_eval: Option<String> = d.get("mom_eval").transpose().unwrap();

            // Apply template expansion and clean quotes for mom_eval
            mom_eval = mom_eval.map(|template| {
                let clean_template = template
                    .strip_prefix('"')
                    .unwrap_or(&template)
                    .strip_suffix('"')
                    .unwrap_or(&template);
                expand_template(clean_template, &d.statements)
            });
            let mut from = None;
            let mut to = None;
            match p {
                HedgePair::Split { source, sink, .. } | HedgePair::Paired { source, sink } => {
                    from = Some(node_store.node_id_ref(source));
                    to = Some(node_store.node_id_ref(sink));
                }
                HedgePair::Unpaired {
                    hedge,
                    flow: Flow::Source,
                } => {
                    from = Some(node_store.node_id_ref(hedge));
                }

                HedgePair::Unpaired {
                    hedge,
                    flow: Flow::Sink,
                } => {
                    to = Some(node_store.node_id_ref(hedge));
                }
            }

            let (pos, constraints) = pin_constaints[eid];

            Self {
                from,
                to,
                pos,
                constraints,
                eval,
                mom_eval,
                shift,
                ..Default::default()
            }
        })
    }

    /// Convert back to DotEdgeData
    fn to_dot(&self) -> DotEdgeData {
        let mut statements = std::collections::BTreeMap::new();

        // Add position as pos attribute
        statements.insert("pos".to_string(), format!("{},{}", self.pos.x, self.pos.y));

        // Add shift if present
        if let Some(s) = self.shift {
            statements.insert("shift".to_string(), format!("{},{}", s.x, s.y));
        }

        // Add bend if non-default
        if let Ok(b) = self.bend {
            statements.insert("bend".to_string(), format!("{}rad", b.0));
        }

        if let Some(eval) = &self.eval {
            statements.insert("eval".to_string(), eval.clone());
        }

        if let Some(eval) = &self.mom_eval {
            statements.insert("mom_eval".to_string(), eval.clone());
        }

        DotEdgeData {
            statements,
            edge_id: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TypstHedge {
    from: usize,
    to: usize,
    weight: f64,
}

impl TypstHedge {
    /// Convert back to DotHedgeData
    fn to_dot(&self) -> DotHedgeData {
        let statement = if self.weight != 0.0 {
            Some(format!("weight={}", self.weight))
        } else {
            None
        };

        DotHedgeData {
            statement,
            id: None,
            port_label: None,
            compasspt: None,
        }
    }

    fn parse(_h: Hedge, _data: DotHedgeData) -> Self {
        Self::default()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypstGraph {
    graph: HedgeGraph<TypstEdge, TypstNode, TypstHedge>,
    spring_params: ParamTuning,
    schedule: GeoSchedule,
    step: Option<f64>,
    temp: Option<f64>,
    seed: Option<u64>,
}

impl Deref for TypstGraph {
    type Target = HedgeGraph<TypstEdge, TypstNode, TypstHedge>;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl From<DotGraph> for TypstGraph {
    fn from(dot: DotGraph) -> Self {
        let mut group_map = HashMap::new();
        let edge_pin_constrains: EdgeVec<(Point2<f64>, Constraints)> =
            dot.graph.new_edgevec(|e, eid, _| {
                let a = e
                    .get::<_, String>("pin")
                    .transpose()
                    .unwrap()
                    .map(|a| PinConstraint::parse(&a))
                    .flatten();

                if let Some(a) = a {
                    a.point_constraint(eid.0, &mut group_map)
                } else {
                    (Point2::new(0., 0.), Constraints::Free)
                }
            });

        let mut group_map = HashMap::new();

        let node_pin_constrains: NodeVec<(Point2<f64>, Constraints)> =
            dot.graph.new_nodevec(|nid, _, n| {
                let a = n
                    .get::<_, String>("pin")
                    .transpose()
                    .unwrap()
                    .map(|a| PinConstraint::parse(&a))
                    .flatten();

                if let Some(a) = a {
                    a.point_constraint(nid.0, &mut group_map)
                } else {
                    (Point2::new(0., 0.), Constraints::Free)
                }
            });

        let graph = dot.graph.map(
            |inv, nid, data| TypstNode::parse(inv, nid, data, &node_pin_constrains),
            |inv, store, p, eid, data| {
                TypstEdge::parse(inv, store, p, eid, data, &edge_pin_constrains)
            },
            TypstHedge::parse,
        );

        let spring_params = ParamTuning::parse(&dot.global_data);
        let schedule = GeoSchedule::parse(&dot.global_data);

        let step = dot
            .global_data
            .statements
            .get("step")
            .and_then(|s| s.parse::<f64>().ok());

        let temp = dot
            .global_data
            .statements
            .get("temp")
            .and_then(|s| s.parse::<f64>().ok());

        let seed = dot
            .global_data
            .statements
            .get("seed")
            .and_then(|s| s.parse::<u64>().ok());
        Self {
            graph,
            spring_params,
            schedule,
            step,
            temp,
            seed,
        }
    }
}

pub struct TreeInitCfg {
    pub dy: f64, // ≈ 1.2 * L
    pub dx: f64, // ≈ 0.9 * L
}

impl TypstGraph {
    pub fn tree_init_cfg(
        &self,
        map: &BTreeMap<String, String>,
    ) -> (TreeInitCfg, SpringChargeEnergy) {
        let viewport_w = map
            .get("viewport_w")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(10.0);
        let viewport_h = map
            .get("viewport_h")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(10.0);
        let tune = self.spring_params.clone();
        let energycfg =
            SpringChargeEnergy::from_graph(self.n_nodes(), viewport_w, viewport_h, tune);

        let l = energycfg.spring_length;
        let dy = map
            .get("tree_dy")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.2);
        let dx = map
            .get("tree_dx")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.9);
        (
            TreeInitCfg {
                dy: dy * l,
                dx: dx * l,
            },
            energycfg,
        )
    }

    pub fn update_positions(&mut self, node: NodeVec<Point2<f64>>, edge: EdgeVec<Point2<f64>>) {
        node.into_iter().for_each(|(i, p)| {
            let p = p + self[i].shift.unwrap_or(Vector2::zero());
            self.graph[i].pos = p;
        });

        edge.into_iter().for_each(|(i, p)| {
            let p = p + self[i].shift.unwrap_or(Vector2::zero());

            let angle = {
                let (_, pair) = &self.graph[&i];

                match pair {
                    HedgePair::Split { source, sink, .. } | HedgePair::Paired { source, sink } => {
                        let so = self.node_id(*source);
                        let a = self[so].pos;
                        let si = self.node_id(*sink);
                        let b = self[si].pos;
                        tangent_angle_toward_c_side(a, b, p)
                    }
                    _ => Ok(Rad::zero()),
                }
            };

            self.graph[i].bend = angle;
            self.graph[i].pos = p;
        });
    }

    pub fn new_positions(&self, cfg: TreeInitCfg) -> (NodeVec<Point2<f64>>, EdgeVec<Point2<f64>>) {
        let mut pos_v = self.new_nodevec(|_, _, n| n.pos);
        let mut pos_e = self.new_edgevec(|e, _, _| e.pos);

        let mut visited_edges: BitVec = self.empty_subgraph();
        let all: BitVec = self.full_filter();

        let mut comps = vec![];

        // Iterate over all edges in the subgraph
        for hedge_index in all.included_iter() {
            if visited_edges.includes(&hedge_index) {
                continue; // Already visited
            }
            let root_node = self.node_id(hedge_index);
            let reachable_edges =
                SimpleTraversalTree::depth_first_traverse(self, &all, &root_node, None).unwrap();
            visited_edges.union_with(&reachable_edges.covers(&all));
            let tree: SimpleTraversalTree<ParentChildStore<()>> = reachable_edges.cast();
            comps.push((tree, root_node));
        }

        let mut level: NodeVec<i32> = self.new_nodevec(|_, _, _| -1);
        let mut n_per_level: Vec<usize> = vec![];
        for (tree, root_node) in comps {
            // 2) compute levels (distance to root)
            let mut q = std::collections::VecDeque::new();
            level[root_node] = 0;
            if n_per_level.is_empty() {
                n_per_level.push(1);
            } else {
                n_per_level[0] += 1;
            }
            q.push_back(root_node);
            while let Some(v) = q.pop_front() {
                for u in tree.iter_children(v, &self.as_ref()) {
                    if level[u] < 0 {
                        level[u] = level[v] + 1;
                        if level[u] == n_per_level.len() as i32 {
                            n_per_level.push(1);
                        } else if level[u] < n_per_level.len() as i32 {
                            n_per_level[level[u] as usize] += 1;
                        } else {
                            panic!("Level out of bounds");
                        }

                        q.push_back(u);
                    }
                }
            }
        }

        for (cl, &n) in n_per_level.iter().enumerate() {
            // place on a horizontal line
            let k = n as f64;
            let width = (k - 1.0) * cfg.dx;
            let y = (cl as f64) * cfg.dy;
            for (i, &l) in level.iter() {
                if l != (cl as i32) {
                    continue;
                }
                let x = -0.5 * width + (i.0 as f64) * cfg.dx;
                self[i].constraints.shift((x, y).into(), i, &mut pos_v);
            }
        }

        // 4) edge control points
        for (pair, _, _) in self.iter_edges() {
            let h = pair.any_hedge();
            let eid = self[&h];
            match pair {
                // internal edge: midpoint + perpendicular bulge
                HedgePair::Paired { source, sink } | HedgePair::Split { source, sink, .. } => {
                    let a = pos_v[self.node_id(source)];
                    let b = pos_v[self.node_id(sink)];
                    let mid = a.midpoint(b);
                    self[eid]
                        .constraints
                        .shift((mid.x, mid.y).into(), eid, &mut pos_e);
                }
                HedgePair::Unpaired { hedge, .. } => {
                    let v = self.node_id(hedge);

                    self[eid].constraints.shift(
                        (pos_v[v].x + 1., pos_v[v].y + 1.).into(),
                        eid,
                        &mut pos_e,
                    );
                }
            }
        }

        (pos_v, pos_e)
    }

    pub fn parse<'a>(dot_str: &str) -> Result<Self, HedgeParseError<'a, (), (), (), ()>> {
        Ok(DotGraph::from_string(dot_str)?.into())
    }

    pub fn to_cbor(&self) -> CBORTypstGraph {
        CBORTypstGraph {
            edges: self.new_edgevec(|e, i, _p| EdgeData::new(e.clone(), self.orientation(i))),
            nodes: self.new_nodevec(|_id, _h, v| v.clone()),
        }
    }

    pub fn serialize_to_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.to_cbor().serialize_to_file(path)
    }

    pub fn to_dot_graph(&self) -> DotGraph {
        let graph = self.graph.map_data_ref(
            |_graph, _neighbors, node| node.to_dot(),
            |_graph, _eid, _pair, edge_data| edge_data.map(|e| e.to_dot()),
            |_hedge, hedge_data| hedge_data.to_dot(),
        );

        // Reconstruct GlobalData from layout parameters

        let mut global_data = GlobalData::from(());
        self.spring_params.add_to_global(&mut global_data);
        self.schedule.add_to_global(&mut global_data);
        DotGraph { graph, global_data }
    }
}

/// WASM function that takes DOT graph as string bytes, parses it into a TypstGraph,
/// applies layout, and returns the CBOR-serialized result as bytes
#[wasm_func]
pub fn layout_graph(arg: &[u8], arg2: &[u8]) -> Result<Vec<u8>, String> {
    // Convert bytes to string
    let dot_string = match std::str::from_utf8(arg) {
        Ok(s) => s,
        Err(_) => return Err("Invalid UTF-8".to_string()), // Return error on invalid UTF-8
    };

    let cbor_map: BTreeMap<String, String> = ciborium::de::from_reader(arg2)
        .map_err(|e| format!("Failed to deserialize CBOR map: {}", e))?;

    let dots = DotGraphSet::from_string(dot_string)
        .map_err(|a| a.to_string())?
        .into_iter();

    let step = cbor_map
        .get("step")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.2);
    let temp = cbor_map
        .get("temp")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.1);
    let seed = cbor_map
        .get("seed")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    let neigh = LayoutNeighbor {};

    let mut graphs = Vec::new();
    for g in dots {
        let mut typst_graph = TypstGraph::from(g);
        let (cfg, energy) = typst_graph.tree_init_cfg(&cbor_map);

        let (pos_n, pos_e) = typst_graph.new_positions(cfg);

        let state = typst_graph.graph.new_layout_state(pos_n, pos_e, 0.4);

        let (out, _stats) = anneal::<_, _, _, _, SmallRng>(
            state,
            typst_graph.step.unwrap_or(step),
            typst_graph.temp.unwrap_or(temp),
            SAConfig {
                seed: typst_graph.seed.unwrap_or(seed),
            },
            &neigh,
            &energy,
            &mut typst_graph.schedule,
        );

        typst_graph.update_positions(out.vertex_points, out.edge_points);

        graphs.push((
            typst_graph.to_cbor(),
            typst_graph.to_dot_graph().debug_dot(),
        ));
    }

    let mut buffer = Vec::new();
    ciborium::ser::into_writer(&graphs, &mut buffer).map_err(|a| a.to_string())?;
    Ok(buffer)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CBORTypstGraph {
    edges: EdgeVec<EdgeData<TypstEdge>>,
    nodes: NodeVec<TypstNode>,
}

impl CBORTypstGraph {
    pub fn serialize_to_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        ciborium::ser::into_writer(self, writer)?;
        Ok(())
    }

    pub fn deserialize_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let graph = ciborium::de::from_reader(file)?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use linnet::dot;

    use linnet::half_edge::swap::Swap;

    use crate::{CBORTypstGraph, TypstGraph};

    #[test]
    fn dot_cbor() {
        let g: TypstGraph = dot!(digraph{ a; a->b}).unwrap().into();

        let _cbor = g.to_cbor();
    }

    #[test]
    fn test_cbor_serialization() {
        // use std::fs;

        let g: TypstGraph = dot!(digraph{ a; a->b; b->c}).unwrap().into();
        let cbor = g.to_cbor();

        let test_path = "test_graph.cbor";

        // Test serialization
        cbor.serialize_to_file(test_path)
            .expect("Failed to serialize to file");

        // Test deserialization
        let deserialized = CBORTypstGraph::deserialize_from_file(test_path)
            .expect("Failed to deserialize from file");

        // Verify the deserialized graph has the same structure
        assert_eq!(cbor.nodes.len(), deserialized.nodes.len());
        assert_eq!(cbor.edges.len(), deserialized.edges.len());

        // Clean up test file
        fs::remove_file(test_path).ok();
    }

    #[test]
    fn test_typst_graph_convenience_serialization() {
        use std::fs;

        let g: TypstGraph = dot!(digraph{ a->b; b->c}).unwrap().into();
        let test_path = "test_convenience_graph.cbor";

        // Test convenience method
        g.serialize_to_file(test_path)
            .expect("Failed to serialize using convenience method");

        // Verify we can deserialize the result
        let deserialized = CBORTypstGraph::deserialize_from_file(test_path)
            .expect("Failed to deserialize from file");

        // Verify the structure
        assert_eq!(g.to_cbor().nodes.len(), deserialized.nodes.len());
        assert_eq!(g.to_cbor().edges.len(), deserialized.edges.len());

        // Clean up test file
        fs::remove_file(test_path).ok();
    }

    #[test]
    fn test_pin_parsing() {
        let _g:TypstGraph = dot!(digraph dot_80_0_GL208 {


            node[
              eval="(stroke:blue,fill :black,
              radius:2pt,
              outset: -2pt)"
            ]

            edge[
              eval=top
            ]
            v0[pin="5.2,2.2", style=invis]
            v1[pin="5.2,-2.2",style=invis]
            v2[pin="-5.2,2.2",style=invis]
            v3[pin="-5.2,-2.2",style=invis]
            v0 -> v11 [eval=photon]
            v1 -> v10 [eval="(..photon,label:[$gamma$],label-side: left)", mom_eval="(label:[$p_1$],label-sep:0mm)"]
            v9 -> v2 [eval=photon]
            v8 -> v3 [eval=photon]
            v4 -> v10
            v10 -> v5
            v5 -> v11 [dir=back]
            v11 -> v4
            v4 -> v7 [eval=gluon]
            v5 -> v6 [eval=gluon]
            v6 -> v8
            v8 -> v7
            v7 -> v9
            v9 -> v6
        }).unwrap().into();
    }
}
