use std::ops::IndexMut;

use bitvec::vec::BitVec;
use cgmath::{EuclideanSpace, MetricSpace, Point2, Vector2};

use rand::{distributions::Uniform, prelude::Distribution, Rng};
use serde::{Deserialize, Serialize};

use crate::{
    half_edge::{
        involution::{EdgeIndex, EdgeVec},
        layout::simulatedanneale::{Energy, Neighbor},
        nodestore::NodeStorageOps,
        subgraph::SubGraph,
        swap::Swap,
        HedgeGraph, NodeIndex, NodeVec,
    },
    parser::GlobalData,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PointConstraint {
    pub x: Constraint,
    pub y: Constraint,
}

impl Default for PointConstraint {
    fn default() -> Self {
        PointConstraint {
            x: Constraint::Free,
            y: Constraint::Free,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Constraint {
    Fixed,
    Free,
    Grouped(usize),
}

impl Default for Constraint {
    fn default() -> Self {
        Constraint::Free
    }
}

pub trait Shiftable {
    fn shift<I: From<usize> + PartialEq + Copy, R: IndexMut<I, Output = Point2<f64>>>(
        &self,
        shift: Vector2<f64>,
        index: I,
        values: &mut R,
    ) -> bool;
}

impl Shiftable for PointConstraint {
    fn shift<I: From<usize> + PartialEq + Copy, R: IndexMut<I, Output = Point2<f64>>>(
        &self,
        shift: Vector2<f64>,
        index: I,
        values: &mut R,
    ) -> bool {
        let mut changed = false;

        match (self.x, self.y) {
            (Constraint::Fixed, Constraint::Fixed) => {}
            (Constraint::Free, Constraint::Free) => {
                values[index] += shift;
                changed = true;
            }
            (Constraint::Fixed, Constraint::Free) => {
                values[index].y += shift.y;
                changed = true;
            }
            (Constraint::Free, Constraint::Fixed) => {
                values[index].x += shift.x;
                changed = true;
            }
            (Constraint::Grouped(r), Constraint::Fixed) => {
                let i = r.into();
                if i != index {
                    values[index].x = values[i].x;
                } else {
                    values[index].x += shift.x;
                    changed = true;
                }
            }
            (Constraint::Grouped(r), Constraint::Free) => {
                let i = r.into();
                if i != index {
                    values[index].x = values[i].x;
                    values[index].y += shift.y;
                    changed = true;
                } else {
                    values[index] += shift;
                    changed = true;
                }
            }

            (Constraint::Fixed, Constraint::Grouped(r)) => {
                let i = r.into();
                if i != index {
                    values[index].y = values[i].y;
                } else {
                    values[index].y += shift.y;
                    changed = true;
                }
            }
            (Constraint::Free, Constraint::Grouped(r)) => {
                let i = r.into();
                if i != index {
                    values[index].y = values[i].y;
                    values[index].x += shift.x;
                    changed = true;
                } else {
                    values[index] += shift;
                    changed = true;
                }
            }
            (Constraint::Grouped(xi), Constraint::Grouped(yi)) => {
                let ix = xi.into();
                let iy = yi.into();
                if ix != index && iy != index {
                    values[index].x = values[ix].x;
                    values[index].y = values[iy].y;
                } else if ix == index && iy != index {
                    values[index].x += shift.x;
                    values[index].y = values[iy].y;
                    changed = true;
                } else if ix != index && iy == index {
                    values[index].x = values[ix].x;
                    values[index].y += shift.y;
                    changed = true;
                } else {
                    values[index] += shift;
                    changed = true;
                }
            }
        }
        changed
    }
}

pub struct LayoutState<'a, E, V, H, N: NodeStorageOps<NodeData = V>> {
    pub graph: &'a HedgeGraph<E, V, H, N>,
    pub ext: BitVec,
    pub vertex_points: NodeVec<Point2<f64>>,
    pub edge_points: EdgeVec<Point2<f64>>,
    pub delta: f64,
}

impl<'a, E, V, H, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, H, N> {
    pub fn new_layout_state(
        &self,
        vertex_points: NodeVec<Point2<f64>>,
        edge_points: EdgeVec<Point2<f64>>,
        delta: f64,
    ) -> LayoutState<'_, E, V, H, N> {
        let ext = self.external_filter();
        LayoutState {
            graph: self,
            ext,
            vertex_points,
            edge_points,
            delta,
        }
    }
}

impl<'a, E, V, H, N: NodeStorageOps<NodeData = V>> Clone for LayoutState<'a, E, V, H, N> {
    fn clone(&self) -> Self {
        LayoutState {
            graph: self.graph,
            ext: self.ext.clone(),
            vertex_points: self.vertex_points.clone(),
            edge_points: self.edge_points.clone(),
            delta: self.delta,
        }
    }
}

pub struct LayoutNeighbor;

impl<'a, E: Shiftable, V: Shiftable, H, N: NodeStorageOps<NodeData = V> + Clone>
    Neighbor<LayoutState<'a, E, V, H, N>> for LayoutNeighbor
{
    fn propose(
        &self,
        s: &LayoutState<'a, E, V, H, N>,
        rng: &mut impl Rng,
        step: f64,
        _temp: f64,
    ) -> LayoutState<'a, E, V, H, N> {
        let mut st = s.clone();
        let n_v: NodeIndex = st.vertex_points.len();
        let n_e: EdgeIndex = st.edge_points.len();
        let step_range: Uniform<f64> = Uniform::try_from(-step..step).unwrap();

        let mut didnothing = true;
        while didnothing {
            match rng.gen_range(0..100) {
                0..=69 => {
                    // single-DOF
                    if rng.gen_bool(0.6) {
                        let v = NodeIndex(rng.gen_range(0..n_v.0));

                        let shift = Vector2::from(if rng.gen_bool(0.5) {
                            (step_range.sample(rng), 0.0)
                        } else {
                            (0.0, step_range.sample(rng))
                        });
                        didnothing = !st.graph[v].shift(shift, v, &mut st.vertex_points);
                    } else {
                        let e = EdgeIndex(rng.gen_range(0..n_e.0));
                        let shift = Vector2::from(if rng.gen_bool(0.5) {
                            (step_range.sample(rng), 0.0)
                        } else {
                            (0.0, step_range.sample(rng))
                        });
                        didnothing = !st.graph[e].shift(shift, e, &mut st.edge_points);
                    }
                }
                _ => {
                    // vertex block
                    let v = NodeIndex(rng.gen_range(0..n_v.0));

                    let shift: Vector2<f64> =
                        Vector2::from((step_range.sample(rng) * 0.6, step_range.sample(rng) * 0.6));

                    didnothing = !st.graph[v].shift(shift, v, &mut st.vertex_points);

                    st.graph.iter_crown(v).for_each(|a| {
                        let index = st.graph[&a];

                        didnothing |= !st.graph[index].shift(shift, index, &mut st.edge_points);
                    });
                } // _ => {
                  //     // everything
                  //     let e = EdgeIndex(rng.gen_range(0..n_e.0));
                  //     st.edge_points[e].x += step_range.sample(rng) * 0.5;
                  //     st.edge_points[e].y += step_range.sample(rng) * 0.5;
                  // }
            }
        }
        st
    }
}

#[derive(Clone, Copy)]
pub struct SpringChargeEnergy {
    pub spring_length: f64,   // L
    pub k_spring: f64,        // 1.0
    pub c_vv: f64,            // vertex-vertex charge (≈ 0.14*L^3)
    pub dangling_charge: f64, // dangling edge charge (≈ 0.14*L^3)
    pub c_ev: f64,            // edge-vertex (≈ 0.028*L^3)
    pub c_ee_local: f64,      // edge-edge local (≈ 0.014*L^3)
    pub c_center: f64,        // central pull (≈ 0.007*L^3)
    pub eps: f64,             // 1e-4
}

impl<'a, E, V, H, N: NodeStorageOps<NodeData = V> + Clone> Energy<LayoutState<'a, E, V, H, N>>
    for SpringChargeEnergy
{
    fn energy(&self, s: &LayoutState<'a, E, V, H, N>) -> f64 {
        let n = s.vertex_points.len().0;
        let m = s.edge_points.len().0;

        let mut energy = 0.0;

        for i in 0..n {
            let ni = NodeIndex(i);
            let np = s.vertex_points[ni];
            // 1) Vertex–vertex repulsion
            for j in (i + 1)..n {
                let nj = NodeIndex(j);
                let vj = s.vertex_points[nj];
                energy += 0.5 * self.c_vv / (np.distance(vj) + self.eps);
            }

            // 2) Edge–vertex repulsion (all vertices vs all edge control points)
            if self.c_ev != 0.0 {
                for e in 0..m {
                    let ei = EdgeIndex(e);
                    let ei = s.edge_points[ei];
                    energy += self.c_ev / (np.distance(ei) + self.eps);
                }
            }

            // 3) Edge spring energy (only between edge points sharing a node)
            for e in s.graph.iter_crown(ni) {
                let ei = s.graph[&e];
                let ep = s.edge_points[ei];

                let t = self.spring_length - np.distance(ep);
                energy += 0.5 * self.k_spring * (t * t);
                // 4) Edge–edge local repulsion (only between edge points sharing a node)ß
                for e in s.graph.iter_crown(ni) {
                    let ej = s.graph[&e];
                    if ei == ej {
                        continue;
                    }
                    let ej = s.edge_points[ej];
                    energy += 0.5 * self.c_ee_local / (ej.distance(ep) + self.eps);
                }
            }

            // 5) Central pull (once per vertex)
            if self.c_center != 0.0 {
                let r = np.distance(EuclideanSpace::origin());
                if r > 1.0 {
                    energy += 0.5 * self.c_center / (r + self.eps);
                }
            }
        }

        // 6) Dangling edge repulsion
        for hi in s.ext.included_iter() {
            for hj in s.ext.included_iter() {
                if hi >= hj {
                    continue;
                }
                let hi = s.graph[&hi];
                let hj = s.graph[&hj];
                let pi = s.edge_points[hi];
                let pj = s.edge_points[hj];
                energy += 0.5 * self.dangling_charge / (pi.distance(pj) + self.eps);
            }
        }

        energy
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ParamTuning {
    pub length_scale: f64,   // scales L: default 1.0
    pub k_spring: f64,       // spring stiffness: default 1.0
    pub beta: f64,           // vertex–vertex strength
    pub gamma_dangling: f64, // dangling edge vs vertex–vertex
    pub gamma_ev: f64,       // edge–vertex vs vertex–vertex
    pub gamma_ee: f64,       // local edge–edge vs vertex–vertex
    pub g_center: f64,       // central vs vertex–vertex
    pub eps: f64,            // softening epsilon
}

impl ParamTuning {
    pub fn add_to_global(&self, global_data: &mut GlobalData) {
        global_data
            .statements
            .insert("length_scale".to_string(), self.length_scale.to_string());
        global_data
            .statements
            .insert("k_spring".to_string(), self.k_spring.to_string());
        global_data
            .statements
            .insert("beta".to_string(), self.beta.to_string());
        global_data
            .statements
            .insert("gamma_ev".to_string(), self.gamma_ev.to_string());

        global_data.statements.insert(
            "gamma_dangling".to_string(),
            self.gamma_dangling.to_string(),
        );
        global_data
            .statements
            .insert("gamma_ee".to_string(), self.gamma_ee.to_string());
        global_data
            .statements
            .insert("g_center".to_string(), self.g_center.to_string());
        global_data
            .statements
            .insert("eps".to_string(), self.eps.to_string());
    }
    pub fn parse(global_data: &GlobalData) -> Self {
        let mut tune = ParamTuning::default();

        for (key, value) in &global_data.statements {
            match key.as_str() {
                "length_scale" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.length_scale = v;
                    }
                }
                "gamma_dangling" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.gamma_dangling = v;
                    }
                }
                "k_spring" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.k_spring = v;
                    }
                }
                "beta" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.beta = v;
                    }
                }
                "gamma_ev" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.gamma_ev = v;
                    }
                }
                "gamma_ee" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.gamma_ee = v;
                    }
                }
                "g_center" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.g_center = v;
                    }
                }
                "eps" => {
                    if let Ok(v) = value.parse::<f64>() {
                        tune.eps = v;
                    }
                }
                _ => {}
            }
        }

        tune
    }
}
impl Default for ParamTuning {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            k_spring: 1.0,
            beta: 0.14,
            gamma_dangling: 0.14,
            gamma_ev: 0.20,
            gamma_ee: 0.10,
            g_center: 0.05,
            eps: 1e-4,
        }
    }
}

impl SpringChargeEnergy {
    pub fn from_graph(n_nodes: usize, viewport_w: f64, viewport_h: f64, tune: ParamTuning) -> Self {
        let area = (viewport_w * viewport_h).max(1e-9);
        let spring_length = tune.length_scale * (area / (n_nodes.max(1) as f64)).sqrt();

        SpringChargeEnergy {
            spring_length,
            k_spring: tune.k_spring,
            c_vv: tune.beta * spring_length.powi(2),
            c_ev: tune.beta * tune.gamma_ev * spring_length.powi(2),
            c_ee_local: tune.beta * tune.gamma_ee * spring_length.powi(2),
            c_center: tune.beta * tune.g_center * spring_length.powi(2),
            dangling_charge: tune.gamma_dangling * tune.beta * spring_length.powi(2),
            eps: tune.eps,
        }
    }
}
