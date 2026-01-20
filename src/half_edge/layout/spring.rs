use std::ops::IndexMut;

use cgmath::{EuclideanSpace, MetricSpace, Point2, Vector2};

use rand::{distributions::Uniform, prelude::Distribution, Rng};
use serde::{Deserialize, Serialize};

use crate::{
    half_edge::{
        involution::{EdgeIndex, EdgeVec},
        layout::simulatedanneale::{Energy, Neighbor},
        nodestore::NodeStorageOps,
        subgraph::{subset::SubSet, Inclusion, ModifySubSet, SuBitGraph, SubSetLike},
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ShiftDirection {
    Any,
    PositiveOnly,
    NegativeOnly,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Constraint {
    Fixed,
    Free,
    Grouped(usize, ShiftDirection),
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

pub trait HasPointConstraint {
    fn point_constraint(&self) -> &PointConstraint;
}

impl HasPointConstraint for PointConstraint {
    fn point_constraint(&self) -> &PointConstraint {
        self
    }
}

fn apply_directional_shift(shift_val: f64, direction: ShiftDirection) -> f64 {
    match direction {
        ShiftDirection::Any => shift_val,
        ShiftDirection::PositiveOnly => shift_val.abs(),
        ShiftDirection::NegativeOnly => -shift_val.abs(),
    }
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
            (Constraint::Grouped(r, dir), Constraint::Fixed) => {
                let i = r.into();
                if i != index {
                    values[index].x = values[i].x;
                } else {
                    let x_shift = apply_directional_shift(shift.x, dir);
                    values[index].x += x_shift;
                    changed = x_shift != 0.0;
                }
            }
            (Constraint::Grouped(r, dir), Constraint::Free) => {
                let i = r.into();
                if i != index {
                    values[index].x = values[i].x;
                    values[index].y += shift.y;
                    changed = true;
                } else {
                    let x_shift = apply_directional_shift(shift.x, dir);
                    values[index].x += x_shift;
                    values[index].y += shift.y;
                    changed = x_shift != 0.0 || shift.y != 0.0;
                }
            }

            (Constraint::Fixed, Constraint::Grouped(r, dir)) => {
                let i = r.into();
                if i != index {
                    values[index].y = values[i].y;
                } else {
                    let y_shift = apply_directional_shift(shift.y, dir);
                    values[index].y += y_shift;
                    changed = y_shift != 0.0;
                }
            }
            (Constraint::Free, Constraint::Grouped(r, dir)) => {
                let i = r.into();
                if i != index {
                    values[index].y = values[i].y;
                    values[index].x += shift.x;
                    changed = true;
                } else {
                    let y_shift = apply_directional_shift(shift.y, dir);
                    values[index].x += shift.x;
                    values[index].y += y_shift;
                    changed = shift.x != 0.0 || y_shift != 0.0;
                }
            }
            (Constraint::Grouped(xi, x_dir), Constraint::Grouped(yi, y_dir)) => {
                let ix = xi.into();
                let iy = yi.into();
                if ix != index && iy != index {
                    values[index].x = values[ix].x;
                    values[index].y = values[iy].y;
                } else if ix == index && iy != index {
                    let x_shift = apply_directional_shift(shift.x, x_dir);
                    values[index].x += x_shift;
                    values[index].y = values[iy].y;
                    changed = x_shift != 0.0;
                } else if ix != index && iy == index {
                    let y_shift = apply_directional_shift(shift.y, y_dir);
                    values[index].x = values[ix].x;
                    values[index].y += y_shift;
                    changed = y_shift != 0.0;
                } else {
                    let x_shift = apply_directional_shift(shift.x, x_dir);
                    let y_shift = apply_directional_shift(shift.y, y_dir);
                    values[index].x += x_shift;
                    values[index].y += y_shift;
                    changed = x_shift != 0.0 || y_shift != 0.0;
                }
            }
        }
        changed
    }
}

pub struct LayoutState<'a, E, V, H, N: NodeStorageOps<NodeData = V>> {
    pub graph: &'a HedgeGraph<E, V, H, N>,
    pub ext: SuBitGraph,
    pub vertex_points: NodeVec<Point2<f64>>,
    pub edge_points: EdgeVec<Point2<f64>>,
    pub delta: f64,
    // Tracks which node/edge entries were mutated during proposal generation so
    // the energy function can update only the affected terms.
    pub changed_nodes: SubSet<NodeIndex>,
    pub changed_edges: SubSet<EdgeIndex>,
    pub incremental: bool,
}

impl<'a, E, V, H, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, H, N> {
    pub fn new_layout_state(
        &self,
        vertex_points: NodeVec<Point2<f64>>,
        edge_points: EdgeVec<Point2<f64>>,
        delta: f64,
        incremental: bool,
    ) -> LayoutState<'_, E, V, H, N> {
        let ext = self.external_filter();
        let len_v = vertex_points.len().0;
        let len_e = edge_points.len().0;
        LayoutState {
            graph: self,
            ext,
            vertex_points,
            edge_points,
            delta,
            changed_nodes: SubSet::empty(len_v),
            changed_edges: SubSet::empty(len_e),
            incremental,
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
            changed_nodes: self.changed_nodes.clone(),
            changed_edges: self.changed_edges.clone(),
            incremental: self.incremental,
        }
    }
}

impl<'a, E, V, H, N: NodeStorageOps<NodeData = V>> LayoutState<'a, E, V, H, N> {
    fn mark_node_changed(&mut self, index: NodeIndex) {
        self.changed_nodes.add(index);
    }

    fn mark_edge_changed(&mut self, index: EdgeIndex) {
        self.changed_edges.add(index);
    }

    pub fn clear_changes(&mut self) {
        self.changed_nodes.clear();
        self.changed_edges.clear();
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

                        let shift = LayoutNeighbor::axis_shift(&step_range, rng);
                        let changed = apply_vertex_shift(&mut st, v, shift);
                        didnothing = !changed;
                    } else {
                        let e = EdgeIndex(rng.gen_range(0..n_e.0));
                        let shift = LayoutNeighbor::axis_shift(&step_range, rng);
                        let changed = apply_edge_shift(&mut st, e, shift);
                        didnothing = !changed;
                    }
                }
                _ => {
                    // vertex block
                    let v = NodeIndex(rng.gen_range(0..n_v.0));

                    let shift = LayoutNeighbor::diagonal_shift(&step_range, rng, 0.6);

                    // Cache whether a vertex move succeeded so we can mark it.
                    let mut changed_any = apply_vertex_shift(&mut st, v, shift);

                    st.graph.iter_crown(v).for_each(|a| {
                        let index = st.graph[&a];

                        // Propagate to incident edge control points; any change gets recorded.
                        changed_any |= apply_edge_shift(&mut st, index, shift);
                    });

                    didnothing = !changed_any;
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

impl LayoutNeighbor {
    fn axis_shift(step_range: &Uniform<f64>, rng: &mut impl Rng) -> Vector2<f64> {
        if rng.gen_bool(0.5) {
            Vector2::from((step_range.sample(rng), 0.0))
        } else {
            Vector2::from((0.0, step_range.sample(rng)))
        }
    }

    fn diagonal_shift(step_range: &Uniform<f64>, rng: &mut impl Rng, scale: f64) -> Vector2<f64> {
        let mut sample = || step_range.sample(rng) * scale;
        Vector2::from((sample(), sample()))
    }
}

pub struct PinnedLayoutNeighbor;

impl<'a, E, V, H, N> Neighbor<LayoutState<'a, E, V, H, N>> for PinnedLayoutNeighbor
where
    E: Shiftable + HasPointConstraint,
    V: Shiftable + HasPointConstraint,
    N: NodeStorageOps<NodeData = V> + Clone,
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

                        let shift = LayoutNeighbor::axis_shift(&step_range, rng);
                        let changed = apply_vertex_shift_with_groups(&mut st, v, shift);
                        didnothing = !changed;
                    } else {
                        let e = EdgeIndex(rng.gen_range(0..n_e.0));
                        let shift = LayoutNeighbor::axis_shift(&step_range, rng);
                        let changed = apply_edge_shift_with_groups(&mut st, e, shift);
                        didnothing = !changed;
                    }
                }
                _ => {
                    // vertex block
                    let v = NodeIndex(rng.gen_range(0..n_v.0));

                    let shift = LayoutNeighbor::diagonal_shift(&step_range, rng, 0.6);

                    // Cache whether a vertex move succeeded so we can mark it.
                    let mut changed_any = apply_vertex_shift_with_groups(&mut st, v, shift);

                    st.graph.iter_crown(v).for_each(|a| {
                        let index = st.graph[&a];

                        // Propagate to incident edge control points; any change gets recorded.
                        changed_any |= apply_edge_shift_with_groups(&mut st, index, shift);
                    });

                    didnothing = !changed_any;
                }
            }
        }
        st
    }
}

fn apply_vertex_shift<
    'a,
    E: Shiftable,
    V: Shiftable,
    H,
    N: NodeStorageOps<NodeData = V> + Clone,
>(
    state: &mut LayoutState<'a, E, V, H, N>,
    idx: NodeIndex,
    shift: Vector2<f64>,
) -> bool {
    let changed = state.graph[idx].shift(shift, idx, &mut state.vertex_points);
    if changed {
        state.mark_node_changed(idx);
    }
    changed
}

fn apply_edge_shift<'a, E: Shiftable, V: Shiftable, H, N: NodeStorageOps<NodeData = V> + Clone>(
    state: &mut LayoutState<'a, E, V, H, N>,
    idx: EdgeIndex,
    shift: Vector2<f64>,
) -> bool {
    let changed = state.graph[idx].shift(shift, idx, &mut state.edge_points);
    if changed {
        state.mark_edge_changed(idx);
    }
    changed
}

fn is_group_reference(constraints: &PointConstraint, reference: usize) -> bool {
    matches!(constraints.x, Constraint::Grouped(r, _) if r == reference)
        || matches!(constraints.y, Constraint::Grouped(r, _) if r == reference)
}

fn propagate_grouped_nodes<'a, E, V, H, N>(
    state: &mut LayoutState<'a, E, V, H, N>,
    reference: NodeIndex,
) -> bool
where
    V: HasPointConstraint,
    N: NodeStorageOps<NodeData = V> + Clone,
{
    let graph = state.graph;
    let reference_point = state.vertex_points[reference];
    let reference_id = reference.0;
    let mut changed_any = false;
    let len = state.vertex_points.len().0;

    for i in 0..len {
        if i == reference_id {
            continue;
        }
        let idx = NodeIndex(i);
        let constraints = graph[idx].point_constraint();
        let mut changed = false;

        if matches!(constraints.x, Constraint::Grouped(r, _) if r == reference_id) {
            if state.vertex_points[idx].x != reference_point.x {
                state.vertex_points[idx].x = reference_point.x;
                changed = true;
            }
        }
        if matches!(constraints.y, Constraint::Grouped(r, _) if r == reference_id) {
            if state.vertex_points[idx].y != reference_point.y {
                state.vertex_points[idx].y = reference_point.y;
                changed = true;
            }
        }

        if changed {
            state.mark_node_changed(idx);
            changed_any = true;
        }
    }

    changed_any
}

fn propagate_grouped_edges<'a, E, V, H, N>(
    state: &mut LayoutState<'a, E, V, H, N>,
    reference: EdgeIndex,
) -> bool
where
    E: HasPointConstraint,
    N: NodeStorageOps<NodeData = V> + Clone,
{
    let graph = state.graph;
    let reference_point = state.edge_points[reference];
    let reference_id = reference.0;
    let mut changed_any = false;
    let len = state.edge_points.len().0;

    for i in 0..len {
        if i == reference_id {
            continue;
        }
        let idx = EdgeIndex(i);
        let constraints = graph[idx].point_constraint();
        let mut changed = false;

        if matches!(constraints.x, Constraint::Grouped(r, _) if r == reference_id) {
            if state.edge_points[idx].x != reference_point.x {
                state.edge_points[idx].x = reference_point.x;
                changed = true;
            }
        }
        if matches!(constraints.y, Constraint::Grouped(r, _) if r == reference_id) {
            if state.edge_points[idx].y != reference_point.y {
                state.edge_points[idx].y = reference_point.y;
                changed = true;
            }
        }

        if changed {
            state.mark_edge_changed(idx);
            changed_any = true;
        }
    }

    changed_any
}

fn apply_vertex_shift_with_groups<
    'a,
    E: Shiftable + HasPointConstraint,
    V: Shiftable + HasPointConstraint,
    H,
    N: NodeStorageOps<NodeData = V> + Clone,
>(
    state: &mut LayoutState<'a, E, V, H, N>,
    idx: NodeIndex,
    shift: Vector2<f64>,
) -> bool {
    let changed = state.graph[idx].shift(shift, idx, &mut state.vertex_points);
    if changed {
        state.mark_node_changed(idx);
        let constraints = state.graph[idx].point_constraint();
        if is_group_reference(constraints, idx.0) {
            return propagate_grouped_nodes(state, idx) || changed;
        }
    }
    changed
}

fn apply_edge_shift_with_groups<
    'a,
    E: Shiftable + HasPointConstraint,
    V: Shiftable + HasPointConstraint,
    H,
    N: NodeStorageOps<NodeData = V> + Clone,
>(
    state: &mut LayoutState<'a, E, V, H, N>,
    idx: EdgeIndex,
    shift: Vector2<f64>,
) -> bool {
    let changed = state.graph[idx].shift(shift, idx, &mut state.edge_points);
    if changed {
        state.mark_edge_changed(idx);
        let constraints = state.graph[idx].point_constraint();
        if is_group_reference(constraints, idx.0) {
            return propagate_grouped_edges(state, idx) || changed;
        }
    }
    changed
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
    fn energy(
        &self,
        prev: Option<(&LayoutState<'a, E, V, H, N>, f64)>,
        next: &LayoutState<'a, E, V, H, N>,
    ) -> f64 {
        if !next.incremental {
            return self.total_energy(next);
        }

        if let Some((prev_state, prev_energy)) = prev {
            if next.changed_nodes.is_empty() && next.changed_edges.is_empty() {
                // No mutations since the last evaluation, so the cached value is exact.
                return prev_energy;
            }

            // Compute only the energy terms affected by the flagged nodes/edges on both states.
            let prev_partial =
                self.partial_energy(prev_state, &next.changed_nodes, &next.changed_edges);
            let next_partial = self.partial_energy(next, &next.changed_nodes, &next.changed_edges);

            // Replace only the interactions influenced by the mutated nodes/edges.
            prev_energy - prev_partial + next_partial
        } else {
            // First evaluation falls back to the full O(n²) pass.
            self.total_energy(next)
        }
    }

    fn on_accept(&self, state: &mut LayoutState<'a, E, V, H, N>) {
        state.clear_changes();
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
    fn total_energy<'a, E, V, H, N>(&self, s: &LayoutState<'a, E, V, H, N>) -> f64
    where
        N: NodeStorageOps<NodeData = V> + Clone,
    {
        let n = s.vertex_points.len().0;
        let m = s.edge_points.len().0;

        let mut energy = 0.0;

        for i in 0..n {
            let ni = NodeIndex(i);
            let np = s.vertex_points[ni];
            for j in (i + 1)..n {
                let nj = NodeIndex(j);
                let vj = s.vertex_points[nj];
                energy += 0.5 * self.c_vv / (np.distance(vj) + self.eps);
            }

            if self.c_ev != 0.0 {
                for e in 0..m {
                    let ei = EdgeIndex(e);
                    let ep = s.edge_points[ei];
                    energy += self.c_ev / (np.distance(ep) + self.eps);
                }
            }

            for e in s.graph.iter_crown(ni) {
                let ei = s.graph[&e];
                let ep = s.edge_points[ei];

                let t = self.spring_length - np.distance(ep);
                energy += 0.5 * self.k_spring * (t * t);
                for e in s.graph.iter_crown(ni) {
                    let ej = s.graph[&e];
                    if ei == ej {
                        continue;
                    }
                    let ejp = s.edge_points[ej];
                    energy += 0.5 * self.c_ee_local / (ejp.distance(ep) + self.eps);
                }
            }

            if self.c_center != 0.0 {
                let r = np.distance(EuclideanSpace::origin());
                if r > 1.0 {
                    energy += 0.5 * self.c_center / (r + self.eps);
                }
            }
        }

        for hi in s.ext.included_iter() {
            for hj in s.ext.included_iter() {
                if hi >= hj {
                    continue;
                }
                let hi_idx = s.graph[&hi];
                let hj_idx = s.graph[&hj];
                let pi = s.edge_points[hi_idx];
                let pj = s.edge_points[hj_idx];
                energy += 0.5 * self.dangling_charge / (pi.distance(pj) + self.eps);
            }
        }

        energy
    }

    /// Recompute the energy contributions that touch the mutated nodes/edges.
    fn partial_energy<'a, E, V, H, N>(
        &self,
        s: &LayoutState<'a, E, V, H, N>,
        node_changes: &SubSet<NodeIndex>,
        edge_changes: &SubSet<EdgeIndex>,
    ) -> f64
    where
        N: NodeStorageOps<NodeData = V> + Clone,
    {
        let n = s.vertex_points.len().0;
        let m = s.edge_points.len().0;
        let mut energy = 0.0;

        for i in 0..n {
            let ni = NodeIndex(i);
            let node_changed = node_changes.includes(&ni);
            let np = s.vertex_points[ni];

            for j in (i + 1)..n {
                let nj = NodeIndex(j);
                if !(node_changed || node_changes.includes(&nj)) {
                    continue;
                }
                let vj = s.vertex_points[nj];
                // Vertex–vertex repulsion.
                energy += 0.5 * self.c_vv / (np.distance(vj) + self.eps);
            }

            if self.c_ev != 0.0 {
                for e in 0..m {
                    let ei = EdgeIndex(e);
                    if !(node_changed || edge_changes.includes(&ei)) {
                        continue;
                    }
                    let ep = s.edge_points[ei];
                    // Edge–vertex repulsion.
                    energy += self.c_ev / (np.distance(ep) + self.eps);
                }
            }

            let include_node = node_changed;
            for hedge in s.graph.iter_crown(ni) {
                let ei = s.graph[&hedge];
                let edge_changed = edge_changes.includes(&ei);
                if !(include_node || edge_changed) {
                    continue;
                }
                let ep = s.edge_points[ei];
                let t = self.spring_length - np.distance(ep);
                // Spring term for the edge control point.
                energy += 0.5 * self.k_spring * (t * t);

                for other in s.graph.iter_crown(ni) {
                    let ej = s.graph[&other];
                    if ei == ej {
                        continue;
                    }
                    let other_edge_changed = edge_changes.includes(&ej);
                    if !(include_node || edge_changed || other_edge_changed) {
                        continue;
                    }
                    let ejp = s.edge_points[ej];
                    // Local edge–edge repulsion around `ni`.
                    energy += 0.5 * self.c_ee_local / (ejp.distance(ep) + self.eps);
                }
            }

            if include_node && self.c_center != 0.0 {
                let r = np.distance(EuclideanSpace::origin());
                if r > 1.0 {
                    // Central pull acts only on moved vertices.
                    energy += 0.5 * self.c_center / (r + self.eps);
                }
            }
        }

        for hi in s.ext.included_iter() {
            let edge_i = s.graph[&hi];
            let edge_i_changed = edge_changes.includes(&edge_i);
            for hj in s.ext.included_iter() {
                if hi >= hj {
                    continue;
                }
                let edge_j = s.graph[&hj];
                let edge_j_changed = edge_changes.includes(&edge_j);
                if !(edge_i_changed || edge_j_changed) {
                    continue;
                }
                let pi = s.edge_points[edge_i];
                let pj = s.edge_points[edge_j];
                // Dangling charge interactions.
                energy += 0.5 * self.dangling_charge / (pi.distance(pj) + self.eps);
            }
        }

        energy
    }

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
