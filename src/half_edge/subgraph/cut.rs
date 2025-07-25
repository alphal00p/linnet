use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

use super::{Cycle, Inclusion, SubGraph, SubGraphHedgeIter, SubGraphOps};
use crate::half_edge::hedgevec::Accessors;
use crate::half_edge::involution::{EdgeIndex, HedgePair};
use crate::half_edge::nodestore::NodeStorageOps;
use crate::half_edge::EdgeAccessors;
use crate::half_edge::{
    involution::SignOrZero, EdgeData, Flow, Hedge, HedgeGraph, InvolutiveMapping, NodeStorage,
    Orientation, PowersetIterator,
};
#[cfg(feature = "drawing")]
use crate::half_edge::{
    layout::{LayoutEdge, LayoutIters, LayoutParams, LayoutSettings, LayoutVertex},
    nodestore::NodeStorageVec,
};
use crate::parser::DotEdgeData;
use bitvec::vec::BitVec;
use std::cmp::Ordering;
use std::{
    fmt::{Display, Formatter},
    hash::Hash,
};
use thiserror::Error;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// Represents an oriented cut in a graph.
///
/// A cut is a partition of the graph's half-edges into two sets, typically
/// defining a boundary. This `OrientedCut` explicitly stores half-edges
/// considered to be on the "left" and "right" sides of this boundary.
/// The orientation implies a direction across the cut.
///
/// For a given full edge (composed of two half-edges, say `h1` and `h2` where
/// `h2 = inv(h1)`), if `h1` is in `left`, then `h2` should be in `right`,
/// and vice-versa, for the cut to be consistent along that edge.
pub struct OrientedCut {
    /// A bitmask representing the set of half-edges on the "left" side of the cut.
    /// In a scattering diagram context, these might be the edges one would drag
    /// to the right to pass through the cut.
    #[cfg_attr(feature = "bincode", bincode(with_serde))]
    pub left: BitVec,
    /// A bitmask representing the set of half-edges on the "right" side of the cut.
    /// These are typically theinvolutional pairs of the half-edges in the `left` set.
    /// In a scattering diagram context, these might be the edges one would drag
    /// to the left.
    ///
    /// It's generally expected that `right` contains `inv(h)` for every `h` in `left`.
    #[cfg_attr(feature = "bincode", bincode(with_serde))]
    pub right: BitVec,
}

impl Display for OrientedCut {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, c) in self.left.iter().enumerate() {
            if *c {
                write!(f, "+{i}")?;
            }
            if self.right[i] {
                write!(f, "-{i}")?;
            }
        }
        Ok(())
    }
}

/// Errors that can occur during operations related to graph cuts.
#[derive(Debug, Error)]
pub enum CutError {
    #[error("Invalid edge: The specified edge is not suitable for the cut operation.")]
    /// Indicates an edge is invalid for the context of a cut operation.
    InvalidEdge,
    #[error("Invalid orientation: The orientation specified or found is not valid for the cut operation.")]
    /// Indicates an orientation is invalid in the context of a cut.
    InvalidOrientation,
    #[error("Cut edge has already been set: The edge is already part of the cut in an incompatible way.")]
    /// Attempted to define a cut edge that was already defined, or its inverse was.
    CutEdgeAlreadySet,
    #[error("Cut edge is identity: An unpaired (identity) half-edge cannot form part of a separating cut in this context.")]
    /// An attempt was made to use an identity (unpaired) half-edge in a cut where a paired edge was expected.
    CutEdgeIsIdentity,
}

impl OrientedCut {
    /// disregards identity edges
    pub fn from_underlying_coerce<E, V, H, N: NodeStorageOps<NodeData = V>>(
        cut: BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Result<Self, CutError> {
        let mut right = graph.empty_subgraph::<BitVec>();

        for i in cut.included_iter() {
            let invh = graph.inv(i);
            if cut.includes(&invh) {
                return Err(CutError::CutEdgeAlreadySet);
            } else if invh == i {
                right.set(i.0, false);
            }
            right.set(invh.0, true);
        }

        cut.subtract(&right);
        Ok(OrientedCut { left: cut, right })
    }

    /// Errors for identity edges
    pub fn from_underlying_strict<E, V, H, N: NodeStorageOps<NodeData = V>>(
        cut: BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Result<Self, CutError> {
        let mut right = graph.empty_subgraph::<BitVec>();

        for i in cut.included_iter() {
            let invh = graph.inv(i);
            if cut.includes(&invh) {
                return Err(CutError::CutEdgeAlreadySet);
            } else if invh == i {
                return Err(CutError::CutEdgeIsIdentity);
            }
            right.set(invh.0, true);
        }
        Ok(OrientedCut { left: cut, right })
    }

    pub fn iter_edges_flow<'a, E, V, H, N: NodeStorageOps<NodeData = V>>(
        &'a self,
        graph: &'a HedgeGraph<E, V, H, N>,
    ) -> impl Iterator<Item = (Flow, EdgeIndex, EdgeData<&'a E>)> {
        graph.iter_edges_of(&self.left).filter_map(|(pair, b, c)| {
            if let HedgePair::Split { split, .. } = pair {
                Some((split, b, c))
            } else {
                None
            }
        })
    }

    /// takes all non-cut edges and gives all possible signs to them.
    ///
    /// If C is a set of edges (pairs of half-edges), then take all S in Pset(C), and put them to the left of the cut. All other edges are put to the right.
    pub fn all_initial_state_cuts<E, V, H, N: NodeStorageOps<NodeData = V>>(
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Vec<Self> {
        let mut all_cuts = Vec::new();

        for c in graph.non_cut_edges() {
            if c.count_ones() == 0 {
                continue;
            }
            let mut all_sources = graph.empty_subgraph::<BitVec>();

            for h in c.included_iter() {
                match graph.edge_store.inv_full(h) {
                    InvolutiveMapping::Identity { .. } => {
                        panic!("cut edge is identity")
                    }
                    InvolutiveMapping::Source { .. } => {
                        all_sources.set(h.0, true);
                    }
                    InvolutiveMapping::Sink { .. } => {}
                }
            }

            let n_cut_edges: u8 = all_sources.count_ones().try_into().unwrap();

            let pset = PowersetIterator::new(n_cut_edges); //.unchecked_sub(1)

            for i in pset {
                let mut left = graph.empty_subgraph::<BitVec>();
                for (j, h) in all_sources.included_iter().enumerate() {
                    // if let Some(j) = j.checked_sub(1) {
                    if i[j] {
                        left.set(h.0, true);
                    } else {
                        left.set(graph.inv(h).0, true);
                    }
                }
                all_cuts.push(Self::from_underlying_strict(left, graph).unwrap());
            }
        }

        all_cuts
    }

    pub fn winding_number(&self, cycle: &Cycle) -> i32 {
        let mut winding_number = 0;

        for h in cycle.filter.included_iter() {
            winding_number += SignOrZero::from(self.relative_orientation(h)) * 1;
        }

        winding_number
    }
    pub fn iter_edges<'a, E, V, H, N: NodeStorageOps<NodeData = V>>(
        &'a self,
        graph: &'a HedgeGraph<E, V, H, N>,
    ) -> impl Iterator<Item = (Orientation, EdgeData<&'a E>)> {
        self.left
            .included_iter()
            .map(|i| (self.orientation(i, graph), graph.get_edge_data_full(i)))
    }

    pub fn iter_edges_idx_relative(&self) -> impl Iterator<Item = (Hedge, Orientation)> + '_ {
        self.left
            .included_iter()
            .map(|i| (i, self.relative_orientation(i)))
    }

    pub fn iter_left_hedges(&self) -> impl Iterator<Item = Hedge> + '_ {
        self.left.included_iter()
    }

    pub fn iter_right_hedges(&self) -> impl Iterator<Item = Hedge> + '_ {
        self.right.included_iter()
    }

    /// - If the left and right are aligned=> panic
    /// - If in left set => Default
    /// - If in right set => Reversed
    /// - If in neither => Undirected
    pub fn relative_orientation(&self, i: Hedge) -> Orientation {
        match (self.left.includes(&i), self.right.includes(&i)) {
            (true, true) => panic!("Both left and right are included in the reference"),
            (true, false) => Orientation::Default,
            (false, true) => Orientation::Reversed,
            (false, false) => Orientation::Undirected,
        }
    }

    /// essentially tells you if the edge is in the cut or not ([Orientation::Undirected])
    ///
    /// If it is in the cut, and the left set contains the source hedge [Orientation::Default] or the sink hedge [Orientation::Reversed].
    ///
    /// Equivalently tells you if the source hedge is in the left side of the cut [Orientation::Default] or the right side [Orientation::Reversed].
    pub fn get_from_pair(&self, pair: HedgePair) -> Orientation {
        match pair {
            HedgePair::Paired { source, sink } => {
                debug_assert!(
                    (self.left.includes(&source) && !self.left.includes(&sink))
                        || (!self.left.includes(&source) && self.left.includes(&sink))
                );
                if self.left.includes(&source) {
                    // debug_assert!(self.right.includes(&sink));
                    Orientation::Default
                } else if self.left.includes(&sink) {
                    // debug_assert!(self.right.includes(&source));
                    Orientation::Reversed
                } else {
                    Orientation::Undirected
                }
            }
            HedgePair::Split {
                source,
                sink,
                split,
            } => {
                debug_assert!(
                    (self.left.includes(&source) && !self.left.includes(&sink))
                        || (!self.left.includes(&source) && self.left.includes(&sink))
                );
                match split {
                    Flow::Sink => {
                        if self.left.includes(&sink) {
                            // debug_assert!(self.right.includes(&source));
                            Orientation::Reversed
                        } else {
                            Orientation::Undirected
                        }
                    }
                    Flow::Source => {
                        if self.left.includes(&source) {
                            // debug_assert!(self.right.includes(&sink));
                            Orientation::Default
                        } else {
                            Orientation::Undirected
                        }
                    }
                }
            }
            HedgePair::Unpaired { hedge, .. } => {
                if self.left.includes(&hedge) {
                    Orientation::Default
                } else if self.right.includes(&hedge) {
                    Orientation::Reversed
                } else {
                    Orientation::Undirected
                }
            }
        }
    }

    /// Set the left cut containing the [Flow] hedge.
    pub fn set(&mut self, pair: HedgePair, flow: Flow) {
        match pair {
            HedgePair::Paired { source, sink } => match flow {
                Flow::Source => {
                    self.left.set(source.0, true);
                    self.right.set(source.0, false);
                    self.left.set(sink.0, false);
                    self.right.set(sink.0, true);
                }
                Flow::Sink => {
                    self.left.set(sink.0, true);
                    self.right.set(sink.0, false);
                    self.left.set(source.0, false);
                    self.right.set(source.0, true);
                }
            },
            HedgePair::Unpaired { hedge, .. } => match flow {
                Flow::Source => {
                    self.left.set(hedge.0, true);
                    self.right.set(hedge.0, false);
                }
                Flow::Sink => {
                    self.left.set(hedge.0, false);
                    self.right.set(hedge.0, true);
                }
            },
            _ => {}
        }
    }

    /// essentially tells you if the edge is in the cut or not ([Orientation::Undirected])
    ///
    /// If it is in the cut, and the left set contains the source hedge [Orientation::Default] or the sink hedge [Orientation::Reversed].
    ///
    /// Equivalently tells you if the source hedge is in the left side of the cut [Orientation::Default] or the right side [Orientation::Reversed].
    pub fn orientation<E, V, H, N: NodeStorage<NodeData = V>>(
        &self,
        i: Hedge,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Orientation {
        let pair = graph.edge_store.pair(i);
        self.get_from_pair(pair)
    }

    pub fn cut_edge<E>(&self, data: E, pair: HedgePair, id: EdgeIndex) -> PossiblyCutEdge<E> {
        let mut edge = PossiblyCutEdge::uncut(data, id);

        let orientation = self.get_from_pair(pair);
        match orientation {
            Orientation::Default => {
                // println!("Source:{id:?}");
                edge.cut(Flow::Source);
            }
            Orientation::Reversed => {
                // println!("Sink:{id:?}");
                edge.cut(Flow::Sink);
            }
            Orientation::Undirected => {}
        }

        edge
    }
}

impl Inclusion<Hedge> for OrientedCut {
    fn includes(&self, other: &Hedge) -> bool {
        self.left.includes(other)
    }
    fn intersects(&self, other: &Hedge) -> bool {
        self.left.intersects(other)
    }
}

impl Inclusion<HedgePair> for OrientedCut {
    fn includes(&self, other: &HedgePair) -> bool {
        self.left.includes(other)
    }
    fn intersects(&self, other: &HedgePair) -> bool {
        self.left.intersects(other)
    }
}

impl Inclusion<BitVec> for OrientedCut {
    fn includes(&self, other: &BitVec) -> bool {
        self.left.includes(other)
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.left.intersects(other)
    }
}

impl Inclusion<OrientedCut> for OrientedCut {
    fn includes(&self, other: &OrientedCut) -> bool {
        self.left.includes(&other.left)
    }

    fn intersects(&self, other: &OrientedCut) -> bool {
        self.left.intersects(&other.left)
    }
}
impl Inclusion<Range<Hedge>> for OrientedCut {
    fn includes(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeTo<Hedge>> for OrientedCut {
    fn includes(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeToInclusive<Hedge>> for OrientedCut {
    fn includes(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeFrom<Hedge>> for OrientedCut {
    fn includes(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeInclusive<Hedge>> for OrientedCut {
    fn includes(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).any(|a| self.includes(&Hedge(a)))
    }
}

impl SubGraph for OrientedCut {
    type Base = BitVec;

    type BaseIter<'a> = SubGraphHedgeIter<'a>;
    fn nedges<E, V, H, N: NodeStorage<NodeData = V>>(
        &self,
        _graph: &HedgeGraph<E, V, H, N>,
    ) -> usize {
        self.nhedges()
    }

    fn size(&self) -> usize {
        self.left.len()
    }

    fn has_greater(&self, hedge: Hedge) -> bool {
        self.left.has_greater(hedge) || self.right.has_greater(hedge)
    }

    fn join_mut(&mut self, other: Self) {
        self.left.join_mut(other.left);
        self.right.join_mut(other.right);
    }

    fn included(&self) -> &BitVec {
        self.left.included()
    }

    fn included_iter(&self) -> Self::BaseIter<'_> {
        self.left.included_iter()
    }

    fn nhedges(&self) -> usize {
        self.left.nhedges()
    }
    fn empty(size: usize) -> Self {
        OrientedCut {
            left: BitVec::empty(size),
            right: BitVec::empty(size),
        }
    }

    fn hairs(&self, node: impl Iterator<Item = Hedge>) -> BitVec {
        self.left.hairs(node)
    }

    fn is_empty(&self) -> bool {
        self.left.count_ones() == 0
    }

    fn string_label(&self) -> String {
        self.left.string_label()
    }
}

impl OrientedCut {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[cfg(feature = "drawing")]
    /// draws hedges in the left set on the right side of the diagram
    /// draws hedges in the right set on the left side of the diagram (so they are on the correct side of the cut)
    ///
    ///
    pub fn layout<E, V, H>(
        self,
        graph: &HedgeGraph<E, V, H, NodeStorageVec<V>>,
        params: LayoutParams,
        iters: LayoutIters,
        edge: f64,
    ) -> HedgeGraph<
        LayoutEdge<PossiblyCutEdge<&'_ E>>,
        LayoutVertex<&'_ V>,
        &'_ H,
        NodeStorageVec<LayoutVertex<&'_ V>>,
    > {
        use indexmap::IndexMap;

        use crate::half_edge::involution::HedgePair;

        let mut left = vec![];
        let mut leftright_map = IndexMap::new();
        let mut right = vec![];

        let graph = self.to_owned_graph_ref(graph);

        for (p, d, i) in graph.iter_edges() {
            if let Some(flow) = i.data.flow() {
                if let HedgePair::Unpaired { .. } = p {
                    match flow {
                        Flow::Sink => {
                            leftright_map
                                .entry(i.data.index)
                                .or_insert_with(|| [Some(d), None])[0] = Some(d)
                        }
                        Flow::Source => {
                            leftright_map
                                .entry(i.data.index)
                                .or_insert_with(|| [None, Some(d)])[1] = Some(d)
                        }
                    }
                }
            }
        }

        for (_, [i, j]) in leftright_map {
            if let Some(i) = i {
                left.push(i);
            }
            if let Some(j) = j {
                right.push(j);
            }
        }

        let settings = LayoutSettings::left_right_square(&graph, params, iters, edge, left, right);

        // println!("{:?}", settings);
        graph.layout(settings)
    }

    /// Take the graph and split it along the cut, putting the cut orientation, and original edge index as additional data.
    pub fn to_owned_graph_ref<E, V, H, N: NodeStorageOps<NodeData = V>>(
        self,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> HedgeGraph<PossiblyCutEdge<&E>, &V, &H, N::OpStorage<&V>> {
        let mut new_graph = graph.map_data_ref(
            |_, _, v| v,
            |_, i, _, e| e.map(|d| PossiblyCutEdge::uncut(d, i)),
            |_, h| h,
        );
        for h in self.iter_left_hedges() {
            new_graph[[&h]].cut(Flow::Source);
            let data = EdgeData::new(new_graph[[&h]].reverse(), new_graph.orientation(h));
            let invh = new_graph.inv(h);
            new_graph.split_edge(invh, data).unwrap();
        }

        new_graph
    }

    /// Take the graph and split it along the cut, putting the cut orientation, and original edge index as additional data.
    pub fn to_owned_graph<E, V, H, N: NodeStorageOps<NodeData = V>>(
        self,
        graph: HedgeGraph<E, V, H, N>,
    ) -> HedgeGraph<PossiblyCutEdge<E>, V, H, N::OpStorage<V>> {
        let mut new_graph = graph.map(
            |_, _, v| v,
            |i, _, h, _, e| {
                e.map(|d| {
                    let h = i[h.any_hedge()];
                    PossiblyCutEdge::uncut(d, h)
                })
            },
            |_, h| h,
        );
        for h in self.iter_left_hedges() {
            new_graph[[&h]].cut(Flow::Source);
            let data = EdgeData::new(
                new_graph[[&h]].duplicate_without_data().reverse(),
                new_graph.orientation(h),
            );
            let invh = new_graph.inv(h);
            new_graph.split_edge(invh, data).unwrap();
        }

        new_graph
    }
}

pub type CutGraph<E, V, H, N> = HedgeGraph<PossiblyCutEdge<E>, V, H, N>;

impl<E, V, H, N: NodeStorageOps<NodeData = V>> CutGraph<E, V, H, N> {
    pub fn split(&mut self) {
        let cut = self.cut();
        for h in cut.iter_left_hedges() {
            self[[&h]].cut(Flow::Source);
            let data = EdgeData::new(
                self[[&h]].duplicate_without_data().reverse(),
                self.orientation(h),
            );
            let invh = self.inv(h);
            self.split_edge(invh, data).unwrap();
        }
    }

    pub fn split_clone(&mut self)
    where
        E: Clone,
    {
        let cut = self.cut();
        for h in cut.iter_left_hedges() {
            self[[&h]].cut(Flow::Source);
            let data = EdgeData::new(self[[&h]].clone().reverse(), self.orientation(h));
            let invh = self.inv(h);
            self.split_edge(invh, data).unwrap();
        }
    }

    pub fn split_copy(&mut self)
    where
        E: Copy,
    {
        let cut = self.cut();
        for h in cut.iter_left_hedges() {
            self[[&h]].cut(Flow::Source);
            let data = EdgeData::new(self[[&h]].reverse(), self.orientation(h));
            let invh = self.inv(h);
            self.split_edge(invh, data).unwrap();
        }
    }

    pub fn round_trip_split(&mut self) {
        self.glue_back_strict();
        self.split();
    }

    pub fn round_trip_glue(&mut self) {
        self.split();
        self.glue_back_strict();
    }

    pub fn cut(&self) -> OrientedCut {
        let mut cut = OrientedCut::empty(self.n_hedges());
        for (h, _, d) in self.iter_edges() {
            match d.data.flow {
                Orientation::Default => cut.set(h, Flow::Source),
                Orientation::Reversed => cut.set(h, Flow::Sink),
                Orientation::Undirected => {}
            }
        }

        cut
    }

    pub fn debug_cut_dot(&self) -> String {
        self.dot_impl(
            &self.full_filter(),
            "",
            &|_| None,
            &|a| Some(format!("label=\"{}\"", a.label())),
            &|_| Some("".to_string()),
        )
    }

    /// glues_back a cut graph. only matches edges with the same cut data, reversed cut flow, and compatible orientation and flow.
    pub fn glue_back_strict(&mut self) {
        self.sew(
            |lf, ld, rf, rd| {
                if lf == -rf {
                    if ld.orientation == rd.orientation {
                        ld.data.matches(rd.data)
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            |lf, ld, rf, rd| {
                let lo: Flow = ld.data.flow.try_into().unwrap();
                let ro: Flow = rd.data.flow.try_into().unwrap();
                debug_assert_eq!(lo, -ro);
                debug_assert_eq!(lf, -rf);

                let mut data = ld.data.merge(rd.data).unwrap();

                let orientation = ld.orientation;
                match (lf, lo) {
                    // A source hedge on the right of the cut
                    // Means that edge data needs to say Orientation::Reversed,
                    // so we need a relative difference in the flow
                    (Flow::Source, Flow::Sink) => {
                        data.cut(Flow::Sink);
                        (Flow::Source, EdgeData::new(data, orientation))
                    }
                    // A sink hedge on the right of the cut
                    // Means that edge data needs to say Orientation::Default
                    // so we need an alignment in the flow
                    (Flow::Sink, Flow::Source) => {
                        data.cut(Flow::Sink);
                        (Flow::Sink, EdgeData::new(data, orientation))
                    }
                    // A source hedge on the left of the cut
                    // Means that edge data needs to say Orientation::Default
                    // so we need an alignment in the flow
                    (Flow::Source, Flow::Source) => {
                        data.cut(Flow::Source);
                        (Flow::Source, EdgeData::new(data, orientation))
                    }
                    // A sink hedge on the left of the cut
                    // Means that edge data needs to say Orientation::Reversed
                    // so we need a relative difference in the flow
                    (Flow::Sink, Flow::Sink) => {
                        data.cut(Flow::Source);
                        (Flow::Sink, EdgeData::new(data, orientation))
                    }
                }
            },
        )
        .unwrap()
    }

    /// glues_back a cut graph. only matches edges with the same cut data, reversed cut flow, and compatible orientation and without flow matching.
    pub fn glue_back_lenient(&mut self) {
        self.sew(
            |lf, ld, rf, rd| {
                if lf == -rf {
                    if ld.orientation == rd.orientation {
                        ld.data.matches(rd.data)
                    } else {
                        false
                    }
                } else if ld.orientation == rd.orientation.reverse() {
                    ld.data.matches(rd.data)
                } else {
                    false
                }
            },
            |lf, ld, _, rd| {
                let lo: Flow = ld.data.flow.try_into().unwrap();
                let ro: Flow = rd.data.flow.try_into().unwrap();
                debug_assert_eq!(lo, -ro);
                // debug_assert_eq!(lf, -rf);

                let mut data = ld.data.merge(rd.data).unwrap();

                let orientation = ld.orientation;
                match (lf, lo) {
                    // A source hedge on the right of the cut
                    // Means that edge data needs to say Orientation::Reversed,
                    // so we need a relative difference in the flow
                    (Flow::Source, Flow::Sink) => {
                        data.cut(Flow::Sink);
                        (Flow::Source, EdgeData::new(data, orientation))
                    }
                    // A sink hedge on the right of the cut
                    // Means that edge data needs to say Orientation::Default
                    // so we need an alignment in the flow
                    (Flow::Sink, Flow::Source) => {
                        data.cut(Flow::Sink);
                        (Flow::Sink, EdgeData::new(data, orientation))
                    }
                    // A source hedge on the left of the cut
                    // Means that edge data needs to say Orientation::Default
                    // so we need an alignment in the flow
                    (Flow::Source, Flow::Source) => {
                        data.cut(Flow::Source);
                        (Flow::Source, EdgeData::new(data, orientation))
                    }
                    // A sink hedge on the left of the cut
                    // Means that edge data needs to say Orientation::Reversed
                    // so we need a relative difference in the flow
                    (Flow::Sink, Flow::Sink) => {
                        data.cut(Flow::Source);
                        (Flow::Sink, EdgeData::new(data, orientation))
                    }
                }
            },
        )
        .unwrap()
    }
}

#[derive(Debug, Eq, Clone, Copy)]
/// Represents an edge that may or may not be part of a cut, and if it is,
/// stores its orientation relative to that cut.
///
/// This struct is typically used as the edge data type `E` in a `HedgeGraph`
/// (often aliased as `CutGraph`) when the graph has been "split" along an
/// [`OrientedCut`]. It allows each edge to remember its original `EdgeIndex`
/// and how it relates to the cut.
///
/// # Type Parameters
///
/// - `E`: The original custom data type of the edge before being potentially marked as cut.
pub struct PossiblyCutEdge<E> {
    /// The original custom data of the edge. This is `None` if the edge was
    /// created as a conceptual part of a split without original data (e.g., the
    /// "other side" of a split external edge).
    data: Option<E>,
    /// The orientation of this edge relative to a cut.
    /// - [`Orientation::Undirected`]: The edge is not part of the cut.
    /// - [`Orientation::Default`]: The edge crosses the cut in one direction (e.g., "left to right", or "source side").
    /// - [`Orientation::Reversed`]: The edge crosses the cut in the opposite direction (e.g., "right to left", or "sink side").
    flow: Orientation,
    /// The original [`EdgeIndex`] of this edge in the graph before any cut operations.
    /// This helps in identifying the edge across different graph representations or
    /// after operations like splitting and gluing.
    pub index: EdgeIndex,
}
impl<E> From<PossiblyCutEdge<E>> for DotEdgeData
where
    DotEdgeData: From<E>,
{
    fn from(value: PossiblyCutEdge<E>) -> Self {
        let mut statements = DotEdgeData::empty();
        if let Some(data) = value.data {
            let data_statements = DotEdgeData::from(data);
            statements.extend(data_statements);
        }

        statements.add_statement(
            "cut_flow",
            match value.flow {
                Orientation::Default => "aligned",
                Orientation::Reversed => "reversed",
                Orientation::Undirected => "uncut",
            },
        );

        let edge_id: usize = value.index.into();
        statements.add_statement("edge_id", edge_id.to_string());

        statements
    }
}

impl<E: TryFrom<DotEdgeData>> TryFrom<DotEdgeData> for PossiblyCutEdge<E> {
    type Error = String;
    fn try_from(dot_edge_data: DotEdgeData) -> Result<Self, Self::Error> {
        let flow = dot_edge_data
            .statements
            .get("cut_flow")
            .ok_or("Missing 'cut_flow' attribute")?;

        let flow = match flow.as_str() {
            "aligned" => Orientation::Default,
            "reversed" => Orientation::Reversed,
            "uncut" => Orientation::Undirected,
            _ => return Err("Invalid 'cut_flow' value".to_string()),
        };

        let edge_id = dot_edge_data
            .statements
            .get("edge_id")
            .ok_or("Missing 'edge_id' attribute")?;
        let edge_id: usize = edge_id
            .parse()
            .map_err(|_| "Invalid 'edge_id' value".to_string())?;

        let data = dot_edge_data.try_into().ok();

        Ok(PossiblyCutEdge {
            data,
            flow,
            index: edge_id.into(),
        })
    }
}

impl<E: Hash> Hash for PossiblyCutEdge<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (&self.data, self.flow).hash(state);
    }
}

impl<E: PartialOrd> PartialOrd for PossiblyCutEdge<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.data, self.flow).partial_cmp(&(&other.data, other.flow))
    }
}

impl<E: Ord> Ord for PossiblyCutEdge<E> {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.data, self.flow).cmp(&(&other.data, other.flow))
    }
}

impl<E: PartialEq> PartialEq for PossiblyCutEdge<E> {
    fn eq(&self, other: &Self) -> bool {
        (&self.data, self.flow).eq(&(&other.data, other.flow))
    }
}

impl<E> PossiblyCutEdge<E> {
    pub fn edge_data(&self) -> &E {
        self.data.as_ref().unwrap()
    }

    pub fn set_data(&mut self, data: E) {
        self.data = Some(data);
    }

    pub fn map<F, T>(self, f: F) -> PossiblyCutEdge<T>
    where
        F: FnOnce(E) -> T,
    {
        PossiblyCutEdge {
            data: self.data.map(f),
            flow: self.flow,
            index: self.index,
        }
    }

    pub fn as_ref(&self) -> PossiblyCutEdge<&E> {
        PossiblyCutEdge {
            data: self.data.as_ref(),
            flow: self.flow,
            index: self.index,
        }
    }

    pub fn duplicate_without_data(&self) -> Self {
        Self {
            data: None,
            flow: self.flow,
            index: self.index,
        }
    }

    pub fn reverse_mut(&mut self) {
        self.flow = self.flow.reverse();
    }

    pub fn reverse(self) -> Self {
        Self {
            data: self.data,
            flow: self.flow.reverse(),
            index: self.index,
        }
    }

    pub fn matches(&self, other: &Self) -> bool {
        self.flow == other.flow.reverse() && self.index == other.index
    }

    pub fn merge(self, other: Self) -> Option<Self> {
        if self.matches(&other) {
            Some(Self {
                data: Some(self.data.or(other.data)?),
                flow: self.flow,
                index: self.index,
            })
        } else {
            None
        }
    }

    pub fn flow(&self) -> Option<Flow> {
        self.flow.try_into().ok()
    }

    pub fn is_cut(&self) -> bool {
        !matches!(self.flow, Orientation::Undirected)
    }

    pub fn label(&self) -> String {
        let mut label = format!("{}", self.index);

        match self.flow {
            Orientation::Default => label.push_str("(left)"),
            Orientation::Reversed => label.push_str("(right)"),
            _ => {}
        }
        label
    }

    pub fn uncut(data: E, index: EdgeIndex) -> Self {
        Self {
            data: Some(data),
            flow: Orientation::Undirected,
            index,
        }
    }

    pub fn cut(&mut self, flow: Flow) {
        self.flow = flow.into();
    }
}

#[cfg(test)]
#[cfg(feature = "drawing")]
pub mod test {
    use super::*;
    use crate::{dot, parser::DotGraph};
    // use similar_asserts::assert_eq;
    //

    #[test]
    fn cut_assembly() {
        let twocycle: DotGraph = dot!(
        digraph{
            a->b
            a->b [dir=back]
        })
        .unwrap();

        // println!("{}", twocycle.dot_display(&twocycle.full_filter()));

        let mut cut_all_source = OrientedCut::empty(twocycle.n_hedges());
        let mut cut_all_sink = OrientedCut::empty(twocycle.n_hedges());
        let mut cut_sink_source = OrientedCut::empty(twocycle.n_hedges());
        let mut cut_source_sink = OrientedCut::empty(twocycle.n_hedges());

        for (p, e, _) in twocycle.iter_edges() {
            cut_all_source.set(p, Flow::Source);
            cut_all_sink.set(p, Flow::Sink);

            if e.0 % 2 == 0 {
                cut_sink_source.set(p, Flow::Sink);
                cut_source_sink.set(p, Flow::Source);
            } else {
                cut_sink_source.set(p, Flow::Source);
                cut_source_sink.set(p, Flow::Sink);
            }
        }

        let cuts = vec![
            cut_all_source,
            cut_all_sink,
            cut_sink_source,
            cut_source_sink,
        ];

        for cut in cuts {
            let mut cut_graph = cut.clone().to_owned_graph_ref(&twocycle);
            // cut.round_trip();

            let mut cut_aligned = cut_graph.clone();

            cut_aligned.align_underlying_to_superficial();

            let occut = cut_graph.cut();
            let occut_aligned = cut_aligned.cut();

            cut_graph.glue_back_strict();
            cut_aligned.glue_back_strict();

            let ocut = cut_graph.cut();
            let ocut_aligned = cut_aligned.cut();

            assert_eq!(ocut, ocut_aligned);
            assert_eq!(occut, occut_aligned);
            assert_eq!(occut, ocut);
            assert_eq!(cut, ocut);

            let a = ocut.clone().layout(
                &cut_graph,
                LayoutParams::default(),
                LayoutIters {
                    n_iters: 10,
                    temp: 1.,
                    seed: 1,
                },
                10.,
            );

            for h in ocut.left.included_iter() {
                assert!(a[[&h]].pos().x > 0.);
            }

            for h in ocut.right.included_iter() {
                assert!(a[[&h]].pos().x < 0.);
            }

            let original = cut_graph.clone();
            let original_aligned = cut_aligned.clone();

            cut_graph.round_trip_glue();
            cut_aligned.round_trip_glue();

            assert_eq!(
                cut_graph,
                original,
                "{}\n//not equal to original\n{}",
                cut_graph.debug_cut_dot(),
                original.debug_cut_dot()
            );
            assert_eq!(
                cut_aligned,
                original_aligned,
                "{}\n//not equal to original\n{}",
                cut_aligned.debug_cut_dot(),
                original_aligned.debug_cut_dot()
            );
        }
    }
}
