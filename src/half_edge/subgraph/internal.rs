use bitvec::vec::BitVec;
use std::hash::Hash;
use std::ops::Index;
use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

use crate::half_edge::involution::Flow;
use crate::half_edge::{
    hedgevec::Accessors, involution::HedgePair, tree::SimpleTraversalTree, Hedge, HedgeGraph,
    NodeStorageOps,
};

use super::{
    node::HedgeNode, Cycle, Inclusion, ModifySubgraph, SubGraph, SubGraphHedgeIter, SubGraphOps,
};

#[derive(Clone, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// Represents a subgraph consisting entirely of "internal" edges.
///
/// An internal subgraph is defined such that if a half-edge is part of this subgraph,
/// its opposite half-edge must also be part of the subgraph. This means it cannot
/// contain any "dangling" or "external" half-edges that connect to parts of the
/// graph outside of itself. It represents a self-contained portion of the graph's
/// edge topology.
///
/// To represent a subgraph that does have external connections (or "hairs"),
/// use types like [`HedgeNode`] or [`ContractedSubGraph`].
pub struct InternalSubGraph {
    // cannot be hairy. I.e. it must always have paired hedges.
    // To represent a hairy subgraph, use a ContractedSubGraph
    /// A bitmask representing the set of half-edges included in this internal subgraph.
    /// For every half-edge `h` included in this filter, its opposite `inv(h)` must also
    /// be included.
    #[cfg_attr(feature = "bincode", bincode(with_serde))]
    pub filter: BitVec,
    /// An optional field, often used to cache the cyclomatic number (number of
    /// independent cycles) within this subgraph once computed.
    pub loopcount: Option<usize>,
}

impl InternalSubGraph {
    /// Create a new subgraph from a filter.
    ///
    /// # Safety
    ///
    /// The filter must be valid, i.e. it must always have paired hedges.
    pub unsafe fn new_unchecked(filter: BitVec) -> Self {
        InternalSubGraph {
            filter,
            loopcount: None,
        }
    }
}

impl Hash for InternalSubGraph {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.filter.hash(state);
    }
}

impl PartialEq for InternalSubGraph {
    fn eq(&self, other: &Self) -> bool {
        self.filter == other.filter
    }
}

impl Index<usize> for InternalSubGraph {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        self.filter.index(index)
    }
}

impl PartialOrd for InternalSubGraph {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.filter.clone() | &other.filter == self.filter {
            Some(std::cmp::Ordering::Greater)
        } else if self.filter.clone() | &other.filter == other.filter {
            Some(std::cmp::Ordering::Less)
        } else {
            None
        }
    }
}

impl Inclusion<Hedge> for InternalSubGraph {
    fn includes(&self, hedge_id: &Hedge) -> bool {
        self.filter[hedge_id.0]
    }

    fn intersects(&self, other: &Hedge) -> bool {
        self.includes(other)
    }
}

impl Inclusion<HedgePair> for InternalSubGraph {
    fn includes(&self, other: &HedgePair) -> bool {
        match other {
            HedgePair::Unpaired { .. } => false,
            HedgePair::Split {
                source,
                sink,
                split,
            } => match split {
                Flow::Sink => self.includes(sink),
                Flow::Source => self.includes(source),
            },
            HedgePair::Paired { source, sink } => self.includes(source) && self.includes(sink),
        }
    }

    fn intersects(&self, other: &HedgePair) -> bool {
        match other {
            HedgePair::Unpaired { hedge, .. } => self.includes(hedge),
            HedgePair::Split {
                source,
                sink,
                split,
            } => match split {
                Flow::Sink => self.includes(sink),
                Flow::Source => self.includes(source),
            },
            HedgePair::Paired { source, sink } => self.includes(source) || self.includes(sink),
        }
    }
}

impl Inclusion<InternalSubGraph> for InternalSubGraph {
    fn includes(&self, other: &InternalSubGraph) -> bool {
        self.filter.intersection(&other.filter) == other.filter
    }

    fn intersects(&self, other: &InternalSubGraph) -> bool {
        self.filter.intersection(&other.filter).count_ones() > 0
    }
}

impl Inclusion<BitVec> for InternalSubGraph {
    fn includes(&self, other: &BitVec) -> bool {
        &self.filter.intersection(other) == other
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.filter.intersection(other).count_ones() > 0
    }
}

impl Inclusion<Range<Hedge>> for InternalSubGraph {
    fn includes(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeTo<Hedge>> for InternalSubGraph {
    fn includes(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeToInclusive<Hedge>> for InternalSubGraph {
    fn includes(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeFrom<Hedge>> for InternalSubGraph {
    fn includes(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeInclusive<Hedge>> for InternalSubGraph {
    fn includes(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).any(|a| self.includes(&Hedge(a)))
    }
}

impl SubGraph for InternalSubGraph {
    type Base = BitVec;
    type BaseIter<'a> = SubGraphHedgeIter<'a>;
    fn nedges<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        _graph: &HedgeGraph<E, V, H, N>,
    ) -> usize {
        self.nhedges() / 2
    }

    fn has_greater(&self, hedge: Hedge) -> bool {
        self.filter.has_greater(hedge)
    }

    fn size(&self) -> usize {
        self.filter.size()
    }

    fn join_mut(&mut self, other: Self) {
        self.filter.join_mut(other.filter);
        if let Some(l) = other.loopcount {
            if let Some(sl) = &mut self.loopcount {
                *sl += l;
            } else {
                self.loopcount = other.loopcount
            }
        }
    }

    fn included_iter(&self) -> Self::BaseIter<'_> {
        self.filter.included_iter()
    }

    fn nhedges(&self) -> usize {
        self.filter.count_ones()
    }

    fn empty(size: usize) -> Self {
        InternalSubGraph {
            filter: BitVec::empty(size),
            loopcount: Some(0),
        }
    }

    fn included(&self) -> &BitVec {
        self.filter.included()
    }

    fn string_label(&self) -> String {
        self.filter.string_label()
    }
    fn is_empty(&self) -> bool {
        self.filter.count_ones() == 0
    }
}

impl SubGraphOps for InternalSubGraph {
    fn intersect_with(&mut self, other: &InternalSubGraph) {
        self.filter &= &other.filter;
        self.loopcount = None;
    }

    fn union_with_iter(&mut self, other: impl Iterator<Item = Hedge>) {
        for h in other {
            self.filter.add(h);
        }
    }

    fn union_with(&mut self, other: &InternalSubGraph) {
        self.filter |= &other.filter;
        self.loopcount = None;
    }

    fn sym_diff_with(&mut self, other: &InternalSubGraph) {
        self.filter ^= &other.filter;
        self.loopcount = None;
    }

    fn empty_intersection(&self, other: &InternalSubGraph) -> bool {
        (self.filter.clone() & &other.filter).count_ones() == 0
    }

    fn empty_union(&self, other: &InternalSubGraph) -> bool {
        (self.filter.clone() | &other.filter).count_ones() == 0
    }

    fn complement<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Self {
        InternalSubGraph {
            filter: !self.filter.clone() & !graph.external_filter(),
            loopcount: None,
        }
    }

    fn subtract_with(&mut self, other: &Self) {
        self.filter = !other.filter.clone() & &self.filter;
        self.loopcount = None;
    }
}

impl InternalSubGraph {
    fn valid_filter<E, V, H, N: NodeStorageOps<NodeData = V>>(
        filter: &BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> bool {
        for i in filter.included_iter() {
            if !filter.includes(&graph.inv(i)) {
                return false;
            }
        }
        true
    }

    pub fn add_edge<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        hedge: Hedge,
        graph: &HedgeGraph<E, V, H, N>,
    ) {
        if !graph.edge_store.pair(hedge).is_unpaired() {
            self.filter.set(hedge.0, true);
            self.filter.set(graph.inv(hedge).0, true);
        }
    }

    pub fn remove_edge<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        hedge: Hedge,
        graph: &HedgeGraph<E, V, H, N>,
    ) {
        if !graph.edge_store.pair(hedge).is_unpaired() {
            self.filter.set(hedge.0, false);
            self.filter.set(graph.inv(hedge).0, false);
        }
    }

    pub fn try_new<E, V, H, N: NodeStorageOps<NodeData = V>>(
        filter: BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Option<Self> {
        if filter.len() != graph.n_hedges() {
            return None;
        }
        if !Self::valid_filter(&filter, graph) {
            return None;
        }

        Some(InternalSubGraph {
            filter,
            loopcount: None,
        })
    }

    pub fn cleaned_filter_optimist<E, V, H, N: NodeStorageOps<NodeData = V>>(
        mut filter: BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Self {
        for (j, _, _) in graph.iter_edges() {
            match j {
                HedgePair::Paired { source, sink } => {
                    if filter.includes(&source) || filter.includes(&sink) {
                        filter.set(source.0, true);
                        filter.set(sink.0, true);
                    }
                }
                HedgePair::Unpaired { hedge, .. } => filter.set(hedge.0, false),
                _ => {}
            }
        }
        InternalSubGraph {
            filter,
            loopcount: None,
        }
    }

    pub fn cleaned_filter_pessimist<E, V, H, N: NodeStorageOps<NodeData = V>>(
        mut filter: BitVec,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> Self {
        for (j, _, _) in graph.iter_edges() {
            match j {
                HedgePair::Paired { source, sink } => {
                    if filter.includes(&source) && filter.includes(&sink) {
                        filter.set(source.0, true);
                        filter.set(sink.0, true);
                    }
                }
                HedgePair::Unpaired { hedge, .. } => filter.set(hedge.0, false),
                _ => {}
            }
        }

        InternalSubGraph {
            filter,
            loopcount: None,
        }
    }

    pub fn valid<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> bool {
        Self::valid_filter(&self.filter, graph)
    }

    pub fn to_hairy_subgraph<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> HedgeNode {
        graph.nesting_node_from_subgraph(self.clone())
    }

    pub fn set_loopcount<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        graph: &HedgeGraph<E, V, H, N>,
    ) {
        self.loopcount = Some(graph.cyclotomatic_number(self));
    }

    pub fn cycle_basis<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, H, N>,
    ) -> (Vec<Cycle>, SimpleTraversalTree) {
        let node = graph.iter_node_ids().next().unwrap();
        graph.paton_cycle_basis(self, &node, None).unwrap()
    }
}
