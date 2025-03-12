use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::ops::Index;

use crate::half_edge::{
    tree::SimpleTraversalTree, Hedge, HedgeGraph, InvolutiveMapping, NodeStorageOps,
};

use super::{node::HedgeNode, Cycle, Inclusion, SubGraph, SubGraphHedgeIter, SubGraphOps};

#[derive(Clone, Debug, Serialize, Deserialize, Eq)]
pub struct InternalSubGraph {
    // cannot be hairy. I.e. it must always have paired hedges.
    // To represent a hairy subgraph, use a ContractedSubGraph
    pub filter: BitVec,
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

impl SubGraph for InternalSubGraph {
    type Base = BitVec;
    type BaseIter<'a> = SubGraphHedgeIter<'a>;
    fn nedges<E, V, N: NodeStorageOps<NodeData = V>>(&self, _graph: &HedgeGraph<E, V, N>) -> usize {
        self.nhedges() / 2
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

    fn hairs(&self, node: &HedgeNode) -> BitVec {
        node.hairs.intersection(&self.filter)
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

    fn complement<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
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
    fn valid_filter<E, V, N: NodeStorageOps<NodeData = V>>(
        filter: &BitVec,
        graph: &HedgeGraph<E, V, N>,
    ) -> bool {
        for i in filter.included_iter() {
            if !filter.includes(&graph.inv(i)) {
                return false;
            }
        }
        true
    }

    pub fn add_edge<E, V, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        hedge: Hedge,
        graph: &HedgeGraph<E, V, N>,
    ) {
        if !graph.edge_store.involution.is_identity(hedge) {
            self.filter.set(hedge.0, true);
            self.filter.set(graph.inv(hedge).0, true);
        }
    }

    pub fn remove_edge<E, V, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        hedge: Hedge,
        graph: &HedgeGraph<E, V, N>,
    ) {
        if !graph.edge_store.involution.is_identity(hedge) {
            self.filter.set(hedge.0, false);
            self.filter.set(graph.inv(hedge).0, false);
        }
    }

    pub fn try_new<E, V, N: NodeStorageOps<NodeData = V>>(
        filter: BitVec,
        graph: &HedgeGraph<E, V, N>,
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

    pub fn cleaned_filter_optimist<E, V, N: NodeStorageOps<NodeData = V>>(
        mut filter: BitVec,
        graph: &HedgeGraph<E, V, N>,
    ) -> Self {
        for (i, m) in graph.edge_store.involution.inv.iter().enumerate() {
            match m {
                InvolutiveMapping::Identity { .. } => filter.set(i, false),
                InvolutiveMapping::Sink { source_idx } => {
                    if filter.includes(&Hedge(i)) {
                        filter.set(source_idx.0, true);
                    }
                }
                InvolutiveMapping::Source { sink_idx, .. } => {
                    if filter.includes(&Hedge(i)) {
                        filter.set(sink_idx.0, true);
                    }
                }
            }
        }
        InternalSubGraph {
            filter,
            loopcount: None,
        }
    }

    pub fn cleaned_filter_pessimist<E, V, N: NodeStorageOps<NodeData = V>>(
        mut filter: BitVec,
        graph: &HedgeGraph<E, V, N>,
    ) -> Self {
        for (i, m) in graph.edge_store.involution.inv.iter().enumerate() {
            match m {
                InvolutiveMapping::Identity { .. } => filter.set(i, false),
                InvolutiveMapping::Sink { source_idx } => {
                    if !filter.includes(&Hedge(i)) {
                        filter.set(source_idx.0, false);
                    }
                }
                InvolutiveMapping::Source { sink_idx, .. } => {
                    if !filter.includes(&Hedge(i)) {
                        filter.set(sink_idx.0, false);
                    }
                }
            }
        }

        InternalSubGraph {
            filter,
            loopcount: None,
        }
    }

    pub fn valid<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> bool {
        Self::valid_filter(&self.filter, graph)
    }

    pub fn to_hairy_subgraph<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> HedgeNode {
        graph.nesting_node_from_subgraph(self.clone())
    }

    pub fn set_loopcount<E, V, N: NodeStorageOps<NodeData = V>>(
        &mut self,
        graph: &HedgeGraph<E, V, N>,
    ) {
        self.loopcount = Some(graph.cyclotomatic_number(self));
    }

    pub fn cycle_basis<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> (Vec<Cycle>, SimpleTraversalTree) {
        let node = graph.base_nodes_iter().next().unwrap();
        graph.paton_cycle_basis(self, &node, None).unwrap()
    }
}
