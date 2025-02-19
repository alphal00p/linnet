use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bitvec::{bitvec, order::Lsb0};
use serde::{Deserialize, Serialize};

use crate::half_edge::builder::HedgeNodeBuilder;
use crate::half_edge::nodestorage::NodeStorageOps;
use crate::half_edge::{Hedge, HedgeGraph, NodeStorage};

use super::Inclusion;
use super::{internal::InternalSubGraph, node::HedgeNode, SubGraph, SubGraphOps};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ContractedSubGraph {
    pub internal_graph: InternalSubGraph, // cannot have any external hedges (i.e. unpaired hedges)
    pub allhedges: BitVec,                // all hedges , including that are in the internal graph.
}

impl Inclusion<Hedge> for ContractedSubGraph {
    fn includes(&self, hedge_id: &Hedge) -> bool {
        self.internal_graph.includes(hedge_id) || self.allhedges.includes(hedge_id)
    }

    fn intersects(&self, other: &Hedge) -> bool {
        self.includes(other)
    }
}

impl Inclusion<ContractedSubGraph> for ContractedSubGraph {
    fn includes(&self, other: &ContractedSubGraph) -> bool {
        self.internal_graph.includes(&other.internal_graph)
    }

    fn intersects(&self, other: &ContractedSubGraph) -> bool {
        self.allhedges.intersects(&other.allhedges)
    }
}

impl Inclusion<BitVec> for ContractedSubGraph {
    fn includes(&self, other: &BitVec) -> bool {
        self.internal_graph.includes(other) || self.allhedges.includes(other)
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.allhedges.intersects(other)
    }
}

impl SubGraph for ContractedSubGraph {
    fn nhedges(&self) -> usize {
        self.allhedges.nhedges()
    }

    fn included(&self) -> &BitSlice {
        self.allhedges.included()
    }

    fn nedges<E, V, N: NodeStorageOps<NodeData = V>>(&self, graph: &HedgeGraph<E, V, N>) -> usize {
        self.allhedges.nedges(graph)
    }

    fn hairs(&self, node: &HedgeNode) -> BitVec {
        let mut hairs = self.allhedges.intersection(&node.hairs);
        hairs.subtract_with(&self.internal_graph.filter);
        hairs
    }

    fn string_label(&self) -> String {
        self.allhedges.string_label() + "⊛" + self.internal_graph.string_label().as_str()
    }

    fn is_empty(&self) -> bool {
        self.allhedges.is_empty()
    }
    fn empty(size: usize) -> Self {
        Self {
            internal_graph: InternalSubGraph::empty(size),
            allhedges: BitVec::empty(size),
        }
    }
}

impl SubGraphOps for ContractedSubGraph {
    fn complement<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> Self {
        let externalhedges = !self.allhedges.clone() & !self.internal_graph.filter.clone();

        Self {
            internal_graph: InternalSubGraph::empty(graph.n_hedges()),
            allhedges: externalhedges,
        }
    }

    fn union_with(&mut self, other: &Self) {
        // union is the intersection of the internal graphs, and the union of the external graph.
        self.internal_graph.intersect_with(&other.internal_graph);
        self.allhedges.union_with(&other.allhedges);
    }

    fn intersect_with(&mut self, other: &Self) {
        // intersection is the union of the internal graphs, and the intersection of the external graph.
        self.internal_graph.union_with(&other.internal_graph);
        self.allhedges.intersect_with(&other.allhedges);
    }

    fn sym_diff_with(&mut self, other: &Self) {
        // external hedges that are only present in one of the two graphs.
        // contracted parts unioned
        self.internal_graph.union_with(&other.internal_graph);
        self.allhedges.sym_diff_with(&other.allhedges);
    }

    fn empty_intersection(&self, other: &Self) -> bool {
        self.allhedges.empty_intersection(&other.allhedges)
    }

    fn empty_union(&self, other: &Self) -> bool {
        self.allhedges.empty_union(&other.allhedges)
    }

    fn subtract_with(&mut self, other: &Self) {
        self.internal_graph.union_with(&other.internal_graph);
        self.allhedges.subtract_with(&other.allhedges);
    }
}

impl ContractedSubGraph {
    pub fn all_edges(&self) -> BitVec {
        self.internal_graph.filter.clone() | &self.allhedges
    }

    pub fn weakly_disjoint(&self, other: &ContractedSubGraph) -> bool {
        let internals = self.internal_graph.filter.clone() & &other.internal_graph.filter;

        internals.count_ones() == 0
    }

    pub fn strongly_disjoint(&self, other: &ContractedSubGraph) -> bool {
        let internals = self.internal_graph.filter.clone() & &other.internal_graph.filter;

        let externals_in_self = self.internal_graph.filter.clone() & &other.allhedges;
        let externals_in_other = self.allhedges.clone() & &other.internal_graph.filter;

        internals.count_ones() == 0
            && externals_in_self.count_ones() == 0
            && externals_in_other.count_ones() == 0
    }

    pub fn node_from_pos(pos: &[usize], len: usize) -> ContractedSubGraph {
        ContractedSubGraph {
            allhedges: ContractedSubGraph::filter_from_pos(pos, len),
            internal_graph: InternalSubGraph::empty(len),
        }
    }

    pub fn filter_from_pos(pos: &[usize], len: usize) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; len];

        for &i in pos {
            filter.set(i, true);
        }

        filter
    }

    pub fn internal_graph_union(&self, other: &ContractedSubGraph) -> InternalSubGraph {
        InternalSubGraph {
            filter: self.internal_graph.filter.clone() | &other.internal_graph.filter,
            loopcount: None,
        }
    }

    pub fn from_builder<V>(builder: &HedgeNodeBuilder<V>, len: usize) -> Self {
        let internal_graph = InternalSubGraph::empty(len);
        let mut externalhedges = bitvec![usize, Lsb0; 0; len];

        for hedge in &builder.hedges {
            let mut bit = externalhedges.get_mut(hedge.0).unwrap();
            *bit = true;
        }

        ContractedSubGraph {
            internal_graph,
            allhedges: externalhedges,
        }
    }

    pub fn is_node(&self) -> bool {
        self.internal_graph.is_empty()
    }

    pub fn is_subgraph(&self) -> bool {
        !self.is_node()
    }
}
