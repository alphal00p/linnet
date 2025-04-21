use crate::tree::{parent_pointer::ParentPointerStore, Forest};

use super::{
    builder::HedgeNodeBuilder,
    involution::{EdgeIndex, Hedge, Involution},
    subgraph::{BaseSubgraph, SubGraph},
    HedgeGraph, HedgeGraphError, NodeIndex,
};

pub trait NodeStorageOps: NodeStorage {
    type OpStorage<N>: NodeStorageOps<NodeData = N>;
    type Base: BaseSubgraph;
    // where
    // Self: 'a;
    fn extend(self, other: Self) -> Self;
    fn extend_mut(&mut self, other: Self);
    fn extract<S: SubGraph<Base = Self::Base>, V2>(
        &mut self,
        subgraph: &S,
        split_node: impl FnMut(&Self::NodeData) -> V2,
        owned_node: impl FnMut(Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2>;
    fn swap(&mut self, a: Hedge, b: Hedge);

    // fn add_node(&mut self, node_data: Self::NodeData) -> NodeIndex;

    /// Identifies nodes, essentially turning them into a single node
    /// Invalidates all previous NodeIndex values if using VecNodeStore.
    /// Does not invalidate if using Forest as NodeStore
    fn identify_nodes(&mut self, nodes: &[NodeIndex], node_data_merge: Self::NodeData)
        -> NodeIndex;

    fn forget_identification_history(&mut self) -> Vec<(Self::NodeData, Hedge)> {
        vec![]
    }

    fn to_forest<U, H>(
        &self,
        map_data: impl Fn(&Self::NodeData) -> U,
    ) -> Forest<U, ParentPointerStore<H>>;

    fn build<I: IntoIterator<Item = HedgeNodeBuilder<Self::NodeData>>>(
        nodes: I,
        n_hedges: usize,
    ) -> Self;

    fn add_dangling_edge(self, source: NodeIndex) -> Result<Self, HedgeGraphError>;

    fn random(sources: &[Self::Neighbors], sinks: &[Self::Neighbors]) -> Self
    where
        Self::NodeData: Default;

    fn hedge_len(&self) -> usize;
    fn node_len(&self) -> usize;
    fn drain(self) -> impl Iterator<Item = (NodeIndex, Self::NodeData)>;
    fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Self::NodeData)>;

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError>;

    fn map_data_ref_graph<'a, E, V2>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        node_map: impl FnMut(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            Self::NeighborsIter<'a>,
            &'a Self::NodeData,
        ) -> V2,
    ) -> Self::OpStorage<V2>;

    fn map_data_ref_mut_graph<'a, V2>(
        &'a mut self,
        node_map: impl FnMut(Self::NeighborsIter<'a>, &'a mut Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2>;

    fn map_data_ref_graph_result<'a, E, V2, Er>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        node_map: impl FnMut(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            Self::NeighborsIter<'a>,
            &'a Self::NodeData,
        ) -> Result<V2, Er>,
    ) -> Result<Self::OpStorage<V2>, Er>;

    fn map_data_graph<'a, V2>(
        self,
        involution: &'a Involution<EdgeIndex>,
        f: impl FnMut(&'a Involution<EdgeIndex>, NodeIndex, Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2>;

    fn iter_node_id(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.node_len()).map(NodeIndex)
    }
    fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = (Self::NeighborsIter<'_>, NodeIndex, &Self::NodeData)>;
    fn iter_nodes_mut(
        &mut self,
    ) -> impl Iterator<Item = (Self::NeighborsIter<'_>, NodeIndex, &mut Self::NodeData)>;
    fn node_id_ref(&self, hedge: Hedge) -> NodeIndex;
    fn get_neighbor_iterator(&self, node_id: NodeIndex) -> Self::NeighborsIter<'_>;
    fn get_node_data(&self, node_id: NodeIndex) -> &Self::NodeData;
    fn get_node_data_mut(&mut self, node_id: NodeIndex) -> &mut Self::NodeData;
}

pub trait NodeStorage: Sized {
    type NodeData;
    type Storage<N>;

    type Neighbors: SubGraph;
    type NeighborsIter<'a>: Iterator<Item = Hedge> + Clone
    where
        Self: 'a;
}

mod bitvec_find;
mod forest;
mod vec;

pub use vec::BitVecNeighborIter;
pub use vec::NodeStorageVec;

#[cfg(test)]
mod test;
