use bincode::{Decode, Encode};
use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};

use crate::tree::{
    parent_pointer::{PPNode, ParentPointerStore},
    Forest, RootData, RootId,
};

use super::{
    builder::HedgeNodeBuilder,
    involution::{EdgeIndex, Hedge, Involution},
    subgraph::{HedgeNode, Inclusion, InternalSubGraph, SubGraph, SubGraphOps},
    HedgeGraph, HedgeGraphError, NodeIndex,
};

pub trait NodeStorageOps: NodeStorage {
    type OpStorage<N>: NodeStorageOps<NodeData = N>;
    fn extend(self, other: Self) -> Self;

    // fn add_node(&mut self, node_data: Self::NodeData) -> NodeIndex;

    /// Identifies nodes, essentially turning them into a single node
    /// Invalidates all previous NodeIndex values
    fn identify_nodes(&mut self, nodes: &[NodeIndex], node_data_merge: Self::NodeData)
        -> NodeIndex;

    fn to_forest<U>(
        &self,
        map_data: impl Fn(&Self::NodeData) -> U,
    ) -> Forest<U, ParentPointerStore<()>>;

    fn build<I: IntoIterator<Item = HedgeNodeBuilder<Self::NodeData>>>(
        nodes: I,
        n_hedges: usize,
    ) -> Self;

    fn add_dangling_edge(self, source: NodeIndex) -> Result<Self, HedgeGraphError>;

    fn random(sources: &[HedgeNode], sinks: &[HedgeNode]) -> Self
    where
        Self::NodeData: Default;

    fn hedge_len(&self) -> usize;
    fn node_len(&self) -> usize;
    fn drain(self) -> impl Iterator<Item = (NodeIndex, Self::NodeData)>;
    fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Self::NodeData)>;

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError>;
    fn set_hedge_data(&mut self, hedge: Hedge, nodeid: NodeIndex);

    fn map_data_ref_graph<'a, E, V2>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        node_map: &impl Fn(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            &'a Self::NodeData,
            &'a HedgeNode,
        ) -> V2,
    ) -> Self::OpStorage<V2>;

    fn map_data_graph<V2>(
        self,
        involution: &Involution<EdgeIndex>,
        f: impl FnMut(&Involution<EdgeIndex>, &HedgeNode, NodeIndex, Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2>;

    fn iter_node_id(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.node_len()).map(NodeIndex)
    }
    fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &Self::NodeData)>;
    fn node_id_ref(&self, hedge: Hedge) -> NodeIndex;
    fn get_node(&self, node_id: NodeIndex) -> &HedgeNode;
    fn get_node_data(&self, node_id: NodeIndex) -> &Self::NodeData;
    fn get_node_data_mut(&mut self, node_id: NodeIndex) -> &mut Self::NodeData;
}

pub trait NodeStorage: Sized {
    type NodeData;
    type Storage<N>;
}

#[derive(
    Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct NodeStorageVec<N> {
    pub(crate) node_data: Vec<N>,
    pub(crate) hedge_data: Vec<NodeIndex>,
    pub(crate) nodes: Vec<HedgeNode>, // Nodes
}

impl<N> NodeStorage for NodeStorageVec<N> {
    type NodeData = N;
    type Storage<M> = NodeStorageVec<M>;
}

impl<N> NodeStorageOps for NodeStorageVec<N> {
    type OpStorage<A> = Self::Storage<A>;

    fn node_len(&self) -> usize {
        self.nodes.len()
    }

    // fn add_node(&mut self, node_data: Self::NodeData) -> NodeIndex {
    //     let empty = HedgeNode::empty(self.hedge_len());
    //     self.nodes.push(empty);
    //     self.node_data.push(node_data);
    //     NodeIndex(self.nodes.len() - 1)
    // }

    fn identify_nodes(
        &mut self,
        nodes: &[NodeIndex],
        node_data_merge: Self::NodeData,
    ) -> NodeIndex {
        let mut removed = BitVec::empty(self.nodes.len());
        let mut full_nodes = BitVec::empty(self.hedge_len());

        let empty_internal = InternalSubGraph {
            filter: BitVec::empty(self.hedge_len()),
            loopcount: None,
        };
        for n in nodes {
            removed.set(n.0, true);
            full_nodes.union_with(&self.nodes[n.0].hairs);
        }

        let full_node = HedgeNode {
            hairs: full_nodes,
            internal_graph: empty_internal,
        };

        let replacement = NodeIndex(removed.iter_ones().next().unwrap());

        for r in removed.iter_ones().skip(1).rev() {
            let last_index = self.nodes.len() - 1;

            // Before doing anything, update any hedge pointers that point to the node being removed.
            for hedge in self.hedge_data.iter_mut() {
                if *hedge == NodeIndex(r) {
                    *hedge = replacement;
                }
            }

            if r != last_index {
                // Swap the target with the last element in both vectors.
                self.nodes.swap(r, last_index);
                self.node_data.swap(r, last_index);

                // After swapping, update any hedge pointer that pointed to the moved element.
                // It used to be at last_index, now it is at r.
                for hedge in self.hedge_data.iter_mut() {
                    if *hedge == NodeIndex(last_index) {
                        *hedge = NodeIndex(r);
                    }
                }
            }
            // Remove the (now last) element.

            self.nodes.pop();
            self.node_data.pop();
        }

        self.nodes[replacement.0] = full_node;
        self.node_data[replacement.0] = node_data_merge;

        replacement
    }

    fn to_forest<U>(
        &self,
        map_data: impl Fn(&Self::NodeData) -> U,
    ) -> Forest<U, ParentPointerStore<()>> {
        let mut nodes = vec![None; self.hedge_data.len()];
        let mut roots = vec![];

        for (set, d) in self.nodes.iter().zip(&self.node_data) {
            let mut first = None;
            for i in set.hairs.included_iter() {
                if let Some(root) = first {
                    nodes[i.0] = Some(PPNode::child((), root))
                } else {
                    first = Some(i.into());
                    nodes[i.0] = Some(PPNode::root((), RootId(roots.len())));
                }
            }
            roots.push(RootData {
                root_id: first.unwrap(),
                data: map_data(d),
            });
        }
        Forest {
            nodes: nodes
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .unwrap()
                .into_iter()
                .collect(),
            roots,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Self::NodeData)> {
        self.node_data
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeIndex(i), v))
    }

    fn drain(self) -> impl Iterator<Item = (NodeIndex, Self::NodeData)> {
        self.node_data
            .into_iter()
            .enumerate()
            .map(|(i, v)| (NodeIndex(i), v))
    }
    fn build<I: IntoIterator<Item = HedgeNodeBuilder<N>>>(node_iter: I, n_hedges: usize) -> Self {
        let mut nodes: Vec<HedgeNode> = vec![];
        let mut node_data = vec![];
        let mut hedgedata = vec![None; n_hedges];

        for (i, n) in node_iter.into_iter().enumerate() {
            for h in &n.hedges {
                hedgedata[h.0] = Some(NodeIndex(i));
            }
            nodes.push(HedgeNode::from_builder(&n, n_hedges));
            node_data.push(n.data);
        }

        let hedge_data = hedgedata.into_iter().map(|x| x.unwrap()).collect();

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &Self::NodeData)> {
        self.nodes.iter().zip(self.node_data.iter())
    }

    fn node_id_ref(&self, hedge: Hedge) -> NodeIndex {
        self.hedge_data[hedge.0]
    }

    fn get_node(&self, node_id: NodeIndex) -> &HedgeNode {
        &self.nodes[node_id.0]
    }

    fn get_node_data(&self, node_id: NodeIndex) -> &N {
        &self.node_data[node_id.0]
    }

    fn get_node_data_mut(&mut self, node_id: NodeIndex) -> &mut Self::NodeData {
        &mut self.node_data[node_id.0]
    }

    fn hedge_len(&self) -> usize {
        self.hedge_data.len()
    }

    fn extend(self, other: Self) -> Self {
        let self_empty_filter = BitVec::empty(self.hedge_data.len());
        let other_empty_filter = BitVec::empty(other.hedge_data.len());
        let mut node_data = self.node_data;
        node_data.extend(other.node_data);

        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .map(|mut k| {
                k.hairs.extend(other_empty_filter.clone());
                k.internal_graph.filter.extend(other_empty_filter.clone());
                k
            })
            .chain(other.nodes.into_iter().map(|mut k| {
                let mut new_hairs = self_empty_filter.clone();
                new_hairs.extend(k.hairs.clone());
                k.hairs = new_hairs;

                let mut internal = self_empty_filter.clone();
                internal.extend(k.internal_graph.filter.clone());
                k.internal_graph.filter = internal;

                k
            }))
            .collect();

        let mut hedge_data = self.hedge_data;
        hedge_data.extend(other.hedge_data);

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn add_dangling_edge(self, source: NodeIndex) -> Result<Self, HedgeGraphError> {
        if self.nodes.len() <= source.0 {
            return Err(HedgeGraphError::NoNode);
        }
        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, mut k)| {
                k.internal_graph.filter.push(false);
                if NodeIndex(i) == source {
                    k.hairs.push(true);
                } else {
                    k.hairs.push(false);
                }
                k
            })
            .collect();
        let mut hedge_data = self.hedge_data;
        hedge_data.push(source);

        Ok(NodeStorageVec {
            node_data: self.node_data,
            hedge_data,
            nodes,
        })
    }

    fn random(sources: &[HedgeNode], sinks: &[HedgeNode]) -> Self
    where
        N: Default,
    {
        let mut nodes = Vec::new();
        let mut node_data = Vec::new();

        let mut hedge_data = vec![NodeIndex(0); sources[0].hairs.len()];

        for (nid, n) in sources.iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(N::default());
            for i in n.hairs.included_iter() {
                hedge_data[i.0] = NodeIndex(nid);
            }
        }

        let len = nodes.len();

        for (nid, n) in sinks.iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(N::default());

            for i in n.hairs.included_iter() {
                hedge_data[i.0] = NodeIndex(nid + len);
            }
        }

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn set_hedge_data(&mut self, hedge: Hedge, nodeid: NodeIndex) {
        self.hedge_data[hedge.0] = nodeid;
    }

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError> {
        let mut cover = BitVec::empty(self.hedge_len());
        for (i, node) in self.nodes.iter().enumerate() {
            for h in node.hairs.included_iter() {
                if cover.includes(&h) {
                    return Err(HedgeGraphError::NodesDoNotPartition);
                } else {
                    cover.set(h.0, true);
                    self.hedge_data[h.0] = NodeIndex(i);
                }
            }
        }

        let full = !BitVec::empty(self.hedge_len());

        if cover.sym_diff(&full).count_ones() > 0 {
            return Err(HedgeGraphError::NodesDoNotPartition);
        }

        Ok(())
    }

    fn map_data_ref_graph<'a, E, V2>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        node_map: &impl Fn(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            &'a Self::NodeData,
            &'a HedgeNode,
        ) -> V2,
    ) -> Self::Storage<V2> {
        let node_data = self
            .node_data
            .iter()
            .zip(self.nodes.iter())
            .map(|(v, h)| node_map(graph, v, h))
            .collect();

        NodeStorageVec {
            node_data,
            hedge_data: self.hedge_data.clone(),
            nodes: self.nodes.clone(),
        }
    }

    fn map_data_graph<'a, V2>(
        self,
        involution: &Involution<EdgeIndex>,
        mut f: impl FnMut(&Involution<EdgeIndex>, &HedgeNode, NodeIndex, Self::NodeData) -> V2,
    ) -> Self::Storage<V2> {
        let node_data = self
            .node_data
            .into_iter()
            .zip(self.nodes.iter())
            .enumerate()
            .map(|(i, (v, h))| f(involution, h, NodeIndex(i), v))
            .collect();

        NodeStorageVec {
            node_data,
            hedge_data: self.hedge_data,
            nodes: self.nodes,
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn extend_test() {}
}
