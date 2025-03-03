use crate::{
    half_edge::{
        nodestorage::{NodeStorage, NodeStorageOps, NodeStorageVec},
        subgraph::HedgeNode,
        NodeIndex,
    },
    tree::{parent_pointer::PPNode, Forest, RootData, RootId},
};

use super::{SetIndex, UnionFind};

pub struct HedgeNodeStore<V> {
    data: V,
    node: HedgeNode,
}
pub struct UnionFindNodeStore<V> {
    nodes: UnionFind<HedgeNodeStore<V>>,
}

impl<V> From<NodeStorageVec<V>> for UnionFindNodeStore<V> {
    fn from(vecnodes: NodeStorageVec<V>) -> Self {
        let node_data: Vec<_> = vecnodes
            .node_data
            .into_iter()
            .zip(vecnodes.nodes)
            .map(|(v, h)| HedgeNodeStore { data: v, node: h })
            .collect();

        let hairs: Vec<_> = node_data.iter().map(|a| a.node.hairs.clone()).collect();

        UnionFindNodeStore {
            nodes: UnionFind::from_bitvec_partition(node_data.into_iter().zip(hairs).collect())
                .unwrap(),
        }
    }
}

impl SetIndex {
    pub fn node_ref(&self) -> &NodeIndex {
        unsafe { std::mem::transmute(self) }
    }
}

impl<V> NodeStorage for UnionFindNodeStore<V> {
    type Storage<N> = UnionFindNodeStore<N>;
    type NodeData = V;
}

impl<V> NodeStorageOps for UnionFindNodeStore<V> {
    fn hedge_len(&self) -> usize {
        self.nodes.n_elements()
    }

    fn to_forest<U>(
        &self,
        map_data: impl Fn(&Self::NodeData) -> U,
    ) -> crate::tree::Forest<U, crate::tree::parent_pointer::ParentPointerStore<()>> {
        Forest {
            roots: self
                .nodes
                .set_data
                .iter()
                .map(|a| RootData {
                    root_id: a.root_pointer.into(),
                    data: map_data(&a.data.as_ref().unwrap().data),
                })
                .collect(),
            nodes: self
                .nodes
                .nodes
                .iter()
                .map(|a| {
                    let ad = a.get();
                    let n = match &ad {
                        super::UFNode::Child(c) => PPNode::child((), (*c).into()),
                        super::UFNode::Root { set_data_idx, .. } => {
                            PPNode::root((), RootId(set_data_idx.0))
                        }
                    };

                    a.set(ad);

                    n
                })
                .collect(),
        }
    }

    fn node_len(&self) -> usize {
        self.nodes.n_sets()
    }

    fn set_hedge_data(&mut self, hedge: crate::half_edge::involution::Hedge, nodeid: NodeIndex) {
        panic!("should not need to set")
    }

    fn check_and_set_nodes(&mut self) -> Result<(), crate::half_edge::HedgeGraphError> {
        Ok(())
    }

    fn map_data_ref_graph<'a, E, V2>(
        &'a self,
        graph: &'a crate::half_edge::HedgeGraph<E, Self::NodeData, Self>,
        node_map: &impl Fn(
            &'a crate::half_edge::HedgeGraph<E, Self::NodeData, Self>,
            &'a Self::NodeData,
            &'a HedgeNode,
        ) -> V2,
    ) -> Self::Storage<V2> {
        UnionFindNodeStore {
            nodes: self.nodes.map_set_data_ref(|n| HedgeNodeStore {
                data: node_map(graph, &n.data, &n.node),
                node: n.node.clone(),
            }),
        }
    }

    fn get_node(&self, node_id: NodeIndex) -> &HedgeNode {
        &self.nodes[SetIndex(node_id.0)].node
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &Self::NodeData)> {
        self.nodes.iter_set_data().map(|(_, d)| (&d.node, &d.data))
    }

    fn node_id_ref(&self, hedge: crate::half_edge::involution::Hedge) -> NodeIndex {
        NodeIndex(self.nodes.find_data_index(hedge).0)
    }

    fn get_node_data_mut(&mut self, node_id: NodeIndex) -> &mut Self::NodeData {
        &mut self.nodes[SetIndex(node_id.0)].data
    }

    fn iter_node_id(&self) -> impl Iterator<Item = NodeIndex> {
        self.nodes.iter_set_data().map(|(i, _)| NodeIndex(i.0))
    }

    fn get_node_data(&self, node_id: NodeIndex) -> &Self::NodeData {
        &self.nodes[SetIndex(node_id.0)].data
    }

    fn map_data_graph<'a, V2>(
        self,
        involution: &crate::half_edge::involution::Involution<
            crate::half_edge::involution::EdgeIndex,
        >,
        mut f: impl FnMut(
            &crate::half_edge::involution::Involution<crate::half_edge::involution::EdgeIndex>,
            &HedgeNode,
            NodeIndex,
            Self::NodeData,
        ) -> V2,
    ) -> Self::Storage<V2> {
        UnionFindNodeStore {
            nodes: self.nodes.map_set_data(|i, n| HedgeNodeStore {
                data: f(involution, &n.node, NodeIndex(i.0), n.data),
                node: n.node.clone(),
            }),
        }
    }

    fn extend(mut self, other: Self) -> Self {
        self.nodes.extend(other.nodes);
        self
    }

    fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Self::NodeData)> {
        self.nodes
            .iter_set_data()
            .map(|(i, d)| (NodeIndex(i.0), &d.data))
    }

    fn build<
        I: IntoIterator<Item = crate::half_edge::builder::HedgeNodeBuilder<Self::NodeData>>,
    >(
        nodes: I,
        n_hedges: usize,
    ) -> Self {
        NodeStorageVec::build(nodes, n_hedges).into()
    }

    fn drain(self) -> impl Iterator<Item = (NodeIndex, Self::NodeData)> {
        self.nodes
            .drain_set_data()
            .map(|(s, d)| (NodeIndex(s.0), d.data))
    }

    fn random(sources: &[HedgeNode], sinks: &[HedgeNode]) -> Self
    where
        Self::NodeData: Default,
    {
        NodeStorageVec::random(sources, sinks).into()
    }

    fn add_dangling_edge(
        mut self,
        source: NodeIndex,
    ) -> Result<Self, crate::half_edge::HedgeGraphError> {
        let setid = SetIndex(source.0);
        let p = self.nodes.add_child(setid);
        self.nodes.iter_set_data_mut().for_each(|(s, n)| {
            if s == setid {
                n.node.hairs.push(true);
            } else {
                n.node.hairs.push(false);
            }
            n.node.internal_graph.filter.push(false);
        });
        Ok(self)
    }
}
