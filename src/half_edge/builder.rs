use super::{
    hedgevec::SmartHedgeVec,
    involution::{Flow, Hedge, Involution, Orientation},
    nodestore::NodeStorageOps,
    subgraph::BaseSubgraph,
    HedgeGraph, NodeIndex,
};

#[derive(Clone, Debug)]
pub struct HedgeNodeBuilder<V> {
    pub(crate) data: V,
    pub(crate) hedges: Vec<Hedge>,
}

impl<V> HedgeNodeBuilder<V> {
    pub fn to_base<S: BaseSubgraph>(&self, len: usize) -> S {
        let mut subgraph = S::empty(len);

        for hedge in &self.hedges {
            subgraph.add(*hedge);
        }

        subgraph
    }
}

#[derive(Clone, Debug)]
pub struct HedgeGraphBuilder<E, V> {
    nodes: Vec<HedgeNodeBuilder<V>>,
    pub(crate) involution: Involution<E>,
}

impl<E, V> HedgeGraphBuilder<E, V> {
    pub fn new() -> Self {
        HedgeGraphBuilder {
            nodes: Vec::new(),
            involution: Involution::new(),
        }
    }

    pub fn build<N: NodeStorageOps<NodeData = V>>(self) -> HedgeGraph<E, V, N> {
        self.into()
    }

    pub fn add_node(&mut self, data: V) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(HedgeNodeBuilder {
            data,
            hedges: Vec::new(),
        });
        NodeIndex(index)
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        data: E,
        directed: impl Into<Orientation>,
    ) {
        let (sourceh, sinkh) = self.involution.add_pair(data, directed);
        self.nodes[source.0].hedges.push(sourceh);
        self.nodes[sink.0].hedges.push(sinkh);
    }

    pub fn add_external_edge(
        &mut self,
        source: NodeIndex,
        data: E,
        orientation: impl Into<Orientation>,
        underlying: Flow,
    ) {
        let id = self.involution.add_identity(data, orientation, underlying);
        self.nodes[source.0].hedges.push(id);
    }
}

impl<E, V> Default for HedgeGraphBuilder<E, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, V, N: NodeStorageOps<NodeData = V>> From<HedgeGraphBuilder<E, V>> for HedgeGraph<E, V, N> {
    fn from(builder: HedgeGraphBuilder<E, V>) -> Self {
        let len = builder.involution.len();

        HedgeGraph {
            node_store: N::build(builder.nodes, len),
            edge_store: SmartHedgeVec::new(builder.involution),
        }
    }
}
