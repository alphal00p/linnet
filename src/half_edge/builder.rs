use super::{
    involution::{EdgeData, EdgeIndex, Flow, Hedge, Involution, Orientation},
    subgraph::HedgeNode,
    HedgeGraph, NodeIndex,
};

#[derive(Clone, Debug)]
pub struct HedgeNodeBuilder<V> {
    data: V,
    pub(crate) hedges: Vec<Hedge>,
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

    pub fn build(self) -> HedgeGraph<E, V> {
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

impl<E, V> From<HedgeGraphBuilder<E, V>> for HedgeGraph<E, V> {
    fn from(builder: HedgeGraphBuilder<E, V>) -> Self {
        let len = builder.involution.len();
        let nodes: Vec<HedgeNode> = builder
            .nodes
            .iter()
            .map(|x| (HedgeNode::from_builder(x, len)))
            .collect();
        let mut hedgedata = vec![None; builder.involution.len()];
        for (v, n) in builder.nodes.iter().enumerate() {
            for h in &n.hedges {
                hedgedata[h.0] = Some(NodeIndex(v));
            }
        }
        let hedge_data = hedgedata.into_iter().map(|x| x.unwrap()).collect();
        let node_data: Vec<V> = builder.nodes.into_iter().map(|x| x.data).collect();
        let mut edge_data = Vec::new();

        let involution = builder.involution.map_full(|h, d| {
            let edgeid = EdgeIndex(edge_data.len());
            edge_data.push((d.data, h));
            EdgeData::new(edgeid, d.orientation)
        });

        HedgeGraph {
            base_nodes: nodes.len(),
            nodes,
            node_data,
            hedge_data,
            edge_data,
            involution,
        }
    }
}
