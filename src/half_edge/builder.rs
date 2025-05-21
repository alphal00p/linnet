use super::{
    hedgevec::SmartHedgeVec,
    involution::{Flow, Hedge, Involution, Orientation},
    nodestore::NodeStorageOps,
    subgraph::BaseSubgraph,
    HedgeGraph, NodeIndex,
};

#[derive(Clone, Debug)]
/// A temporary structure used during the construction of a [`HedgeGraph`].
///
/// It holds the data for a node and a list of half-edges that are incident to it
/// before the full graph topology (e.g., specific `NodeStorage` format) is finalized.
pub struct HedgeNodeBuilder<V> {
    /// The data associated with the node being built.
    pub(crate) data: V,
    /// A list of [`Hedge`] identifiers that are incident to this node.
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
/// A builder for programmatically constructing [`HedgeGraph`] instances.
///
/// This builder allows for the incremental addition of nodes and edges (both
/// paired and external/dangling) before finalizing the graph structure.
///
/// # Type Parameters
///
/// - `E`: The type of data to be associated with edges.
/// - `V`: The type of data to be associated with nodes.
///
/// # Example
///
/// ```rust,ignore
/// use linnet::half_edge::HedgeGraphBuilder;
/// use linnet::half_edge::involution::{Flow, Orientation};
///
/// let mut builder = HedgeGraphBuilder::<&str, &str>::new();
///
/// let node0 = builder.add_node("Node_0_Data");
/// let node1 = builder.add_node("Node_1_Data");
/// let node2 = builder.add_node("Node_2_Data");
///
/// builder.add_edge(node0, node1, "Edge_01_Data", Orientation::Default);
/// builder.add_external_edge(node2, "External_Edge_Data", Orientation::Undirected, Flow::Source);
///
/// // Assuming a NodeStorage type MyNodeStore is defined and implements NodeStorageOps
/// // let graph: HedgeGraph<&str, &str, MyNodeStore> = builder.build();
/// ```
pub struct HedgeGraphBuilder<E, V> {
    /// A list of nodes currently being built, stored as [`HedgeNodeBuilder`] instances.
    nodes: Vec<HedgeNodeBuilder<V>>,
    /// The [`Involution`] structure managing the half-edges being added to the graph.
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
