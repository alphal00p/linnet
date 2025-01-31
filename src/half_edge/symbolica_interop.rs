use ahash::AHashMap;

use super::{
    builder::HedgeGraphBuilder,
    involution::{HedgePair, Orientation},
    HedgeGraph, HedgeGraphError, NodeIndex,
};

impl<N: Clone, E: Clone> From<symbolica::graph::Graph<N, E>> for HedgeGraph<E, N> {
    fn from(graph: symbolica::graph::Graph<N, E>) -> Self {
        let mut builder = HedgeGraphBuilder::new();
        let mut map = AHashMap::new();

        for (i, node) in graph.nodes().iter().enumerate() {
            map.insert(i, builder.add_node(node.data.clone()));
        }

        for edge in graph.edges() {
            let vertices = edge.vertices;
            let source = map[&vertices.0];
            let sink = map[&vertices.1];
            builder.add_edge(source, sink, edge.data.clone(), edge.directed);
        }

        builder.into()
    }
}

impl<N: Clone, E: Clone> TryFrom<HedgeGraph<E, N>> for symbolica::graph::Graph<N, E> {
    type Error = HedgeGraphError;

    fn try_from(value: HedgeGraph<E, N>) -> Result<Self, Self::Error> {
        let mut graph = symbolica::graph::Graph::new();
        let mut map = AHashMap::new();

        for (i, node) in value.node_data.iter().enumerate() {
            map.insert(NodeIndex(i), graph.add_node(node.clone()));
        }

        for (i, _, d) in value.iter_all_edges() {
            if let HedgePair::Paired { source, sink } = i {
                let source = map[&value.node_id(source)];
                let sink = map[&value.node_id(sink)];

                let data = d.data.clone();
                let orientation = d.orientation;

                match orientation {
                    Orientation::Default => {
                        graph
                            .add_edge(source, sink, true, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                    Orientation::Reversed => {
                        graph
                            .add_edge(sink, source, true, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                    Orientation::Undirected => {
                        graph
                            .add_edge(source, sink, false, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                }
            } else {
                return Err(HedgeGraphError::HasIdentityHedge);
            }
        }

        Ok(graph)
    }
}
