use std::path::Path;

use ahash::AHashMap;
use dot_parser::ast::{GraphFromFileError, PestError};
use itertools::Either;

use crate::half_edge::{
    builder::HedgeGraphBuilder,
    involution::{Flow, Orientation},
    nodestorage::NodeStorageOps,
    HedgeGraph, HedgeGraphError, NodeIndex,
};

pub struct DotEdgeData {
    statements: AHashMap<String, String>,
}
pub struct DotVertexData {
    id: String,
    statements: AHashMap<String, String>,
}

impl DotEdgeData {
    pub fn from_parser(
        edge: dot_parser::canonical::Edge<(String, String)>,
        map: &AHashMap<String, NodeIdOrDangling>,
        orientation: impl Into<Orientation>,
    ) -> (Self, Orientation, NodeIndex, Either<NodeIndex, Flow>) {
        let mut orientation = orientation.into();
        let mut statements: AHashMap<_, _> = edge
            .attr
            .into_iter()
            .filter_map(|(key, value)| {
                if &key == "dir" {
                    match value.as_str() {
                        "forward" => orientation = Orientation::Default,
                        "back" => orientation = Orientation::Reversed,
                        "none" => orientation = Orientation::Undirected,
                        _ => panic!("Invalid edge direction"),
                    }
                    None
                } else {
                    Some((key, value))
                }
            })
            .collect();

        let source = map[&edge.from].clone();
        let target = map[&edge.to].clone();

        let (source, target) = match (source, target) {
            (NodeIdOrDangling::Id(source), NodeIdOrDangling::Id(target)) => {
                (source, Either::Left(target))
            }
            (
                NodeIdOrDangling::Id(source),
                NodeIdOrDangling::Dangling {
                    flow,
                    statements: states,
                },
            ) => {
                statements.extend(states);
                (source, Either::Right(flow))
            }
            (
                NodeIdOrDangling::Dangling {
                    flow,
                    statements: states,
                },
                NodeIdOrDangling::Id(source),
            ) => {
                statements.extend(states);
                (source, Either::Right(-flow))
            }
            _ => panic!("Cannot connect an edge to two external nodes"),
        };

        (DotEdgeData { statements }, orientation, source, target)
    }
}

impl DotVertexData {
    pub fn from_parser(
        value: dot_parser::canonical::Node<(String, String)>,
    ) -> Either<Self, (Flow, AHashMap<String, String>)> {
        let mut flow = None;
        let statements = value
            .attr
            .into_iter()
            .filter_map(|(key, value)| {
                if &key == "flow" {
                    match value.as_str() {
                        "source" => flow = Some(Flow::Source),
                        "sink" => flow = Some(Flow::Sink),
                        _ => panic!("Invalid flow"),
                    }
                    None
                } else {
                    Some((key, value))
                }
            })
            .collect();

        if let Some(flow) = flow {
            Either::Right((flow, statements))
        } else {
            Either::Left(DotVertexData {
                id: value.id,
                statements,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeIdOrDangling {
    Id(NodeIndex),
    Dangling {
        flow: Flow,
        statements: AHashMap<String, String>,
    },
}

impl<S: NodeStorageOps<NodeData = DotVertexData>> HedgeGraph<DotEdgeData, DotVertexData, S> {
    #[allow(clippy::result_large_err)]
    pub fn from_file<P>(p: P) -> Result<Self, GraphFromFileError>
    where
        P: AsRef<Path>,
    {
        let ast_graph = dot_parser::ast::Graph::from_file(p)?;
        let can_graph = dot_parser::canonical::Graph::from(ast_graph);
        Ok(can_graph.into())
    }

    #[allow(clippy::result_large_err)]
    pub fn from_string<Str: AsRef<str>>(s: Str) -> Result<Self, PestError> {
        let ast_graph = dot_parser::ast::Graph::try_from(s.as_ref())?;
        let can_graph = dot_parser::canonical::Graph::from(
            ast_graph.filter_map(&|a| Some((a.0.to_string(), a.1.to_string()))),
        );
        Ok(can_graph.into())
    }
}

impl<S: NodeStorageOps<NodeData = DotVertexData>>
    From<dot_parser::canonical::Graph<(String, String)>>
    for HedgeGraph<DotEdgeData, DotVertexData, S>
{
    fn from(value: dot_parser::canonical::Graph<(String, String)>) -> Self {
        let mut g = HedgeGraphBuilder::new();
        let mut map = AHashMap::new();

        for (id, n) in value.nodes.set {
            let idorstatements = match DotVertexData::from_parser(n) {
                Either::Left(d) => NodeIdOrDangling::Id(g.add_node(d)),
                Either::Right((flow, statements)) => {
                    NodeIdOrDangling::Dangling { flow, statements }
                }
            };

            map.insert(id, idorstatements);
        }

        for mut e in value.edges.set {
            e.attr
                .elems
                .extend([("iter".to_string(), "to".to_string())]);
            let (data, orientation, source, target) =
                DotEdgeData::from_parser(e, &map, value.is_digraph);
            match target {
                Either::Left(a) => {
                    g.add_edge(source, a, data, orientation);
                }
                Either::Right(flow) => {
                    g.add_external_edge(source, data, orientation, flow);
                }
            }
        }

        g.build()
    }
}

#[cfg(test)]
pub mod test {
    use crate::half_edge::HedgeGraph;

    #[test]
    fn test_from_string() {
        let s = "digraph G {
            A -> B [label=\"Hello\"];
            B -> C [label=\"World\"];
        }";
        let graph: HedgeGraph<crate::dot_parser::DotEdgeData, crate::dot_parser::DotVertexData> =
            HedgeGraph::from_string(s).unwrap();
        println!(
            "Graph: {}",
            graph.dot_impl(
                &graph.full_filter(),
                "",
                &|a| Some(format!("label={}", a.statements["label"])),
                &|b| Some(format!("label={}", b.id))
            )
        );
        assert_eq!(graph.n_nodes(), 3);
        assert_eq!(graph.n_internals(), 2);
    }
}
