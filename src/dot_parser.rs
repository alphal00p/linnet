use std::{collections::BTreeMap, fmt::Display, path::Path};

use ahash::AHashMap;
use dot_parser::ast::{GraphFromFileError, PestError};
use itertools::{Either, Itertools};

use crate::half_edge::{
    builder::HedgeGraphBuilder,
    involution::{EdgeData, Flow, Orientation},
    nodestorage::NodeStorageOps,
    GVEdgeAttrs, HedgeGraph, NodeIndex,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotEdgeData {
    pub statements: BTreeMap<String, String>,
}

impl Display for DotEdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.statements {
            write!(f, "{}={},", key, value)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotVertexData {
    pub id: String,
    pub statements: BTreeMap<String, String>,
}

impl DotVertexData {
    pub fn id(&self) -> String {
        if let Some(d) = self.statements.get("id") {
            d.clone()
        } else {
            self.id.clone()
        }
    }
}

impl Display for DotVertexData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "id={},", self.id())?;
        for (key, value) in &self.statements {
            if key == "id" {
                continue;
            }
            write!(f, "{}={},", key, value)?;
        }
        Ok(())
    }
}

impl DotEdgeData {
    pub fn from_parser(
        edge: dot_parser::canonical::Edge<(String, String)>,
        map: &BTreeMap<String, NodeIdOrDangling>,
        orientation: impl Into<Orientation>,
    ) -> (Self, Orientation, NodeIndex, Either<NodeIndex, Flow>) {
        let mut orientation = orientation.into();
        let mut statements: BTreeMap<_, _> = edge
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
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                orientation = orientation.relative_to(-flow);
                (source, Either::Right(-flow))
            }
            (
                NodeIdOrDangling::Dangling {
                    flow,
                    statements: states,
                },
                NodeIdOrDangling::Id(source),
            ) => {
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                orientation = orientation.relative_to(flow);
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
    ) -> Either<Self, (Flow, BTreeMap<String, String>)> {
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
        statements: BTreeMap<String, String>,
    },
}

impl<E: TryFrom<DotEdgeData>, V: TryFrom<DotVertexData>, S: NodeStorageOps<NodeData = V>>
    HedgeGraph<E, V, S>
{
    #[allow(clippy::result_large_err)]
    pub fn from_file<P>(p: P) -> Result<Self, HedgeParseError<E::Error, V::Error>>
    where
        P: AsRef<Path>,
    {
        let ast_graph = dot_parser::ast::Graph::from_file(p)?;
        let can_graph = dot_parser::canonical::Graph::from(ast_graph);
        can_graph.try_into()
    }

    #[allow(clippy::result_large_err)]
    pub fn from_string<Str: AsRef<str>>(
        s: Str,
    ) -> Result<Self, HedgeParseError<E::Error, V::Error>> {
        let ast_graph = dot_parser::ast::Graph::try_from(s.as_ref())?;
        let can_graph = dot_parser::canonical::Graph::from(
            ast_graph.filter_map(&|a| Some((a.0.to_string(), a.1.to_string()))),
        );
        can_graph.try_into()
    }
}

#[derive(Debug)]
pub enum HedgeParseError<E, V> {
    DotVertexDataError(V),
    GraphFromFile(GraphFromFileError),
    ParseError(PestError),
    DotEdgeDataError(E),
}

impl<E, V> From<GraphFromFileError> for HedgeParseError<E, V> {
    fn from(e: GraphFromFileError) -> Self {
        HedgeParseError::GraphFromFile(e)
    }
}

impl<E, V> From<PestError> for HedgeParseError<E, V> {
    fn from(e: PestError) -> Self {
        HedgeParseError::ParseError(e)
    }
}

pub trait HedgeParseExt<D> {
    type Error;
    fn edge<V>(self) -> Result<D, HedgeParseError<Self::Error, V>>;
    fn vertex<E>(self) -> Result<D, HedgeParseError<E, Self::Error>>;
}

impl<Err, D> HedgeParseExt<D> for Result<D, Err> {
    type Error = Err;
    fn edge<V>(self) -> Result<D, HedgeParseError<Self::Error, V>> {
        self.map_err(|e| HedgeParseError::DotEdgeDataError(e))
    }
    fn vertex<E>(self) -> Result<D, HedgeParseError<E, Self::Error>> {
        self.map_err(|e| HedgeParseError::DotVertexDataError(e))
    }
}

impl<
        E: TryFrom<DotEdgeData> + Into<DotEdgeData>,
        V: TryFrom<DotVertexData> + Into<DotVertexData>,
        S: NodeStorageOps<NodeData = V>,
    > HedgeGraph<E, V, S>
where
    E::Error: std::fmt::Debug,
    V::Error: std::fmt::Debug,
{
    pub fn back_and_forth_dot(self) -> HedgeGraph<E, V, S::OpStorage<V>> {
        let mapped: HedgeGraph<DotEdgeData, DotVertexData, S::OpStorage<DotVertexData>> = self.map(
            |_, _, _, v: V| v.into(),
            |_, _, _, e: EdgeData<E>| e.map(|data| data.into()),
        );

        let serialize = mapped.dot_serialize(&|d| format!("{d}"), &|d| format!("{d}"));

        HedgeGraph::<E, V, S::OpStorage<V>>::from_string(serialize).unwrap()
    }
}

impl<E: TryFrom<DotEdgeData>, V: TryFrom<DotVertexData>, S: NodeStorageOps<NodeData = V>>
    TryFrom<dot_parser::canonical::Graph<(String, String)>> for HedgeGraph<E, V, S>
{
    type Error = HedgeParseError<E::Error, V::Error>;
    fn try_from(
        value: dot_parser::canonical::Graph<(String, String)>,
    ) -> Result<Self, Self::Error> {
        let mut g = HedgeGraphBuilder::new();
        let mut map = BTreeMap::new();

        let nodes = BTreeMap::from_iter(value.nodes.set);

        for (id, n) in nodes {
            let idorstatements = match DotVertexData::from_parser(n) {
                Either::Left(d) => NodeIdOrDangling::Id(g.add_node(d.try_into().vertex()?)),
                Either::Right((flow, statements)) => {
                    NodeIdOrDangling::Dangling { flow, statements }
                }
            };

            map.insert(id, idorstatements);
        }

        for mut e in value
            .edges
            .set
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&(&a.from, &a.to), &(&b.from, &b.to)))
        {
            e.attr
                .elems
                .extend([("iter".to_string(), "to".to_string())]);
            let (data, orientation, source, target) =
                DotEdgeData::from_parser(e, &map, value.is_digraph);
            match target {
                Either::Left(a) => {
                    g.add_edge(source, a, data.try_into().edge()?, orientation);
                }
                Either::Right(flow) => {
                    g.add_external_edge(source, data.try_into().edge()?, orientation, flow);
                }
            }
        }

        Ok(g.build())
    }
}

#[macro_export]
macro_rules! dot {
    ($($t:tt)*) => {
        HedgeGraph::from_string(stringify!($($t)*))
    };
}

impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn dot_serialize(
        &self,
        edge_map: &impl Fn(&E) -> String,
        node_map: &impl Fn(&V) -> String,
    ) -> String {
        let mut out = "digraph {\n".to_string();
        // out.push_str(
        //     "  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\"; layout=\"neato\";\n ",
        // );

        for (n, (_, v)) in self.iter_nodes().enumerate() {
            out.push_str(format!("  {} [{}];\n", n, node_map(v)).as_str());
        }

        for (hedge_pair, _, data) in self.iter_all_edges() {
            let attr = GVEdgeAttrs {
                color: None,
                label: None,
                other: Some(edge_map(data.data)),
            };

            out.push_str("  ");
            out.push_str(&hedge_pair.dot(self, data.orientation, attr));
        }

        out += "}";
        out
    }
}

pub type DotGraph = HedgeGraph<crate::dot_parser::DotEdgeData, crate::dot_parser::DotVertexData>;
#[cfg(test)]
pub mod test {
    use crate::{dot_parser::DotGraph, half_edge::HedgeGraph};

    #[test]
    fn test_from_string() {
        let s = "digraph G {
            A      [flow=sink]
            A -> B [label=\"Hello\"];
            B -> C [label=\"World\"dir=none];
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
        assert_eq!(graph.n_nodes(), 2);
        assert_eq!(graph.n_internals(), 1);
    }

    #[test]
    fn test_macro() {
        let _: HedgeGraph<crate::dot_parser::DotEdgeData, crate::dot_parser::DotVertexData> =
            dot!( digraph {
               node [shape=circle,height=0.1,label=""];  overlap="scale"; layout="neato";
             0 -> 7[ dir=none color="red:blue;0.5",label="a"];
            0 -> 12[ dir=forward color="red:blue;0.5",label="d"];
            1 -> 0[ dir=forward color="red:blue;0.5",label="d"];
            1 -> 3[ dir=none color="red:blue;0.5",label="a"];
            2 -> 1[ dir=forward color="red:blue;0.5",label="d"];
            2 -> 6[ dir=none color="red:blue;0.5",label="a"];
            3 -> 13[ dir=forward color="red:blue;0.5",label="d"];
            4 -> 3[ dir=forward color="red:blue;0.5",label="d"];
            4 -> 5[ dir=none color="red:blue;0.5",label="g"];
            5 -> 2[ dir=forward color="red:blue;0.5",label="d"];
            6 -> 7[ dir=forward color="red:blue;0.5",label="e-"];
            7 -> 11[ dir=forward color="red:blue;0.5",label="e-"];
            8 -> 6[ dir=forward color="red:blue;0.5",label="e-"];
            9 -> 4[ dir=forward color="red:blue;0.5",label="d"];
            10 -> 5[ dir=forward color="red:blue;0.5",label="d"];
            })
            .unwrap();
    }

    #[test]
    fn underlying_alignment() {
        let s = "digraph {
          0 [id=B,];
          1 [id=C,];
          ext0 [flow=source];
          ext0 -> 0[dir=forward iter=to,];
          ext1 [flow=sink];
          ext1 -> 0[dir=forward iter=to,];
          ext2 [flow=sink];
          ext2 -> 1[dir=back iter=to,];
          1 -> 0[ dir=back iter=to,];
          ext5 [flow=source];
          ext5 -> 1[dir=forward iter=to,];
        }";
        let mut graph: DotGraph = HedgeGraph::from_string(s).unwrap();

        let serialized = graph.dot_serialize(&|e| format!("{e}"), &|v| format!("{v}"));

        let colored = graph.dot_impl(&graph.full_filter(), "", &|e| Some(format!("{e}")), &|v| {
            Some(format!("{v}"))
        });

        // println!(
        //     "{}",
        //     graph.dot_impl(&graph.full_filter(), "", &|a| None, &|b| Some(format!(
        //         "label={}",
        //         b.id
        //     )))
        // );

        let mut graph2: DotGraph = HedgeGraph::from_string(serialized.clone()).unwrap();

        let serialized2 = graph2.dot_serialize(&|e| format!("{e}"), &|v| format!("{v}"));

        let colored2 =
            graph2.dot_impl(&graph2.full_filter(), "", &|e| Some(format!("{e}")), &|v| {
                Some(format!("{v}"))
            });

        assert_eq!(
            serialized, serialized2,
            "{}//not equal to \n{}",
            serialized, serialized2
        );
        assert_eq!(colored, colored2);

        println!(
            "{}",
            graph2.dot_impl(&graph.full_filter(), "", &|a| None, &|b| Some(format!(
                "label={}",
                b.id
            )))
        );
        // println!("{}",graph.ed)
        graph2.align_underlying_to_superficial();
        println!(
            "{}",
            graph2.dot_impl(&graph.full_filter(), "", &|a| None, &|b| Some(format!(
                "label={}",
                b.id
            )))
        );

        println!(
            "{}",
            graph2.dot_serialize(&|e| format!("{e}"), &|v| format!("{v}"))
        );

        let aligned: DotGraph = dot!(
        digraph {
          node [shape=circle,height=0.1,label=""];  overlap="scale"; layout="neato";
        start=2;
          0 -> 1[ dir=forward];
          ext2 [flow=source];
          ext2 -> 0[dir=forward];
          ext5 -> 1[dir=forward];
          ext4 [flow=sink];
          1 [id=C,];
          ext3 [flow=source];
          ext3 -> 0[dir=forward];
          ext4 -> 1[dir=back];
          0 [id=B,];
          ext5 [flow=source];
        })
        .unwrap();

        graph2 = graph2.back_and_forth_dot();

        assert_eq!(
            aligned,
            graph2,
            "{}\n//not equal to\n{}",
            aligned.dot_display(&aligned.full_filter()),
            graph2.dot_display(&graph2.full_filter())
        );
        // assert_eq!(graph.n_nodes(), 2);
        // assert_eq!(graph.n_internals(), 1);
    }
}
