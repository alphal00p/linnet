use std::{borrow::Borrow, collections::BTreeMap, fmt::Display, path::Path, str::FromStr};

use dot_parser::ast::{GraphFromFileError, PestError};
use itertools::{Either, Itertools};

use crate::half_edge::{
    builder::HedgeGraphBuilder,
    involution::{EdgeData, Flow, Orientation},
    nodestore::{NodeStorage, NodeStorageOps},
    GVEdgeAttrs, HedgeGraph, NodeIndex,
};

pub struct GlobalGraph<E, V, G, N: NodeStorage<NodeData = V>> {
    pub global_data: G,
    pub graph: HedgeGraph<E, V, N>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalData {
    pub statements: BTreeMap<String, String>,
}

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
    pub id: Option<String>,
    pub statements: BTreeMap<String, String>,
}

impl DotVertexData {
    pub fn id(&self) -> Option<&str> {
        if let Some(d) = self.statements.get("id") {
            Some(d.as_str())
        } else {
            self.id.as_deref()
        }
    }

    pub fn format(&self, template: impl AsRef<str>) -> String {
        let mut result = template.as_ref().to_owned();

        // Find all occurrences of {key} in the template
        for (key, value) in &self.statements {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Remove any remaining {whatever} patterns
        while let Some(start) = result.find('{') {
            if let Some(end) = result[start..].find('}') {
                result.replace_range(start..=start + end, "");
            } else {
                break;
            }
        }

        result
    }

    pub fn empty() -> Self {
        DotVertexData {
            id: None,
            statements: BTreeMap::new(),
        }
    }

    pub fn extend(&mut self, other: Self) {
        self.statements.extend(other.statements);
    }

    pub fn get<Q: Ord + ?Sized, F: FromStr>(&self, key: &Q) -> Option<Result<F, F::Err>>
    where
        String: Borrow<Q>,
    {
        self.statements.get(key).map(|s| s.parse())
    }

    pub fn add_statement(&mut self, key: impl ToString, value: impl ToString) {
        self.statements.insert(key.to_string(), value.to_string());
    }
}

impl Display for DotVertexData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(id) = self.id() {
            write!(f, "id={},", id)?;
        }
        for (key, value) in &self.statements {
            if key == "id" {
                continue;
            }
            write!(f, "{}={},", key, value)?;
        }
        Ok(())
    }
}

// pub trait DotFormat<T> {
//     fn format(&self, template: T) -> String;
// }

// impl<F> DotFormat<F> for DotVertexData
// where
//     F: Fn(&DotVertexData) -> String,
// {
//     fn format(&self, template: F) -> String {
//         let mut result = template.as_ref().to_owned();

//         // Find all occurrences of {key} in the template
//         for (key, value) in &self.statements {
//             let placeholder = format!("{{{}}}", key);
//             result = result.replace(&placeholder, value);
//         }

//         // Remove any remaining {whatever} patterns
//         while let Some(start) = result.find('{') {
//             if let Some(end) = result[start..].find('}') {
//                 result.replace_range(start..=start + end, "");
//             } else {
//                 break;
//             }
//         }

//         result
//     }
// }

// impl<S: AsRef<str>> DotFormat<S> for DotVertexData {
//     fn format(&self, template: S) -> String {
//         let mut result = template.as_ref().to_owned();

//         // Find all occurrences of {key} in the template
//         for (key, value) in &self.statements {
//             let placeholder = format!("{{{}}}", key);
//             result = result.replace(&placeholder, value);
//         }

//         // Remove any remaining {whatever} patterns
//         while let Some(start) = result.find('{') {
//             if let Some(end) = result[start..].find('}') {
//                 result.replace_range(start..=start + end, "");
//             } else {
//                 break;
//             }
//         }

//         result
//     }
// }

impl DotEdgeData {
    pub fn empty() -> Self {
        DotEdgeData {
            statements: BTreeMap::new(),
        }
    }

    pub fn extend(&mut self, other: Self) {
        self.statements.extend(other.statements);
    }

    pub fn format(&self, template: impl AsRef<str>) -> String {
        let mut result = template.as_ref().to_owned();

        // Find all occurrences of {key} in the template
        for (key, value) in &self.statements {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Remove any remaining {whatever} patterns
        while let Some(start) = result.find('{') {
            if let Some(end) = result[start..].find('}') {
                result.replace_range(start..=start + end, "");
            } else {
                break;
            }
        }

        result
    }

    pub fn add_statement(&mut self, key: impl ToString, value: impl ToString) {
        self.statements.insert(key.to_string(), value.to_string());
    }

    pub fn get<Q: Ord + ?Sized, F: FromStr>(&self, key: &Q) -> Option<Result<F, F::Err>>
    where
        String: Borrow<Q>,
    {
        self.statements.get(key).map(|s| s.parse())
    }

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
                id: Some(value.id),
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

pub struct HedgeGraphSet<E, V, S: NodeStorageOps<NodeData = V>> {
    pub set: Vec<HedgeGraph<E, V, S>>,
}

impl<E: TryFrom<DotEdgeData>, V: TryFrom<DotVertexData>, S: NodeStorageOps<NodeData = V>>
    HedgeGraphSet<E, V, S>
{
    #[allow(clippy::result_large_err)]
    pub fn from_file<'a, P>(p: P) -> Result<Self, HedgeParseError<'a, E::Error, V::Error>>
    where
        P: AsRef<Path>,
    {
        let ast_graphs = dot_parser::ast::Graphs::from_file(p)?;
        let mut set = Vec::with_capacity(ast_graphs.graphs.len());
        for g in ast_graphs.graphs {
            let graph =
                HedgeGraph::<_, _, _>::try_from_life(dot_parser::canonical::Graph::from(g))?;

            set.push(graph);
        }
        Ok(HedgeGraphSet { set })
    }

    #[allow(clippy::result_large_err)]
    pub fn from_string<'a, Str: AsRef<str>>(
        s: Str,
    ) -> Result<Self, HedgeParseError<'a, E::Error, V::Error>> {
        let ast_graphs = dot_parser::ast::Graphs::try_from(s.as_ref())?;

        let mut set = Vec::with_capacity(ast_graphs.graphs.len());
        for g in ast_graphs.graphs {
            let can_graph = dot_parser::canonical::Graph::from(
                g.filter_map(&|a| Some((a.0.to_string(), a.1.to_string()))),
            );
            let graph = HedgeGraph::<_, _, _>::try_from_life(can_graph)?;

            set.push(graph);
        }

        Ok(HedgeGraphSet { set })
    }
}

impl<E: TryFrom<DotEdgeData>, V: TryFrom<DotVertexData>, S: NodeStorageOps<NodeData = V>>
    HedgeGraph<E, V, S>
{
    #[allow(clippy::result_large_err)]
    pub fn from_file<'a, P>(p: P) -> Result<Self, HedgeParseError<'a, E::Error, V::Error>>
    where
        P: AsRef<Path>,
    {
        let ast_graph = dot_parser::ast::Graph::from_file(p)?;
        let can_graph = dot_parser::canonical::Graph::from(ast_graph);

        Self::try_from_life(can_graph)
    }

    #[allow(clippy::result_large_err)]
    pub fn from_string<'a, Str: AsRef<str>>(
        s: Str,
    ) -> Result<Self, HedgeParseError<'a, E::Error, V::Error>> {
        let ast_graph = dot_parser::ast::Graph::try_from(s.as_ref())?;
        let can_graph = dot_parser::canonical::Graph::from(
            ast_graph.filter_map(&|a| Some((a.0.to_string(), a.1.to_string()))),
        );
        Self::try_from_life(can_graph)
    }
}

#[derive(Debug)]
pub enum HedgeParseError<'a, E, V> {
    DotVertexDataError(V),
    GraphFromFile(GraphFromFileError<'a>),
    ParseError(PestError),
    DotEdgeDataError(E),
}

impl<'a, E, V> From<GraphFromFileError<'a>> for HedgeParseError<'a, E, V> {
    fn from(e: GraphFromFileError<'a>) -> Self {
        HedgeParseError::GraphFromFile(e)
    }
}

impl<E, V> From<PestError> for HedgeParseError<'_, E, V> {
    fn from(e: PestError) -> Self {
        HedgeParseError::ParseError(e)
    }
}

#[allow(clippy::result_large_err)]
pub trait HedgeParseExt<'a, D> {
    type Error;
    fn edge<V>(self) -> Result<D, HedgeParseError<'a, Self::Error, V>>;
    fn vertex<E>(self) -> Result<D, HedgeParseError<'a, E, Self::Error>>;
}

impl<'a, Err, D> HedgeParseExt<'a, D> for Result<D, Err> {
    type Error = Err;
    fn edge<V>(self) -> Result<D, HedgeParseError<'a, Self::Error, V>> {
        self.map_err(|e| HedgeParseError::DotEdgeDataError(e))
    }
    fn vertex<E>(self) -> Result<D, HedgeParseError<'a, E, Self::Error>> {
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
            |_, _, v: V| v.into(),
            |_, _, _, e: EdgeData<E>| e.map(|data| data.into()),
        );

        let mut out = String::new();
        mapped
            .dot_serialize_fmt(&mut out, &|d| format!("{d}"), &|d| format!("{d}"))
            .unwrap();

        HedgeGraph::<E, V, S::OpStorage<V>>::from_string(out).unwrap()
    }

    pub fn debug_dot(self) -> String {
        let mapped: HedgeGraph<DotEdgeData, DotVertexData, S::OpStorage<DotVertexData>> = self.map(
            |_, _, v: V| v.into(),
            |_, _, _, e: EdgeData<E>| e.map(|data| data.into()),
        );
        let mut out = String::new();
        mapped
            .dot_serialize_fmt(&mut out, &|d| format!("{d}"), &|d| format!("{d}"))
            .unwrap();
        out
    }

    pub fn format_dot(
        self,
        edge_format: impl AsRef<str>,
        vertex_format: impl AsRef<str>,
    ) -> String {
        let mapped: HedgeGraph<DotEdgeData, DotVertexData, S::OpStorage<DotVertexData>> = self.map(
            |_, _, v: V| v.into(),
            |_, _, _, e: EdgeData<E>| e.map(|data| data.into()),
        );
        mapped.dot_impl(
            &mapped.full_filter(),
            "",
            &|d| Some(format!("{d}label={}", d.format(&edge_format))),
            &|d| Some(format!("{d}label={}", d.format(&vertex_format))),
        )
    }
}

pub trait LifeTimeTryFrom<'a, O>: Sized {
    type Error<'b>
    where
        'b: 'a;

    fn try_from_life(value: O) -> Result<Self, Self::Error<'a>>;
}
impl<'a, E: TryFrom<DotEdgeData>, V: TryFrom<DotVertexData>, S: NodeStorageOps<NodeData = V>>
    LifeTimeTryFrom<'a, dot_parser::canonical::Graph<(String, String)>> for HedgeGraph<E, V, S>
{
    type Error<'b>
        = HedgeParseError<'b, E::Error, V::Error>
    where
        'b: 'a;
    fn try_from_life(
        value: dot_parser::canonical::Graph<(String, String)>,
    ) -> Result<Self, Self::Error<'a>> {
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
    pub fn dot_serialize_io(
        &self,
        writer: &mut impl std::io::Write,
        edge_map: &impl Fn(&E) -> String,
        node_map: &impl Fn(&V) -> String,
    ) -> Result<(), std::io::Error> {
        writeln!(writer, "digraph {{")?;

        for (n, (_, _, v)) in self.iter_nodes().enumerate() {
            writeln!(writer, "  {} [{}];", n, node_map(v))?;
        }

        for (hedge_pair, _, data) in self.iter_all_edges() {
            let attr = GVEdgeAttrs {
                color: None,
                label: None,
                other: Some(edge_map(data.data)),
            };

            write!(writer, "  ")?;
            hedge_pair.dot_io(writer, self, data.orientation, attr)?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }

    pub fn dot_serialize_fmt(
        &self,
        writer: &mut impl std::fmt::Write,
        edge_map: &impl Fn(&E) -> String,
        node_map: &impl Fn(&V) -> String,
    ) -> Result<(), std::fmt::Error> {
        writeln!(writer, "digraph {{")?;

        for (n, (_, _, v)) in self.iter_nodes().enumerate() {
            writeln!(writer, "  {} [{}];", n, node_map(v))?;
        }

        for (hedge_pair, _, data) in self.iter_all_edges() {
            let attr = GVEdgeAttrs {
                color: None,
                label: None,
                other: Some(edge_map(data.data)),
            };

            write!(writer, "  ")?;
            hedge_pair.dot_fmt(writer, self, data.orientation, attr)?;
        }
        writeln!(writer, "}}")?;
        Ok(())
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
                &|b| Some(format!("label={:?}", b.id))
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
        let graph: DotGraph = HedgeGraph::from_string(s).unwrap();

        let mut serialized = String::new();
        graph
            .dot_serialize_fmt(&mut serialized, &|e| format!("{e}"), &|v| format!("{v}"))
            .unwrap();

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

        let mut serialized2 = String::new();
        graph2
            .dot_serialize_fmt(&mut serialized2, &|e| format!("{e}"), &|v| format!("{v}"))
            .unwrap();

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
            graph2.dot_impl(&graph.full_filter(), "", &|_| None, &|b| Some(format!(
                "label={:?}",
                b.id
            )))
        );
        // println!("{}",graph.ed)
        graph2.align_underlying_to_superficial();
        println!(
            "{}",
            graph2.dot_impl(&graph.full_filter(), "", &|_| None, &|b| Some(format!(
                "label={:?}",
                b.id
            )))
        );

        let mut serialized2 = String::new();
        graph2
            .dot_serialize_fmt(&mut serialized2, &|e| format!("{e}"), &|v| format!("{v}"))
            .unwrap();

        println!("{}", serialized2);

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
