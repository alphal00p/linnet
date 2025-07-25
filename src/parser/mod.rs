//! # DOT Language Parser and Serializer
//!
//! This module provides the functionality to parse graph descriptions written in the
//! [DOT language](https://graphviz.org/doc/info/lang.html) and convert them into
//! `HedgeGraph` instances. It also handles the reverse process: serializing
//! `HedgeGraph` instances back into the DOT language format, suitable for use
//! with Graphviz tools or for storage.
//!
//! ## Key Features:
//!
//! - **Parsing DOT:**
//!   - Parses DOT files or strings into `HedgeGraph<E, V, S>` instances.
//!   - Uses the external `dot-parser` crate for initial AST parsing, then converts
//!     this AST into the `HedgeGraph` structure.
//!   - Handles DOT node and edge attributes, storing them in `DotVertexData` and
//!     `DotEdgeData` respectively before they are (optionally) converted into
//!     the generic `V` and `E` types of the `HedgeGraph`.
//!   - Supports parsing of "flow" attributes for nodes to indicate sources/sinks,
//!     and "dir" attributes for edges to set orientation.
//! - **Serialization to DOT:**
//!   - Converts `HedgeGraph` instances into DOT language strings.
//!   - Allows custom mapping of generic node and edge data (`V`, `E`) to DOT attributes
//!     via provided closure functions.
//!
//! ## Core Components:
//!
//! - **`DotVertexData`**: A struct that holds attributes (key-value pairs) parsed from
//!   a DOT node definition. This includes standard DOT attributes like `label`, `shape`,
//!   `color`, as well as custom attributes. It can also capture an explicit `id` and
//!   a `flow` (source/sink) for external port representation.
//! - **`DotEdgeData`**: Similar to `DotVertexData`, this struct stores attributes for
//!   edges parsed from DOT, such as `label`, `color`, `dir` (direction), etc.
//! - **`HedgeGraph::from_file(path)` and `HedgeGraph::from_string(dot_string)`**:
//!   These are the primary functions for parsing DOT files and strings into a
//!   `HedgeGraph`. They require that the target `E` (edge data) and `V` (vertex data)
//!   types implement `TryFrom<DotEdgeData>` and `TryFrom<DotVertexData>` respectively.
//! - **`HedgeGraphSet::from_file(path)` and `HedgeGraphSet::from_string(dot_string)`**:
//!   Similar to the above, but can parse files or strings containing multiple DOT graphs.
//! - **`HedgeGraph::dot_serialize_io(writer, edge_map, node_map)` and `HedgeGraph::dot_serialize_fmt(formatter, edge_map, node_map)`**:
//!   Methods on `HedgeGraph` used to serialize the graph to an `io::Write` or `fmt::Write`
//!   target. The `edge_map` and `node_map` closures define how to convert the graph's
//!   edge and node data into DOT attribute strings.
//! - **`dot!(...)` macro**: A utility macro for conveniently creating a `HedgeGraph`
//!   from an inline DOT string literal, typically used in tests or examples.
//!   The graph created will have `DotEdgeData` and `DotVertexData` as its edge and node
//!   data types.
//! - **`HedgeParseError`**: Enum representing potential errors during parsing.
//!
//! ## Usage Example (Conceptual):
//!
//! ```rust,ignore
//! use linnet::half_edge::HedgeGraph;
//! use linnet::dot_parser::{DotEdgeData, DotVertexData};
//!
//! // Define how your custom V/E types are created from DOT attributes
//! impl TryFrom<DotVertexData> for MyVertexData { /* ... */ }
//! impl TryFrom<DotEdgeData> for MyEdgeData { /* ... */ }
//!
//! // Parsing a DOT string
//! let dot_string = "digraph G { a -> b [label=\"edge1\"]; }";
//! let graph: Result<HedgeGraph<MyEdgeData, MyVertexData, _>, _> = HedgeGraph::from_string(dot_string);
//!
//! // Serializing a graph to DOT
//! if let Ok(g) = graph {
//!     let mut output = String::new();
//!     g.dot_serialize_fmt(
//!         &mut output,
//!         &|edge_data| format!("label=\"{}\"", edge_data.custom_label),
//!         &|vertex_data| format!("label=\"Node {}\"", vertex_data.id)
//!     ).unwrap();
//!     println!("{}", output);
//! }
//! ```
//!
//! This module acts as a bridge between the `linnet` graph structures and the widely-used
//! DOT language, facilitating interoperability and visualization.

use std::{
    collections::BTreeMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
    path::Path,
};

use ahash::{HashSet, HashSetExt};
use itertools::{Either, Itertools};

use crate::{
    half_edge::{
        builder::HedgeGraphBuilder,
        involution::{EdgeIndex, Flow, Hedge},
        nodestore::{NodeStorage, NodeStorageOps, NodeStorageVec},
        swap::Swap,
        GVEdgeAttrs, HedgeGraph, NodeIndex,
    },
    permutation::Permutation,
};

pub mod set;
pub use set::GraphSet;

pub mod global;
pub use global::GlobalData;

pub mod vertex;
pub use vertex::DotVertexData;

pub mod edge;
pub use edge::DotEdgeData;

pub mod hedge;
pub use hedge::DotHedgeData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotGraph<N: NodeStorage<NodeData = DotVertexData> = NodeStorageVec<DotVertexData>> {
    pub global_data: GlobalData,
    pub graph: HedgeGraph<DotEdgeData, DotVertexData, DotHedgeData, N>,
}

impl<N: NodeStorage<NodeData = DotVertexData>> Deref for DotGraph<N> {
    type Target = HedgeGraph<DotEdgeData, DotVertexData, DotHedgeData, N>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<N: NodeStorage<NodeData = DotVertexData>> DerefMut for DotGraph<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
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

impl<S: NodeStorageOps<NodeData = DotVertexData>> DotGraph<S> {
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    pub fn from_file<'a, P>(p: P) -> Result<Self, HedgeParseError<'a, (), (), (), ()>>
    where
        P: AsRef<Path>,
    {
        let ast_graph = dot_parser::ast::Graph::from_file(p)?;
        let can_graph = dot_parser::canonical::Graph::from(ast_graph);

        Ok(Self::from(can_graph))
    }

    #[allow(clippy::result_large_err, clippy::type_complexity)]
    pub fn from_string<'a, Str: AsRef<str>>(
        s: Str,
    ) -> Result<Self, HedgeParseError<'a, (), (), (), ()>> {
        let ast_graph = dot_parser::ast::Graph::try_from(s.as_ref())?;
        let can_graph = dot_parser::canonical::Graph::from(
            ast_graph.filter_map(&|a| Some((a.0.to_string(), a.1.to_string()))),
        );
        Ok(Self::from(can_graph))
    }

    pub fn back_and_forth_dot(self) -> Self {
        let mut out = String::new();
        self.graph
            .dot_serialize_fmt(
                &mut out,
                &DotHedgeData::dot_serialize,
                &DotEdgeData::to_string,
                &DotVertexData::to_string,
            )
            .unwrap();

        Self::from_string(out).unwrap()
    }

    pub fn debug_dot(&self) -> String {
        let mut out = String::new();
        self.graph
            .dot_serialize_fmt(
                &mut out,
                &DotHedgeData::dot_serialize,
                &|d| format!("{d}"),
                &|d| format!("{d}"),
            )
            .unwrap();
        out
    }

    pub fn format_dot(
        self,
        edge_format: impl AsRef<str>,
        vertex_format: impl AsRef<str>,
    ) -> String {
        self.graph.dot_impl(
            &self.graph.full_filter(),
            "",
            &|d| Some(format!("{d}label={}", d.format(&edge_format))),
            &|d| Some(format!("{d}label={}", d.format(&vertex_format))),
        )
    }
}

pub mod error;
pub use error::{HedgeParseError, HedgeParseExt};

impl<S: NodeStorageOps<NodeData = DotVertexData>>
    From<dot_parser::canonical::Graph<(String, String)>> for DotGraph<S>
{
    fn from(value: dot_parser::canonical::Graph<(String, String)>) -> Self {
        let global_data = GlobalData::try_from(&value.attr).unwrap();
        let mut g = HedgeGraphBuilder::new();
        let mut map = BTreeMap::new();

        let nodes = BTreeMap::from_iter(value.nodes.set);

        for (id, n) in nodes {
            let idorstatements = match DotVertexData::from_parser(n, &global_data) {
                Either::Left(d) => NodeIdOrDangling::Id(g.add_node(d)),
                Either::Right((flow, statements)) => {
                    NodeIdOrDangling::Dangling { flow, statements }
                }
            };

            map.insert(id, idorstatements);
        }

        for e in value
            .edges
            .set
            .into_iter()
            .sorted_by(|a, b| Ord::cmp(&(&a.from, &a.to), &(&b.from, &b.to)))
        {
            let (data, orientation, source, target) =
                DotEdgeData::from_parser(e, &map, value.is_digraph, &global_data);
            match target {
                Either::Left(a) => {
                    g.add_edge(source, a, data, orientation);
                }
                Either::Right(flow) => {
                    g.add_external_edge(source, data, orientation, flow);
                }
            }
        }

        let mut g: HedgeGraph<DotEdgeData, DotVertexData, DotHedgeData, S> = g.build();

        // println!("{}", g.debug_dot());

        let mut used_edges = HashSet::new();
        let n_edges = g.n_edges();
        let mut edge_map = g.new_edgevec(|d, e, _| {
            d.edge_id.inspect(|d| {
                assert!(used_edges.insert(*d), "Duplicate edge ID: {d} for edge {e}");
                assert!(d.0 < n_edges, "Edge {d} out of bounds (len={n_edges})")
            })
        });

        let mut used_hedges = HashSet::new();

        let n_hedges = g.n_hedges();
        let mut hedge_map = g.new_hedgevec(|h, d| {
            d.id.inspect(|d| {
                assert!(
                    used_hedges.insert(*d),
                    "Duplicate hedge ID: {d} for hedge {h}",
                );
                assert!(d.0 < n_hedges, "Hedge {d} out of bounds (len={n_hedges})")
            })
        });

        let mut used_nodes = HashSet::new();
        let n_nodes = g.n_nodes();
        let mut node_map = g.new_nodevec(|ni, _, v| {
            v.index.inspect(|i| {
                assert!(
                    used_nodes.insert(*i),
                    "Duplicate node index: {i} for node {ni}"
                );
                assert!(i.0 < n_nodes, "Node {i} out of bounds (len ={n_nodes})")
            })
        });

        // println!(
        //     "Hedge Map: {}",
        //     hedge_map.display_string(|i| format!("{i:?}"))
        // );
        // println!(
        //     "Node Map: {}",
        //     node_map.display_string(|i| format!("{i:?}"))
        // );
        // println!(
        //     "Edge Map: {}",
        //     edge_map.display_string(|i| format!("{i:?}"))
        // );

        node_map.fill_in(|id| used_nodes.contains(id));
        edge_map.fill_in(|id| used_edges.contains(id));

        hedge_map.fill_in(|id| used_hedges.contains(id));
        // println!(
        //     "Filled Hedge Map: {}",
        //     hedge_map.display_string(|i| format!("{i:?}"))
        // );

        // println!(
        //     "Filled Node Map: {}",
        //     node_map.display_string(|i| format!("{i:?}"))
        // );
        // println!(
        //     "Filled Edge Map: {}",
        //     edge_map.display_string(|i| format!("{i:?}"))
        // );
        let edge_perm: Permutation = edge_map.try_into().unwrap();
        let node_perm: Permutation = node_map.try_into().unwrap();

        let hedge_perm: Permutation = hedge_map.try_into().unwrap();

        <HedgeGraph<_, _, _, _> as Swap<Hedge>>::permute(&mut g, &hedge_perm);

        // println!("Hedge Perm: {hedge_perm}");
        // println!("Edge Perm: {edge_perm}");
        // println!("Node Perm: {node_perm}");
        <HedgeGraph<_, _, _, _> as Swap<EdgeIndex>>::permute(&mut g, &edge_perm);
        <HedgeGraph<_, _, _, _> as Swap<NodeIndex>>::permute(&mut g, &node_perm);
        DotGraph {
            global_data,
            graph: g,
        }
    }
}

#[macro_export]
macro_rules! dot {
    ($($t:tt)*) => {
        $crate::parser::DotGraph::from_string(stringify!($($t)*))
    };
}

impl<E, V, H, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, H, N> {
    pub fn dot_serialize_io(
        &self,
        writer: &mut impl std::io::Write,
        hedge_map: &impl Fn(&H) -> Option<String>,
        edge_map: &impl Fn(&E) -> String,
        node_map: &impl Fn(&V) -> String,
    ) -> Result<(), std::io::Error> {
        writeln!(writer, "digraph {{")?;

        for (n, (_, _, v)) in self.iter_nodes().enumerate() {
            writeln!(writer, "  {} [{}];", n, node_map(v))?;
        }

        for (hedge_pair, _, data) in self.iter_edges() {
            let attr = GVEdgeAttrs {
                color: None,
                label: None,
                other: Some(edge_map(data.data)),
            };

            write!(writer, "  ")?;
            hedge_pair
                .add_data(self)
                .dot_io(writer, self, hedge_map, data.orientation, attr)?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }

    pub fn dot_serialize_fmt(
        &self,
        writer: &mut impl std::fmt::Write,
        hedge_map: &impl Fn(&H) -> Option<String>,
        edge_map: &impl Fn(&E) -> String,
        node_map: &impl Fn(&V) -> String,
    ) -> Result<(), std::fmt::Error> {
        writeln!(writer, "digraph {{")?;

        for (n, (_, _, v)) in self.iter_nodes().enumerate() {
            writeln!(writer, "  {} [{}];", n, node_map(v))?;
        }

        for (hedge_pair, _, data) in self.iter_edges() {
            let attr = GVEdgeAttrs {
                color: None,
                label: None,
                other: Some(edge_map(data.data)),
            };

            write!(writer, "  ")?;
            hedge_pair
                .add_data(self)
                .dot_fmt(writer, self, hedge_map, data.orientation, attr)?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use crate::parser::{DotEdgeData, DotGraph, DotHedgeData, DotVertexData};

    #[test]
    fn test_from_string() {
        let s = "digraph G {
            A      [flow=sink]
            A -> B [label=\"Hello\" sink=\"AAA\"];
            B -> C [label=\"World\"dir=none];
        }";
        let graph: DotGraph = DotGraph::from_string(s).unwrap();
        // println!("{graph:?}");
        println!(
            "Graph: {}",
            graph.dot_impl(
                &graph.full_filter(),
                "",
                &|a| Some(format!("label={}", a.statements["label"])),
                &|b| Some(format!("label={:?}", b.id))
            )
        );
        let mut out = String::new();
        graph
            .dot_serialize_fmt(
                &mut out,
                &DotHedgeData::dot_serialize,
                &DotEdgeData::to_string,
                &DotVertexData::to_string,
            )
            .unwrap();

        println!("{out}");
        assert_eq!(graph.n_nodes(), 2);
        assert_eq!(graph.n_internals(), 1);
        let g = graph.back_and_forth_dot();
        let gg = g.clone().back_and_forth_dot();
        // println!("{g:?}");
        assert_eq!(g, gg);
    }

    #[test]
    fn test_macro() {
        let _: DotGraph = dot!( digraph {
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
        let graph: DotGraph = DotGraph::from_string(s).unwrap();

        let mut serialized = String::new();
        graph
            .dot_serialize_fmt(
                &mut serialized,
                &DotHedgeData::dot_serialize,
                &|e| format!("{e}"),
                &|v| format!("{v}"),
            )
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

        let mut graph2: DotGraph = DotGraph::from_string(serialized.clone()).unwrap();

        let mut serialized2 = String::new();
        graph2
            .dot_serialize_fmt(
                &mut serialized2,
                &DotHedgeData::dot_serialize,
                &|e| format!("{e}"),
                &|v| format!("{v}"),
            )
            .unwrap();

        let colored2 =
            graph2.dot_impl(&graph2.full_filter(), "", &|e| Some(format!("{e}")), &|v| {
                Some(format!("{v}"))
            });

        assert_eq!(
            serialized, serialized2,
            "{serialized}//not equal to \n{serialized2}",
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
            .dot_serialize_fmt(
                &mut serialized2,
                &DotHedgeData::dot_serialize,
                &|e| format!("{e}"),
                &|v| format!("{v}"),
            )
            .unwrap();

        println!("{serialized2}");

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
