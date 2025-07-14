// use bitvec::vec::BitVec;
// use iai_callgrind::{
//     library_benchmark, library_benchmark_group, main, FlamegraphConfig, LibraryBenchmarkConfig,
// };
// use linnet::{
//     half_edge::{builder::HedgeGraphBuilder, HedgeGraph, NodeIndex},
//     union_find::union_find_node::UnionFindNodeStore,
// };
// use std::hint::black_box;
// use symbolica::graph::Graph;

// fn k33() -> (HedgeGraph<(), ()>, [NodeIndex; 2]) {
//     let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
//     let a = builder.add_node(());
//     let b = builder.add_node(());
//     let c = builder.add_node(());
//     let d = builder.add_node(());
//     let e = builder.add_node(());
//     let f = builder.add_node(());

//     builder.add_edge(a, d, (), false);
//     builder.add_edge(a, e, (), false);
//     builder.add_edge(a, f, (), false);

//     builder.add_edge(b, d, (), false);
//     builder.add_edge(b, e, (), false);
//     builder.add_edge(b, f, (), false);

//     builder.add_edge(c, d, (), false);
//     builder.add_edge(c, e, (), false);
//     builder.add_edge(c, f, (), false);
//     (builder.build(), [a, d])
// }

// fn k33_for_tree() -> (HedgeGraph<(), ()>, BitVec, NodeIndex) {
//     let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
//     let a = builder.add_node(());
//     let b = builder.add_node(());
//     let c = builder.add_node(());
//     let d = builder.add_node(());
//     let e = builder.add_node(());
//     let f = builder.add_node(());

//     builder.add_edge(a, d, (), false);
//     builder.add_edge(a, e, (), false);
//     builder.add_edge(a, f, (), false);

//     builder.add_edge(b, d, (), false);
//     builder.add_edge(b, e, (), false);
//     builder.add_edge(b, f, (), false);

//     builder.add_edge(c, d, (), false);
//     builder.add_edge(c, e, (), false);
//     builder.add_edge(c, f, (), false);
//     let g = builder.build();
//     let full = g.full_filter();
//     (g, full, a)
// }

// fn k33_for_tree_uf() -> (
//     HedgeGraph<(), (), UnionFindNodeStore<()>>,
//     BitVec,
//     NodeIndex,
// ) {
//     let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
//     let a = builder.add_node(());
//     let b = builder.add_node(());
//     let c = builder.add_node(());
//     let d = builder.add_node(());
//     let e = builder.add_node(());
//     let f = builder.add_node(());

//     builder.add_edge(a, d, (), false);
//     builder.add_edge(a, e, (), false);
//     builder.add_edge(a, f, (), false);

//     builder.add_edge(b, d, (), false);
//     builder.add_edge(b, e, (), false);
//     builder.add_edge(b, f, (), false);

//     builder.add_edge(c, d, (), false);
//     builder.add_edge(c, e, (), false);
//     builder.add_edge(c, f, (), false);
//     let g = builder.build();
//     let full = g.full_filter();
//     (g, full, a)
// }

// fn k33_for_tree_sym() -> (Graph<(), ()>, usize) {
//     let mut k33 = Graph::new();
//     let a = k33.add_node(());
//     let b = k33.add_node(());
//     let c = k33.add_node(());
//     let d = k33.add_node(());
//     let e = k33.add_node(());
//     let f = k33.add_node(());

//     k33.add_edge(a, d, false, ()).unwrap();
//     k33.add_edge(a, e, false, ()).unwrap();
//     k33.add_edge(a, f, false, ()).unwrap();

//     k33.add_edge(b, d, false, ()).unwrap();
//     k33.add_edge(b, e, false, ()).unwrap();
//     k33.add_edge(b, f, false, ()).unwrap();

//     k33.add_edge(c, d, false, ()).unwrap();
//     k33.add_edge(c, e, false, ()).unwrap();
//     k33.add_edge(c, f, false, ()).unwrap();
//     (k33, a)
// }

// #[library_benchmark]
// #[bench::k33st(setup=k33)]
// fn bench_stcut(
//     graph_st: (HedgeGraph<(), ()>, [NodeIndex; 2]),
// ) -> Vec<(BitVec, linnet::half_edge::subgraph::OrientedCut, BitVec)> {
//     black_box(
//         graph_st
//             .0
//             .all_cuts_from_ids(&[graph_st.1[0]], &[graph_st.1[1]]),
//     )
// }

// #[library_benchmark]
// #[bench::k33(setup=k33_for_tree)]
// fn bench_spanning(
//     graph: (HedgeGraph<(), ()>, BitVec, NodeIndex),
// ) -> Result<linnet::half_edge::tree::SimpleTraversalTree, linnet::half_edge::HedgeGraphError> {
//     black_box(
//         linnet::half_edge::tree::SimpleTraversalTree::depth_first_traverse(
//             &graph.0, &graph.1, &graph.2, None,
//         ),
//     )
// }

// #[library_benchmark]
// #[bench::k33(setup=k33_for_tree_uf)]
// fn bench_spanning_uf(
//     graph: (
//         HedgeGraph<(), (), UnionFindNodeStore<()>>,
//         BitVec,
//         NodeIndex,
//     ),
// ) -> Result<linnet::half_edge::tree::SimpleTraversalTree, linnet::half_edge::HedgeGraphError> {
//     black_box(
//         linnet::half_edge::tree::SimpleTraversalTree::depth_first_traverse(
//             &graph.0, &graph.1, &graph.2, None,
//         ),
//     )
// }

// #[library_benchmark]
// #[bench::k33(setup=k33_for_tree_uf)]
// fn bench_spanning_uf_bfs(
//     graph: (
//         HedgeGraph<(), (), UnionFindNodeStore<()>>,
//         BitVec,
//         NodeIndex,
//     ),
// ) -> Result<linnet::half_edge::tree::SimpleTraversalTree, linnet::half_edge::HedgeGraphError> {
//     black_box(
//         linnet::half_edge::tree::SimpleTraversalTree::breadth_first_traverse(
//             &graph.0, &graph.1, &graph.2, None,
//         ),
//     )
// }

// #[library_benchmark]
// #[bench::k33(setup=k33_for_tree_sym)]
// fn bench_spanning_sym(graph: (Graph<(), ()>, usize)) -> symbolica::graph::SpanningTree {
//     black_box(black_box(graph.0).get_spanning_tree(black_box(graph.1)))
// }

// library_benchmark_group!(
//     name = bench_stcut_group;
//     benchmarks = bench_stcut
// );

// library_benchmark_group!(
//     name = bench_tree_group;

//     compare_by_id = true;
//     benchmarks = bench_spanning,bench_spanning_sym,bench_spanning_uf,bench_spanning_uf_bfs
// );

// main!(
//     config = LibraryBenchmarkConfig::default()
//             .flamegraph(FlamegraphConfig::default());
//     library_benchmark_groups = bench_stcut_group,bench_tree_group
// );
fn main() {
    // main!()
}
