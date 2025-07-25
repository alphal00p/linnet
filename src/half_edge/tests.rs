#[test]
fn cycle_basis() {
    let two: DotGraph = dot!(
    digraph {
          ext0 [shape=none, label="" flow=sink];
          ext0 -> 2[dir=none color="red"];
          ext1 [shape=none, label="" flow=source];
          ext1 -> 3[dir=none color="blue"];
          2 -> 3[ dir=none color="red:blue;0.5"];
          3 -> 2[ dir=none color="red:blue;0.5"];
    })
    .unwrap();

    two.all_cycle_sym_diffs().unwrap();
    assert_eq!(two.cycle_basis().0.len(), 1);
}

#[test]
fn test_spanning_trees_of_tree() {
    let tree: DotGraph = dot!(
    digraph {
         1->2
         1->0
         2->5
          2 -> 3
          3 -> 4
    })
    .unwrap();
    let trees = tree.all_spanning_trees(&tree.full_filter());

    for t in &trees {
        println!("{}", tree.dot(t))
    }

    assert_eq!(trees.len(), 1);
}

#[test]
fn join_mut_simple() {
    let two: DotGraph = dot!(
    digraph {

      0 [label = "∑"];
      1 [label = "S:4"];
      ext0 [shape=none, label="" flow=sink];
      ext0 -> 0[dir=back color="red"];
      ext2 [shape=none, label="" flow=source];
      ext2 -> 0[dir=forward color="blue"];
      1 -> 0[ dir=forward color="red:blue;0.5"];
    })
    .unwrap();

    //with

    let one: DotGraph = dot!(digraph {
      node [shape=circle,height=0.1,label=""];  overlap="scale"; layout="neato";

      0 [label = "S:5"];
      ext0 [shape=none, label="" flow=sink];
      ext0 -> 0[dir=back color="red"];
    })
    .unwrap();

    let mut one = one.graph;

    one.join_mut(
        two.graph,
        |sf, _, of, _| {
            println!("{sf:?}vs{of:?}");
            sf == -of
        },
        |sf, sd, _, _| (sf, sd),
    )
    .unwrap();

    println!("{}", one.base_dot())
}

#[test]
fn threeloop() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());

    builder.add_edge(a, b, (), false);
    builder.add_edge(b, a, (), false);
    builder.add_edge(a, b, (), false);

    builder.add_edge(b, c, (), false);
    builder.add_edge(c, d, (), false);
    builder.add_edge(d, a, (), false);

    let graph: HedgeGraph<(), (), ()> = builder.build();

    insta::assert_snapshot!("three_loop_dot", graph.base_dot());
    #[cfg(feature = "serde")]
    insta::assert_ron_snapshot!("three_loop", graph);

    for i in 0..graph.n_hedges() {
        assert_eq!(
            3,
            graph
                .paton_cycle_basis(&graph.full_graph(), &graph.node_id(Hedge(i)), None)
                .unwrap()
                .0
                .len()
        );
    }

    let (cycles, tree) = graph.cycle_basis();
    assert_eq!(tree.covers(&graph.full_filter()), graph.full_filter());

    assert_eq!(3, cycles.len());

    let all_cycles = graph.all_cycles();

    assert_eq!(6, all_cycles.len());

    #[cfg(feature = "serde")]
    insta::assert_ron_snapshot!("three_loop_cycles", cycles);
}

#[test]
fn hairythreeloop() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());

    builder.add_edge(a, b, (), false);
    builder.add_edge(b, a, (), false);
    builder.add_edge(a, b, (), false);
    builder.add_external_edge(a, (), false, Flow::Sink);
    builder.add_external_edge(b, (), false, Flow::Sink);
    builder.add_external_edge(b, (), false, Flow::Sink);

    builder.add_edge(b, c, (), false);
    builder.add_edge(c, d, (), false);
    builder.add_edge(d, a, (), false);

    assert_eq!(builder.involution.len(), 15);
    let graph: HedgeGraph<(), (), ()> = builder.build();

    insta::assert_snapshot!("hairy_three_loop_dot", graph.base_dot());

    #[cfg(feature = "serde")]
    insta::assert_ron_snapshot!("hairy_three_loop", graph);
    insta::assert_snapshot!(
        "hairy_three_loop_dot_internal",
        graph.dot(&graph.full_node())
    );

    for i in graph.full_node().internal_graph.filter.included_iter() {
        assert_eq!(
            3,
            graph
                .paton_cycle_basis(&graph.full_graph(), &graph.node_id(i), None)
                .unwrap()
                .0
                .len()
        );
    }

    #[cfg(feature = "serde")]
    let cycles = graph.cycle_basis().0;

    #[cfg(feature = "serde")]
    insta::assert_ron_snapshot!("hairy_three_loop_cycles", cycles);
}

#[test]
fn banana_cuts() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    builder.add_edge(a, b, (), false);
    builder.add_edge(a, b, (), false);
    builder.add_edge(a, b, (), false);

    let three_banana: HedgeGraph<(), (), ()> = builder.clone().build();

    assert_eq!(6, three_banana.non_cut_edges().len());
    builder.add_edge(a, b, (), false);

    let four_banana: HedgeGraph<(), (), ()> = builder.build();
    assert_eq!(14, four_banana.non_cut_edges().len());
}

#[test]
fn three_loop_fly() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());
    builder.add_edge(a, b, (), false);
    builder.add_edge(b, a, (), false);
    builder.add_edge(b, c, (), false);
    builder.add_edge(d, a, (), false);

    builder.add_edge(c, d, (), false);
    builder.add_edge(d, c, (), false);

    let fly: HedgeGraph<(), (), ()> = builder.clone().build();
    assert_eq!(32, fly.non_cut_edges().len());
}

#[test]
fn doubletriangle() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());
    builder.add_edge(a, b, (), true);
    builder.add_edge(b, c, (), true);
    builder.add_edge(d, a, (), true);

    builder.add_edge(c, d, (), true);
    builder.add_edge(a, c, (), true);

    let fly: HedgeGraph<(), (), ()> = builder.clone().build();

    for _c in fly.non_cut_edges() {
        // println!("{c:?}");
        //
        // sum += (2 ^ (c.count_ones() / 2)) / 2;

        // println!("{}", fly.dot(&c));
    }
    // println!("{sum}");
    assert_eq!(13, fly.non_cut_edges().len());
    // println!("{}", SignedCut::all_initial_state_cuts(&fly).len());

    // for c in SignedCut::all_initial_state_cuts(&fly) {
    //     println!("//{}", c.bare_signature(&fly));
    //     println!("{}", fly.dot(&c.cut_content));
    // }
}

#[test]
fn cube() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());
    let e = builder.add_node(());
    let f = builder.add_node(());
    let g = builder.add_node(());
    let h = builder.add_node(());

    builder.add_edge(a, b, (), false);
    builder.add_edge(b, c, (), false);
    builder.add_edge(c, d, (), false);
    builder.add_edge(d, a, (), false);

    builder.add_edge(e, f, (), false);
    builder.add_edge(f, g, (), false);
    builder.add_edge(g, h, (), false);
    builder.add_edge(h, e, (), false);

    builder.add_edge(a, e, (), false);
    builder.add_edge(b, f, (), false);
    builder.add_edge(c, g, (), false);
    builder.add_edge(d, h, (), false);

    let graph: HedgeGraph<(), (), ()> = builder.build();

    insta::assert_snapshot!("cube_dot", graph.base_dot());

    // let mut all_spinneys = graph.all_spinneys().into_iter().collect_vec();
    // all_spinneys.sort_by(|a, b| a.filter.cmp(&b.filter));

    // assert_eq!(162, all_spinneys.len());
}

#[test]
#[ignore]
fn alt_vs_pair() {
    for s in 0..100 {
        let rand_graph = HedgeGraph::<(), (), (), NodeStorageVec<()>>::random(10, 14, s);

        let before = Instant::now();
        let all_spinneys = rand_graph.all_spinneys();
        let after = before.elapsed();
        let before = Instant::now();
        let all_spinneys_alt = rand_graph.all_spinneys_alt();
        let after_alt = before.elapsed();
        println!("{s} {after:?} {after_alt:?}");

        assert_eq!(
            all_spinneys.len(),
            all_spinneys_alt.len(),
            "{}",
            rand_graph.base_dot()
        );
    }
    // let rand_graph = NestingGraph::<(), ()>::random(6, 9, 8);

    // println!("{}", rand_graph.base_dot());

    // println!("loops {}", rand_graph.cycle_basis().len());

    // let all_spinneys_other = rand_graph.all_spinneys();

    // // println!("all spinneys read tarjan {}", all_spinneys.len());

    // println!("all spinneys {}", all_spinneys_other.len());
    // println!("all spinneys alt{}", rand_graph.all_spinneys_alt().len());
}

// #[test]
// #[should_panic]
// fn read_tarjan_vs_cycle_space() {
//     for s in 0..100 {
//         let rand_graph = HedgeGraph::<(), ()>::random(6, 9, s);

//         let all_cycles = rand_graph.read_tarjan();
//         let all_cycles_alt = rand_graph.all_cycles();

//         assert_eq!(
//             all_cycles.len(),
//             all_cycles_alt.len(),
//             "{} with seed {s}",
//             rand_graph.base_dot()
//         );
//     }
// }

// #[test]
// fn random_graph() {
//     let rand_graph = NestingGraph::<(), ()>::random(6, 9, 3);

//     println!(
//         "{} loop graph: \n {}",
//         rand_graph.cycle_basis().len(),
//         rand_graph.base_dot()
//     );

//     for c in rand_graph.all_cycles() {
//         println!(" {}", rand_graph.dot(&c));
//     }

//     for c in rand_graph.read_tarjan() {
//         println!("{}", rand_graph.dot(&c));
//     }
// }
#[allow(non_snake_case)]
#[test]
fn K33() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());
    let e = builder.add_node(());
    let f = builder.add_node(());

    builder.add_edge(a, d, (), false);
    builder.add_edge(a, e, (), false);
    builder.add_edge(a, f, (), false);

    builder.add_edge(b, d, (), false);
    builder.add_edge(b, e, (), false);
    builder.add_edge(b, f, (), false);

    builder.add_edge(c, d, (), false);
    builder.add_edge(c, e, (), false);
    builder.add_edge(c, f, (), false);

    let graph: HedgeGraph<(), (), ()> = builder.build();
    graph.full_node();
    println!("built");

    println!("{}", graph.dot(&graph.full_node()));

    let t1 = SimpleTraversalTree::depth_first_traverse(
        &graph,
        &graph.full_filter(),
        &NodeIndex(4),
        None,
    )
    .unwrap();

    println!(
        "{}",
        graph.dot(&graph.nesting_node_from_subgraph(t1.tree_subgraph(&graph).clone()))
    );
    // println!("{:?}", t1.traversal);

    // println!(
    //     "{}",
    //     t1.inv.print(
    //         &graph.full_filter(),
    //         &|a| match a {
    //             Parent::Root => Some("Root       \t |     \t|".to_string()),
    //             Parent::Hedge {
    //                 hedge_to_root,
    //                 traversal_order,
    //             } => Some(format!(
    //                 "parent {} \t | rank {} \t|",
    //                 hedge_to_root, traversal_order
    //             )),
    //             Parent::Unset => None,
    //         },
    //         &|_| None
    //     )
    // );
    println!(
        "{}",
        graph.dot(&graph.nesting_node_from_subgraph(t1.tree_subgraph(&graph)))
    );
    println!(
        "{}",
        graph
            .paton_count_loops(&graph.full_graph(), &graph.node_id(Hedge(0)))
            .unwrap()
    );

    println!("{}", graph.cyclotomatic_number(&graph.full_graph()));

    let cycles = graph
        .paton_cycle_basis(&graph.full_graph(), &NodeIndex(4), None)
        .unwrap()
        .0;

    for c in cycles {
        println!(
            "{}",
            graph.dot(&graph.nesting_node_from_subgraph(c.internal_graph(&graph)))
        );
    }

    // assert_eq!(graph.all_spinneys().len(), graph.all_spinneys_alt().len());
}

#[test]
fn petersen() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());
    let e = builder.add_node(());
    let f = builder.add_node(());
    let g = builder.add_node(());
    let h = builder.add_node(());
    let i = builder.add_node(());
    let j = builder.add_node(());

    builder.add_edge(a, b, (), false);
    builder.add_edge(a, f, (), false);
    builder.add_edge(a, e, (), false);

    builder.add_edge(b, c, (), false);
    builder.add_edge(b, g, (), false);

    builder.add_edge(c, d, (), false);
    builder.add_edge(c, h, (), false);

    builder.add_edge(d, e, (), false);
    builder.add_edge(d, i, (), false);

    builder.add_edge(e, j, (), false);

    builder.add_edge(f, h, (), false);
    builder.add_edge(f, i, (), false);

    builder.add_edge(g, i, (), false);
    builder.add_edge(g, j, (), false);

    builder.add_edge(h, j, (), false);

    let graph: HedgeGraph<(), (), ()> = builder.build();

    println!("{}", graph.base_dot());

    // assert_eq!(graph.all_spinneys().len(), graph.all_spinneys_alt().len());

    println!("loop count {}", graph.cycle_basis().0.len());
    println!("cycle count {}", graph.all_cycles().len());
    println!(
        "loop count alt {}",
        graph.cyclotomatic_number(&graph.full_node().internal_graph)
    );
    if let Some((s, v)) = graph
        .all_spinneys()
        .iter()
        .find(|(s, _)| graph.full_filter() == s.filter)
    {
        println!(
            "{}",
            graph.dot(&graph.nesting_node_from_subgraph(s.clone()))
        );
        for (ci, cj) in v {
            println!("{}", graph.dot(&ci.to_hairy_subgraph(&graph)));
            println!(
                "{}",
                graph.dot(&cj.as_ref().unwrap().to_hairy_subgraph(&graph))
            );
        }
    } else {
        println!("not found");
    }
}

#[test]
fn wagner_graph() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let n1 = builder.add_node(());
    let n2 = builder.add_node(());
    let n3 = builder.add_node(());
    let n4 = builder.add_node(());
    let n5 = builder.add_node(());
    let n6 = builder.add_node(());
    let n7 = builder.add_node(());
    let n8 = builder.add_node(());

    builder.add_edge(n1, n2, (), false);
    builder.add_edge(n1, n5, (), false);

    builder.add_edge(n2, n3, (), false);
    builder.add_edge(n2, n6, (), false);

    builder.add_edge(n3, n4, (), false);
    builder.add_edge(n3, n7, (), false);

    builder.add_edge(n4, n5, (), false);
    builder.add_edge(n4, n8, (), false);

    builder.add_edge(n5, n6, (), false);

    builder.add_edge(n6, n7, (), false);

    builder.add_edge(n7, n8, (), false);

    builder.add_edge(n8, n1, (), false);

    let graph: HedgeGraph<(), (), ()> = builder.build();

    println!("{}", graph.base_dot());

    // assert_eq!(graph.all_spinneys().len(), graph.all_spinneys_alt().len());

    println!("loop count {}", graph.cycle_basis().0.len());
    println!("cycle count {}", graph.all_cycles().len());
    if let Some((s, v)) = graph
        .all_spinneys()
        .iter()
        .find(|(s, _)| graph.full_filter() == s.filter)
    {
        println!(
            "{}",
            graph.dot(&graph.nesting_node_from_subgraph(s.clone()))
        );
        for (ci, cj) in v {
            println!("{}", graph.dot(&ci.to_hairy_subgraph(&graph)));
            println!(
                "{}",
                graph.dot(&cj.as_ref().unwrap().to_hairy_subgraph(&graph))
            );
        }
    } else {
        println!("not found");
    }
}

#[test]
fn flower_snark() {
    let mut builder: HedgeGraphBuilder<(), ()> = HedgeGraphBuilder::new();
    let n1 = builder.add_node(());
    let n2 = builder.add_node(());
    let n3 = builder.add_node(());
    let n4 = builder.add_node(());
    let n5 = builder.add_node(());
    let n6 = builder.add_node(());
    let n7 = builder.add_node(());
    let n8 = builder.add_node(());
    let n9 = builder.add_node(());
    let n10 = builder.add_node(());
    let n11 = builder.add_node(());
    let n12 = builder.add_node(());
    let n13 = builder.add_node(());
    let n14 = builder.add_node(());
    let n15 = builder.add_node(());
    let n16 = builder.add_node(());
    let n17 = builder.add_node(());
    let n18 = builder.add_node(());
    let n19 = builder.add_node(());
    let n20 = builder.add_node(());

    builder.add_edge(n1, n2, (), false);
    builder.add_edge(n2, n3, (), false);
    builder.add_edge(n3, n4, (), false);
    builder.add_edge(n4, n5, (), false);
    builder.add_edge(n5, n1, (), false);

    builder.add_edge(n6, n1, (), false); // center
    builder.add_edge(n6, n7, (), false); //next

    builder.add_edge(n7, n17, (), false); //+10
    builder.add_edge(n7, n8, (), false); //next

    builder.add_edge(n8, n13, (), false); //+5
    builder.add_edge(n8, n9, (), false); //next

    builder.add_edge(n9, n2, (), false); //center
    builder.add_edge(n9, n10, (), false); //next

    builder.add_edge(n10, n20, (), false); //+10
    builder.add_edge(n10, n11, (), false); //next

    builder.add_edge(n11, n16, (), false); //+5
    builder.add_edge(n11, n12, (), false); //next

    builder.add_edge(n12, n3, (), false); //center
    builder.add_edge(n12, n13, (), false); //next

    builder.add_edge(n13, n14, (), false); //next

    builder.add_edge(n14, n19, (), false); //+5
    builder.add_edge(n14, n15, (), false); //next

    builder.add_edge(n15, n4, (), false); //center
    builder.add_edge(n15, n16, (), false); //next

    builder.add_edge(n16, n17, (), false); //next

    builder.add_edge(n17, n18, (), false); //next

    builder.add_edge(n18, n5, (), false); //center
    builder.add_edge(n18, n19, (), false); //next

    builder.add_edge(n19, n20, (), false); //next

    builder.add_edge(n20, n6, (), false); //next

    let graph: HedgeGraph<(), (), ()> = builder.build();

    println!("{}", graph.base_dot());

    // assert_eq!(graph.all_spinneys().len(), graph.all_spinneys_alt().len());
    assert_eq!(11, graph.cyclotomatic_number(&graph.full_graph()));
    println!("cycle count {}", graph.all_cycles().len());
    assert_eq!(
        11,
        graph
            .paton_count_loops(&graph.full_graph(), &graph.node_id(Hedge(0)))
            .unwrap()
    );
    if let Some((s, v)) = graph
        .all_spinneys()
        .iter()
        .find(|(s, _)| graph.full_filter() == s.filter)
    {
        println!(
            "{}",
            graph.dot(&graph.nesting_node_from_subgraph(s.clone()))
        );
        for (ci, cj) in v {
            println!("{}", graph.dot(&ci.to_hairy_subgraph(&graph)));
            println!(
                "{}",
                graph.dot(&cj.as_ref().unwrap().to_hairy_subgraph(&graph))
            );
        }
    } else {
        println!("not found");
    }
}

#[test]
fn join() {
    let mut ab = HedgeGraphBuilder::new();
    let v1 = ab.add_node("a");
    let v2 = ab.add_node("a");

    ab.add_edge(v1, v2, "ie", true);
    ab.add_edge(v2, v1, "ie", true);
    ab.add_external_edge(v1, "esink", true, Flow::Sink);
    ab.add_external_edge(v2, "esource", true, Flow::Source);

    let a: HedgeGraph<&'static str, &'static str> = ab.build();

    let mut ab = HedgeGraphBuilder::new();
    let v1 = ab.add_node("b");
    let v2 = ab.add_node("b");

    ab.add_edge(v1, v2, "if", true);
    ab.add_edge(v2, v1, "if", true);
    ab.add_external_edge(v1, "f", true, Flow::Sink);
    ab.add_external_edge(v2, "f", true, Flow::Source);

    let b = ab.build();

    let mut c = a
        .clone()
        .join(b.clone(), |af, _, bf, _| af == -bf, |af, ad, _, _| (af, ad))
        .unwrap();

    assert_eq!(c.node_store.node_data.len(), 4);
    assert_eq!(c.node_store.node_len(), 4);
    assert_eq!(c.node_store.hedge_len(), 12);
    assert!(c.node_store.check_and_set_nodes().is_ok());

    assert_snapshot!(c.dot_display(&c.full_filter()));

    let mut a = HedgeGraphBuilder::new();
    let n = a.add_node("a");
    a.add_external_edge(n, "e", true, Flow::Sink);
    let a: HedgeGraph<&'static str, &'static str> = a.build();

    let mut b = HedgeGraphBuilder::new();
    let n = b.add_node("b");
    b.add_external_edge(n, "f", true, Flow::Sink);
    let b = b.build();
    let c = a
        .join(
            b,
            |_, _, _, _| true,
            |af, ad, bf, bd| {
                println!("af: {af:?}, ad: {ad:?}, bf: {bf:?}, bd: {bd:?}");
                (af, ad)
            },
        )
        .unwrap();

    assert_eq!(
        "e",
        *c.iter_edges_of(&c.full_filter()).next().unwrap().2.data
    );
}
use std::time::Instant;

use insta::assert_snapshot;
use nodestore::NodeStorageVec;

use crate::{dot, parser::DotGraph};

use super::*;

#[test]
fn double_triangle() {
    let mut builder = HedgeGraphBuilder::new();
    let a = builder.add_node(());
    let b = builder.add_node(());
    let c = builder.add_node(());
    let d = builder.add_node(());

    builder.add_edge(a, b, (), true);
    builder.add_edge(b, c, (), true);
    builder.add_edge(c, d, (), true);
    builder.add_edge(d, a, (), true);
    builder.add_edge(b, d, (), true);

    let graph: HedgeGraph<(), (), ()> = builder.build();
    let cuts = graph.all_cuts_from_ids(&[a], &[c]);
    assert_eq!(cuts.len(), 4);
    for cut in cuts {
        assert!(!(cut.1.is_empty()))
    }
}

#[test]
fn double_self_loop() {
    let mut builder = HedgeGraphBuilder::new();
    let a = builder.add_node(());

    builder.add_edge(a, a, (), true);
    builder.add_edge(a, a, (), true);

    let graph: HedgeGraph<(), (), ()> = builder.build();
    let (loops, _) = graph.cycle_basis();
    assert_eq!(loops.len(), 2);
}

#[test]
fn self_energy_cut() {
    let mut epem_builder = HedgeGraphBuilder::new();
    let nodes = (0..4)
        .map(|_| epem_builder.add_node(()))
        .collect::<Vec<_>>();

    epem_builder.add_edge(nodes[0], nodes[1], (), true);
    epem_builder.add_edge(nodes[1], nodes[2], (), true);
    epem_builder.add_edge(nodes[1], nodes[2], (), true);
    epem_builder.add_edge(nodes[2], nodes[3], (), true);

    epem_builder.add_external_edge(nodes[0], (), true, Flow::Sink);
    epem_builder.add_external_edge(nodes[0], (), true, Flow::Sink);
    epem_builder.add_external_edge(nodes[3], (), true, Flow::Source);
    epem_builder.add_external_edge(nodes[3], (), true, Flow::Source);

    let epem: HedgeGraph<(), ()> = epem_builder.build::<NodeStorageVec<()>>();
    println!("{}", epem.dot(&epem.full_filter()));

    let cuts = epem.all_cuts_from_ids(&nodes[0..=0], &nodes[3..=3]);

    assert_eq!(cuts.len(), 3);
}

#[test]
fn double_pentagon_all_cuts() {
    let graph: DotGraph = dot!(
        digraph {
    node [shape=circle,height=0.1,label=""];  overlap="scale"; layout="neato";
    00 -> 07[ dir=none,label="a"];
    00 -> 12[ dir=forward,label="d"];
    01 -> 00[ dir=forward,label="d"];
    01 -> 03[ dir=none,label="a"];
    02 -> 01[ dir=forward,label="d"];
    02 -> 06[ dir=none,label="a"];
    03 -> 13[ dir=forward,label="d"];
    04 -> 03[ dir=forward,label="d"];
    04 -> 05[ dir=none,label="g"];
    05 -> 02[ dir=forward,label="d"];
    06 -> 07[ dir=forward,label="e-"];
    07 -> 11[ dir=forward,label="e-"];
    08 -> 06[ dir=forward,label="e-"];
    09 -> 04[ dir=forward,label="d"];
    10 -> 05[ dir=forward,label="d"];
    }
    )
    .unwrap();

    // println!(
    //     "{}",
    //     graph.dot_impl(&graph.full_filter(), "", &|a| None, &|n| Some(format!(
    //         "label=\"{}\"",
    //         n.id
    //     )))
    // );

    let cuts = graph.all_cuts(
        graph.iter_crown(NodeIndex(10)).clone().into(),
        graph.iter_crown(NodeIndex(9)).clone().into(),
    );

    assert_eq!(cuts.len(), 9);

    let cuts = graph.all_cuts(
        graph.iter_crown(NodeIndex(10)).clone().into(),
        graph.iter_crown(NodeIndex(13)).clone().into(),
    );

    assert_eq!(cuts.len(), 14);

    let cuts = graph.all_cuts(
        graph.iter_crown(NodeIndex(1)).clone().into(),
        graph.iter_crown(NodeIndex(2)).clone().into(),
    );

    assert_eq!(cuts.len(), 16);

    // for (l, c, r) in cuts {
    //     assert_eq!("//cut:\n{}", graph.dot(&c.reference));
    // }
    // let cuts = graph.all_cuts(NodeIndex(10), NodeIndex(13));

    // println!("All cuts: {}", cuts.len());
}

#[test]
fn box_all_cuts_multiple() {
    let graph: DotGraph = dot!(
        digraph G {
         00->01
         01->02
         02->03
         03->00
         10->00
         11->01
         12->02
         13->03
        }
    )
    .unwrap();

    // println!(
    //     "{}",
    //     graph.dot_impl(&graph.full_filter(), "", &|a| None, &|n| Some(format!(
    //         "label=\"{}\"",
    //         n.id
    //     )))
    // );

    let cuts =
        graph.all_cuts_from_ids(&[NodeIndex(4), NodeIndex(6)], &[NodeIndex(5), NodeIndex(7)]);

    // for (l, c, r) in &cuts {
    //     println!(
    //         "//cut:\n{}",
    //         graph.dot_impl(l, "start=2;\n", &|a| None, &|n| Some(format!(
    //             "label=\"{}\"",
    //             n.id
    //         )))
    //     );
    // }
    // let cuts = graph.all_cuts(NodeIndex(10), NodeIndex(13));

    assert_eq!(11, cuts.len());
}

#[test]
#[cfg(feature = "serde")]
fn self_energy_box() {
    let mut self_energy_builder: HedgeGraphBuilder<(), (), ()> = HedgeGraphBuilder::new();
    let nodes = (0..8)
        .map(|_| self_energy_builder.add_node(()))
        .collect::<Vec<_>>();

    self_energy_builder.add_edge(nodes[0], nodes[2], (), true);
    self_energy_builder.add_edge(nodes[2], nodes[1], (), true);
    self_energy_builder.add_edge(nodes[1], nodes[7], (), true);
    self_energy_builder.add_edge(nodes[7], nodes[5], (), true);
    self_energy_builder.add_edge(nodes[5], nodes[6], (), true);
    self_energy_builder.add_edge(nodes[6], nodes[0], (), true);
    self_energy_builder.add_edge(nodes[3], nodes[2], (), true);
    self_energy_builder.add_edge(nodes[4], nodes[5], (), true);
    self_energy_builder.add_edge(nodes[3], nodes[4], (), true);
    self_energy_builder.add_edge(nodes[4], nodes[3], (), true);

    self_energy_builder.add_external_edge(nodes[0], (), true, Flow::Sink);
    self_energy_builder.add_external_edge(nodes[1], (), true, Flow::Sink);
    self_energy_builder.add_external_edge(nodes[6], (), true, Flow::Source);
    self_energy_builder.add_external_edge(nodes[7], (), true, Flow::Source);

    let mut cut_to_look_for = vec![
        EdgeIndex::from(5),
        EdgeIndex::from(2),
        EdgeIndex::from(8),
        EdgeIndex::from(9),
    ];

    cut_to_look_for.sort();

    let self_energy: HedgeGraph<(), (), _> = self_energy_builder.build::<NodeStorageVec<()>>();

    let cuts = self_energy.all_cuts_from_ids(&[nodes[0]], &[nodes[7]]);

    for (left, cut, right) in cuts.iter() {
        let mut edges_in_cut: Vec<_> = self_energy
            .iter_edges_of(cut)
            .map(|(_, edge, _)| edge)
            .collect();

        edges_in_cut.sort();

        if cut_to_look_for == edges_in_cut {
            insta::assert_ron_snapshot!(left);

            insta::assert_ron_snapshot!(right);

            insta::assert_ron_snapshot!(cut.left);
        }
    }
}

#[test]
fn tadpoles() {
    let graph: DotGraph = dot!(
        digraph{
            a->b->c
            b->aa->d
            aa->ab
            ab->ab

            e->d->f
        }
    )
    .unwrap();

    let tads = graph.tadpoles(&[NodeIndex(0), NodeIndex(4), NodeIndex(6), NodeIndex(7)]);

    for t in tads {
        println!("//Tadpole: \n{}", graph.dot(&t));
    }
    println!("Graph: {}", graph.base_dot());
}
