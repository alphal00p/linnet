use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, Criterion};
use figment::{providers::Serialized, Figment, Profile};
use linnest::TypstGraph;
use linnet::{dot, parser::DotGraph};

fn small_dot_graph() -> DotGraph {
    dot!(digraph small_layout {
        a -> b; a -> c; a -> d; a -> e;
        b -> c; b -> f; b -> g;
        c -> d; c -> h; c -> i;
        d -> e; d -> j; d -> f;
        e -> f; e -> g;
        f -> h; f -> i;
        g -> h; g -> j;
        h -> i; h -> j;
        i -> j;
    })
    .unwrap()
}

fn figment_for_incremental(enabled: bool) -> Figment {
    if enabled {
        Figment::from(Serialized::from(
            BTreeMap::<String, String>::new(),
            Profile::Default,
        ))
    } else {
        let mut map = BTreeMap::new();
        map.insert("incremental_energy".to_string(), "false".to_string());
        Figment::from(Serialized::from(map, Profile::Default))
    }
}

fn bench_small_layout(c: &mut Criterion) {
    let dot = small_dot_graph();
    let figment_incremental = figment_for_incremental(true);
    let figment_full = figment_for_incremental(false);

    let mut graph_full: TypstGraph = TypstGraph::from_dot(dot.clone(), &figment_full);
    let mut graph: TypstGraph = TypstGraph::from_dot(dot.clone(), &figment_incremental);
    c.bench_function("typst_layout_small_graph_incremental", |b| {
        b.iter(|| {
            graph.layout();
        });
    });

    c.bench_function("typst_layout_small_graph_full_energy", |b| {
        b.iter(|| {
            graph_full.layout();
        });
    });
}

criterion_group!(layout_benches, bench_small_layout);
criterion_main!(layout_benches);
