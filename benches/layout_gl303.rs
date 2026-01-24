use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, Criterion};
use figment::{providers::Serialized, Figment, Profile};
use linnest::TypstGraph;
use linnet::parser::DotGraph;

const GL303_DOT: &str = include_str!("../linnest/examples/gl303.dot");

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

fn bench_gl303_layout(c: &mut Criterion) {
    let dot: DotGraph = DotGraph::from_string(GL303_DOT).unwrap();
    let figment_incremental = figment_for_incremental(true);
    let figment_full = figment_for_incremental(false);

    let mut graph_full: TypstGraph = TypstGraph::from_dot(dot.clone(), &figment_full);
    let mut graph: TypstGraph = TypstGraph::from_dot(dot, &figment_incremental);

    c.bench_function("typst_layout_gl303_incremental", |b| {
        b.iter(|| {
            graph.layout();
        });
    });

    c.bench_function("typst_layout_gl303_full_energy", |b| {
        b.iter(|| {
            graph_full.layout();
        });
    });
}

criterion_group!(layout_benches, bench_gl303_layout);
criterion_main!(layout_benches);
