use criterion::{criterion_group, criterion_main, Criterion};
use figment::Figment;
use linnest::TypstGraph;
#[cfg(feature = "energy_trace")]
use linnet::half_edge::layout::spring::{energy_timing_reset, energy_timing_snapshot};
use linnet::{
    half_edge::{layout::simulatedanneale::Energy, subgraph::ModifySubSet, NodeIndex},
    parser::DotGraph,
};
#[cfg(feature = "energy_trace")]
#[derive(Clone, Copy, Default)]
struct EnergyAvg {
    total: f64,
    vv: f64,
    ev: f64,
    spring: f64,
    ee_local: f64,
    center: f64,
    dangling: f64,
    crossing: f64,
}
const GL303_DOT: &str = include_str!("../linnest/examples/gl303.dot");

fn bench_gl303_energy(c: &mut Criterion) {
    let dot: DotGraph = DotGraph::from_string(GL303_DOT).unwrap();
    let graph: TypstGraph = TypstGraph::from_dot(dot, &Figment::new());
    let (mut state, energy) = graph.layout_energy_state();

    state.incremental = true;

    let mut next = state.clone();
    if next.vertex_points.len().0 > 0 {
        let idx = NodeIndex(0);
        next.vertex_points[idx].x += 0.01;
        next.changed_nodes.add(idx);
    }

    let prev_energy = energy.energy(None, &state);

    #[cfg(feature = "energy_trace")]
    let total_avg = std::cell::RefCell::new(None::<EnergyAvg>);
    #[cfg(feature = "energy_trace")]
    let partial_avg = std::cell::RefCell::new(None::<EnergyAvg>);

    c.bench_function("energy_gl303_total", |b| {
        #[cfg(feature = "energy_trace")]
        b.iter_custom(|iters| {
            energy_timing_reset();
            for _ in 0..iters {
                energy.energy(None, &state);
            }
            let t = energy_timing_snapshot();
            let iters_f = iters as f64;
            total_avg.replace(Some(EnergyAvg {
                total: t.total_energy_ns as f64 / iters_f,
                vv: t.vv_ns as f64 / iters_f,
                ev: t.ev_ns as f64 / iters_f,
                spring: t.spring_ns as f64 / iters_f,
                ee_local: t.ee_local_ns as f64 / iters_f,
                center: t.center_ns as f64 / iters_f,
                dangling: t.dangling_ns as f64 / iters_f,
                crossing: t.crossing_ns as f64 / iters_f,
            }));
            std::time::Duration::from_nanos(t.total_energy_ns)
        });
        #[cfg(not(feature = "energy_trace"))]
        b.iter(|| {
            energy.energy(None, &state);
        });
    });

    c.bench_function("energy_gl303_partial", |b| {
        #[cfg(feature = "energy_trace")]
        b.iter_custom(|iters| {
            energy_timing_reset();
            for _ in 0..iters {
                energy.energy(Some((&state, prev_energy)), &next);
            }
            let t = energy_timing_snapshot();
            let iters_f = iters as f64;
            partial_avg.replace(Some(EnergyAvg {
                total: t.partial_energy_ns as f64 / iters_f,
                vv: t.vv_ns as f64 / iters_f,
                ev: t.ev_ns as f64 / iters_f,
                spring: t.spring_ns as f64 / iters_f,
                ee_local: t.ee_local_ns as f64 / iters_f,
                center: t.center_ns as f64 / iters_f,
                dangling: t.dangling_ns as f64 / iters_f,
                crossing: t.crossing_ns as f64 / iters_f,
            }));
            std::time::Duration::from_nanos(t.partial_energy_ns)
        });
        #[cfg(not(feature = "energy_trace"))]
        b.iter(|| {
            energy.energy(Some((&state, prev_energy)), &next);
        });
    });

    #[cfg(feature = "energy_trace")]
    {
        let total = total_avg.borrow_mut().take();
        let partial = partial_avg.borrow_mut().take();
        if let (Some(t), Some(p)) = (total, partial) {
            eprintln!(
                "energy_trace averages (ns/iter): total={:.2} vv={:.2} ev={:.2} spring={:.2} ee_local={:.2} center={:.2} dangling={:.2} crossing={:.2}",
                t.total, t.vv, t.ev, t.spring, t.ee_local, t.center, t.dangling, t.crossing
            );
            eprintln!(
                "energy_trace averages (ns/iter): partial={:.2} vv={:.2} ev={:.2} spring={:.2} ee_local={:.2} center={:.2} dangling={:.2} crossing={:.2}",
                p.total, p.vv, p.ev, p.spring, p.ee_local, p.center, p.dangling, p.crossing
            );
        }
    }
}

criterion_group!(energy_benches, bench_gl303_energy);
criterion_main!(energy_benches);
