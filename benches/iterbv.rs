use std::ops::ControlFlow;

use bit_vec::BitVec;
use fixedbitset::FixedBitSet;
use hi_sparse_bitset::{BitSet, BitSetInterface};
use iai_callgrind::{black_box, library_benchmark, library_benchmark_group, main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vob::Vob;

fn print_true_vals(num_bytes: usize) {
    println!("true vals: {num_bytes}");
}

// ===== Helper: Create a pair of two different seeds =====
fn new_rng_pair(seed_a: u64, seed_b: u64) -> (StdRng, StdRng) {
    (StdRng::seed_from_u64(seed_a), StdRng::seed_from_u64(seed_b))
}

// ----- Vob -----
// Setup: construct a vob::Vob with the given size and set each bit true with probability ‘density’.
fn setup_vob((size, density): (usize, f64)) -> vob::Vob {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bits = vob::Vob::from_elem(false, size);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            bits.set(i, true);
        }
    }
    bits
}

// For union/intersection benchmarks, we need a pair.
fn setup_vob_pair((size, density): (usize, f64)) -> (vob::Vob, vob::Vob) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = vob::Vob::from_elem(false, size);
    let mut b = vob::Vob::from_elem(false, size);
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.set(i, true);
        }
        if rng2.gen::<f64>() < density {
            b.set(i, true);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_vob, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_vob, teardown = print_true_vals)]
// new macro lines for density 0.9
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_vob, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_vob, teardown = print_true_vals)]
fn bench_vob(bits: vob::Vob) -> usize {
    // Count all bits that are true.
    let count = bits.iter_set_bits(..).count();
    black_box(count);
    count
}

#[library_benchmark]
// Vob union benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_vob_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_vob_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_vob_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_vob_pair)]
fn bench_vob_union(pair: (vob::Vob, vob::Vob)) -> Vob {
    let (mut a, b) = pair;
    // Use bitwise OR to compute the union (vob implements BitOr).
    a |= &b;
    // let count = union.iter_set_bits(..).count();
    // black_box(count);

    a
}

#[library_benchmark]
// Vob intersection benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_vob_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_vob_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_vob_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_vob_pair)]
fn bench_vob_intersection(pair: (vob::Vob, vob::Vob)) -> vob::Vob {
    let (mut a, b) = pair;
    // Use bitwise AND to compute the intersection.
    a &= &b;
    // let count = inter.iter_set_bits(..).count();
    // black_box(count);
    a
}

// ----- Bitvec (from the bitvec crate) -----
// Setup: create a bitvec::vec::BitVec with the given size and density.
fn setup_bitvec((size, density): (usize, f64)) -> bitvec::vec::BitVec {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bits = bitvec::vec::BitVec::repeat(false, size);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            bits.set(i, true);
        }
    }
    bits
}

fn setup_bitvec_pair((size, density): (usize, f64)) -> (bitvec::vec::BitVec, bitvec::vec::BitVec) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = bitvec::vec::BitVec::repeat(false, size);
    let mut b = bitvec::vec::BitVec::repeat(false, size);
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.set(i, true);
        }
        if rng2.gen::<f64>() < density {
            b.set(i, true);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bitvec, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_bitvec, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bitvec, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bitvec, teardown = print_true_vals)]
fn bench_bitvec(bits: bitvec::vec::BitVec) -> usize {
    let count = bits.iter_ones().count();
    black_box(count);
    count
}

#[library_benchmark]
// Bitvec union benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bitvec_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_bitvec_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bitvec_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bitvec_pair)]
fn bench_bitvec_union(pair: (bitvec::vec::BitVec, bitvec::vec::BitVec)) -> bitvec::vec::BitVec {
    let (mut a, b) = pair;
    a &= &b;
    // let count = inter.iter_ones().count();
    // black_box(count);
    a
}

#[library_benchmark]
// Bitvec intersection benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bitvec_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_bitvec_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bitvec_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bitvec_pair)]
fn bench_bitvec_intersection(
    pair: (bitvec::vec::BitVec, bitvec::vec::BitVec),
) -> bitvec::vec::BitVec {
    let (mut a, b) = pair;
    a &= &b;
    // let count = inter.iter_ones().count();
    // black_box(count);
    a
}

// ----- SmallBitVec -----
// Setup: create a smallbitvec::SmallBitVec with the given capacity and density.
fn setup_smallbitvec((size, density): (usize, f64)) -> smallbitvec::SmallBitVec {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bits = smallbitvec::SmallBitVec::with_capacity(size);
    bits.resize(size, false);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            bits.set(i, true);
        }
    }
    bits
}

fn setup_smallbitvec_pair(
    (size, density): (usize, f64),
) -> (smallbitvec::SmallBitVec, smallbitvec::SmallBitVec) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = smallbitvec::SmallBitVec::with_capacity(size);
    a.resize(size, false);
    let mut b = smallbitvec::SmallBitVec::with_capacity(size);
    b.resize(size, false);
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.set(i, true);
        }
        if rng2.gen::<f64>() < density {
            b.set(i, true);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_smallbitvec, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_smallbitvec, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_smallbitvec, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_smallbitvec, teardown = print_true_vals)]
fn bench_smallbitvec(bits: smallbitvec::SmallBitVec) -> usize {
    let count = bits.iter().filter(|&b| b).count();
    black_box(count);
    count
}

// ----- bit-vec -----
// Setup: create a bit_vec::BitVec with the given size and density.
fn setup_bit_vec((size, density): (usize, f64)) -> bit_vec::BitVec {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bits = bit_vec::BitVec::from_elem(size, false);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            bits.set(i, true);
        }
    }
    bits
}

fn setup_bit_vec_pair((size, density): (usize, f64)) -> (bit_vec::BitVec, bit_vec::BitVec) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = bit_vec::BitVec::from_elem(size, false);
    let mut b = bit_vec::BitVec::from_elem(size, false);
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.set(i, true);
        }
        if rng2.gen::<f64>() < density {
            b.set(i, true);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bit_vec, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_bit_vec, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bit_vec, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bit_vec, teardown = print_true_vals)]
fn bench_bit_vec(bits: bit_vec::BitVec) -> usize {
    let count = bits.iter().filter(|&b| b).count();
    black_box(count);
    count
}

// #[library_benchmark]
// // bit-vec union benchmark.
// #[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup(args = [(100000, 0.1)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// fn bench_bit_vec_union(pair: (bit_vec::BitVec, bit_vec::BitVec)) -> usize {
//     let (mut a, b) = pair;
//     // Assuming bit_vec supports bitwise OR (or you might implement manually):
//     a.un b;
//     let count = union.iter().filter(|&b| b).count();
//     black_box(count);
//     count
// }

// #[library_benchmark]
// // bit-vec intersection benchmark.
// #[bench::with_setup_small(args = [(46, 0.1)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup(args = [(100000, 0.1)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// #[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_bit_vec_pair, teardown = print_true_vals)]
// fn bench_bit_vec_intersection(pair: (bit_vec::BitVec, bit_vec::BitVec)) -> usize {
//     let (a, b) = pair;
//     let inter = a.clone() & b.clone();
//     let count = inter.iter().filter(|&b| b).count();
//     black_box(count);
//     count
// }

// ----- Roaring Bitmap -----
// Setup: create a roaring::RoaringBitmap by adding indices [0, size) with probability `density`.
fn setup_roaring((size, density): (usize, f64)) -> roaring::RoaringBitmap {
    let mut rng = StdRng::seed_from_u64(42);
    let mut bitmap = roaring::RoaringBitmap::new();
    for i in 0..size {
        if rng.gen::<f64>() < density {
            bitmap.insert(i as u32);
        }
    }
    bitmap
}

fn setup_roaring_pair(
    (size, density): (usize, f64),
) -> (roaring::RoaringBitmap, roaring::RoaringBitmap) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = roaring::RoaringBitmap::new();
    let mut b = roaring::RoaringBitmap::new();
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.insert(i as u32);
        }
        if rng2.gen::<f64>() < density {
            b.insert(i as u32);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_roaring, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_roaring, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_roaring, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_roaring, teardown = print_true_vals)]
fn bench_roaring(bitmap: roaring::RoaringBitmap) -> usize {
    let count = bitmap.iter().count();
    black_box(count);
    count
}

#[library_benchmark]
// Roaring union benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_roaring_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_roaring_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_roaring_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_roaring_pair)]
fn bench_roaring_union(
    pair: (roaring::RoaringBitmap, roaring::RoaringBitmap),
) -> roaring::RoaringBitmap {
    let (mut a, b) = pair;
    a |= &b;
    a
}

#[library_benchmark]
// Roaring intersection benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_roaring_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_roaring_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_roaring_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_roaring_pair)]
fn bench_roaring_intersection(
    pair: (roaring::RoaringBitmap, roaring::RoaringBitmap),
) -> roaring::RoaringBitmap {
    let (mut a, b) = pair;
    a &= &b;
    a
}
fn setup_hiset((size, density): (usize, f64)) -> BitSet<hi_sparse_bitset::config::_64bit> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut set = BitSet::new();
    for i in 0..size {
        if rng.gen::<f64>() < density {
            set.insert(i);
        }
    }
    set
}

// Setup a pair of hi_sparse_bitset instances for union/intersection benchmarks
fn setup_hiset_pair(
    (size, density): (usize, f64),
) -> (
    BitSet<hi_sparse_bitset::config::_64bit>,
    BitSet<hi_sparse_bitset::config::_64bit>,
) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = BitSet::new();
    let mut b = BitSet::new();
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.insert(i);
        }
        if rng2.gen::<f64>() < density {
            b.insert(i);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hiset, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hiset, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hiset, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hiset, teardown = print_true_vals)]
fn bench_hiset(bitmap: BitSet<hi_sparse_bitset::config::_64bit>) -> usize {
    let mut iter = bitmap.iter();
    let mut count = 0;
    iter.traverse(|a| {
        count += 1;
        ControlFlow::Continue(())
    });
    black_box(count);
    count
}

#[library_benchmark]
// Roaring union benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hiset_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hiset_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hiset_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hiset_pair)]
fn bench_hiset_union(
    pair: (
        BitSet<hi_sparse_bitset::config::_64bit>,
        BitSet<hi_sparse_bitset::config::_64bit>,
    ),
) -> usize {
    let (mut a, b) = pair;
    (&a | &b).iter().count()
}

#[library_benchmark]
// Roaring intersection benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hiset_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hiset_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hiset_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hiset_pair)]
fn bench_hiset_intersection(
    pair: (
        BitSet<hi_sparse_bitset::config::_64bit>,
        BitSet<hi_sparse_bitset::config::_64bit>,
    ),
) -> usize {
    let (mut a, b) = pair;
    (&a & &b).iter().count()
}

fn setup_hibitset((size, density): (usize, f64)) -> hibitset::BitSet {
    let mut rng = StdRng::seed_from_u64(42);
    let mut set = hibitset::BitSet::new();
    for i in 0..size {
        if rng.gen::<f64>() < density {
            set.add(i as u32);
        }
    }
    set
}

// Setup a pair of hi_sparse_bitset instances for union/intersection benchmarks
fn setup_hibitset_pair((size, density): (usize, f64)) -> (hibitset::BitSet, hibitset::BitSet) {
    let (mut rng1, mut rng2) = new_rng_pair(42, 43);
    let mut a = hibitset::BitSet::new();
    let mut b = hibitset::BitSet::new();
    for i in 0..size {
        if rng1.gen::<f64>() < density {
            a.add(i as u32);
        }
        if rng2.gen::<f64>() < density {
            b.add(i as u32);
        }
    }
    (a, b)
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hibitset, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hibitset, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hibitset, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hibitset, teardown = print_true_vals)]
fn bench_hibitset(bitmap: hibitset::BitSet) -> usize {
    let mut sum = 0;
    black_box(hibitset::BitSetLike::iter(bitmap)).for_each(|x| sum += x);
    // black_box(count);
    sum as usize
}

#[library_benchmark]
// Roaring union benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hibitset_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hibitset_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hibitset_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hibitset_pair)]
fn bench_hibitset_union(pair: (hibitset::BitSet, hibitset::BitSet)) -> usize {
    let (mut a, b) = pair;
    hibitset::BitSetLike::iter((&a | &b)).count()
}

#[library_benchmark]
// Roaring intersection benchmark.
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_hibitset_pair)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_hibitset_pair)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_hibitset_pair)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_hibitset_pair)]
fn bench_hibitset_intersection(pair: (hibitset::BitSet, hibitset::BitSet)) -> usize {
    let (mut a, b) = pair;
    hibitset::BitSetLike::iter((&a & &b)).count()
}

fn setup_vecbitset((size, density): (usize, f64)) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut set = Vec::with_capacity(size);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            set.push(i);
        }
    }
    set
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_vecbitset, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_vecbitset, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_vecbitset, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_vecbitset, teardown = print_true_vals)]
fn bench_vecbitset(bitmap: Vec<usize>) -> usize {
    let mut sum = 0;
    black_box(bitmap).iter().for_each(|x| sum += x);
    // black_box(count);
    sum
}

fn setup_fixedbitset((size, density): (usize, f64)) -> FixedBitSet {
    let mut rng = StdRng::seed_from_u64(42);
    let mut set = FixedBitSet::with_capacity(size);
    for i in 0..size {
        if rng.gen::<f64>() < density {
            set.set(i, true);
        }
    }
    set
}

#[library_benchmark]
#[bench::with_setup_small(args = [(46, 0.1)], setup = setup_fixedbitset, teardown = print_true_vals)]
#[bench::with_setup(args = [(100000, 0.1)], setup = setup_fixedbitset, teardown = print_true_vals)]
#[bench::with_setup_small_dense(args = [(46, 0.9)], setup = setup_fixedbitset, teardown = print_true_vals)]
#[bench::with_setup_dense(args = [(100000, 0.9)], setup = setup_fixedbitset, teardown = print_true_vals)]
fn bench_fixedbitset(bitmap: FixedBitSet) -> usize {
    let mut sum = 0;
    black_box(bitmap).ones().for_each(|x| sum += x);
    // black_box(count);
    sum
}

// ----- Group & Main -----
// Group all library benchmarks into one group.
library_benchmark_group!(
    name = bitvec_iter_group;

    compare_by_id = true;
    benchmarks =
        bench_vob,
        bench_bitvec,
        bench_smallbitvec,
        bench_bit_vec,
        bench_roaring,
        bench_hiset,
        bench_hibitset,
        bench_vecbitset,
        bench_fixedbitset
);

library_benchmark_group!(
    name = bitvec_union_group;
    compare_by_id = true;

    benchmarks =
        bench_vob_union,
        bench_bitvec_union,
        // bench_smallbitvec_union,
        // bench_bit_vec_union,
        bench_roaring_union,
        bench_hiset_union
);

library_benchmark_group!(
    name = bitvec_intersection_group;
    // compare_by_id = true;

    benchmarks =
        bench_vob_intersection,
        bench_bitvec_intersection,
        // bench_smallbitvec_intersection,
        // bench_bit_vec_intersection,
        bench_roaring_intersection,
        bench_hiset_intersection
);

// Run the benchmark harness.
main!(
    library_benchmark_groups = bitvec_iter_group,
    // bitvec_union_group,
    // bitvec_intersection_group
);
