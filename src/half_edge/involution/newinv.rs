use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use super::EdgeData;
use super::Flow;

/// An index type that can be used to index nodes in a graph.
/// Used to save memory for small graphs.
pub trait HedgeIndex: Default + Copy + PartialOrd + Ord + Eq + Hash + Display + Debug {
    const MAX: usize;
    fn to_usize(&self) -> usize;
    fn from_usize(x: usize) -> Self;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hedex<I = usize> {
    index: I,
}

macro_rules! impl_hedge_index {
    ($type:ty, $max:expr) => {
        impl HedgeIndex for Hedex<$type> {
            const MAX: usize = $max;

            #[inline]
            fn to_usize(&self) -> usize {
                self.index as usize
            }

            #[inline]
            fn from_usize(x: usize) -> Self {
                debug_assert!(
                    x <= Self::MAX,
                    "Index {} exceeds maximum value {}",
                    x,
                    Self::MAX
                );
                Self { index: x as $type }
            }
        }

        impl Default for Hedex<$type> {
            fn default() -> Self {
                Self { index: 0 }
            }
        }

        impl std::fmt::Display for Hedex<$type> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.index)
            }
        }
    };
}

// Implement for common unsigned integer types
impl_hedge_index!(u8, u8::MAX as usize);
impl_hedge_index!(u16, u16::MAX as usize);
impl_hedge_index!(u32, u32::MAX as usize);
impl_hedge_index!(u64, u64::MAX as usize);
impl_hedge_index!(usize, usize::MAX);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Hedge<E, Idx> {
    Identity { data: EdgeData<E>, underlying: Flow },
    Source { data: EdgeData<E>, sink_idx: Idx },
    Sink { source_idx: Idx },
    Dummy,
}

/// Involutive mapping between half-edges.
///
/// It is a data structure that represents a self inverse permutation of half-edges.
/// And allows for data carrying only on one of the two half-edges.
///
/// If the half-edge is mapped to itself, we call it an Identity hedge.
/// It always carries data, and a flow.
///
/// If the half-edge is mapped to another half-edge, it is either a Source or a Sink hedge.
/// Source hedges carry data, while Sink hedges carry a source index.
///
/// Additionally, it guarantees that all data carrying half-edges (Sources and Identities) are at the start of the vector.
///
/// This means that indices up to `n_data` can function as full edge indices.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Inv<D, Idx: HedgeIndex> {
    pub(super) n_data: Idx, // number of data carriers
    pub(super) inv: Vec<Hedge<D, Idx>>,
}
