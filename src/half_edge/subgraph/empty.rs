use std::{
    marker::PhantomData,
    ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive},
};

use bitvec::vec::BitVec;

use crate::half_edge::{
    involution::{Hedge, HedgePair},
    nodestore::NodeStorageOps,
    HedgeGraph,
};

use super::{Inclusion, SubGraph};

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
/// Represents an empty subgraph, containing no half-edges.
///
/// This struct is used as a placeholder or a null object in subgraph operations.
/// It implements the [`SubGraph`] trait, with most methods reflecting its emptiness
/// (e.g., `nhedges()` returns 0, `is_empty()` returns `true`).
pub struct Empty {
    /// The total number of half-edges in the context of the parent graph
    /// to which this empty subgraph belongs. This is used to maintain
    /// consistent sizing for operations, even though the subgraph itself is empty.
    size: usize,
}

impl Inclusion<Hedge> for Empty {
    fn includes(&self, _hedge_id: &Hedge) -> bool {
        false
    }

    fn intersects(&self, _other: &Hedge) -> bool {
        false
    }
}
impl Inclusion<HedgePair> for Empty {
    fn includes(&self, _other: &HedgePair) -> bool {
        false
    }

    fn intersects(&self, _other: &HedgePair) -> bool {
        false
    }
}

impl Inclusion<Empty> for Empty {
    fn includes(&self, other: &Empty) -> bool {
        // true
        self.size == other.size
    }

    fn intersects(&self, other: &Empty) -> bool {
        // true
        self.size == other.size
    }
}

impl Inclusion<BitVec> for Empty {
    fn includes(&self, other: &BitVec) -> bool {
        other.nhedges() == 0
    }

    fn intersects(&self, other: &BitVec) -> bool {
        other.nhedges() == 0
    }
}

impl Inclusion<Range<Hedge>> for Empty {
    fn includes(&self, other: &Range<Hedge>) -> bool {
        other.start >= other.end
    }

    fn intersects(&self, other: &Range<Hedge>) -> bool {
        other.start >= other.end
    }
}

impl Inclusion<RangeTo<Hedge>> for Empty {
    fn includes(&self, _other: &RangeTo<Hedge>) -> bool {
        false
    }

    fn intersects(&self, _other: &RangeTo<Hedge>) -> bool {
        false
    }
}

impl Inclusion<RangeToInclusive<Hedge>> for Empty {
    fn includes(&self, _other: &RangeToInclusive<Hedge>) -> bool {
        false
    }

    fn intersects(&self, _other: &RangeToInclusive<Hedge>) -> bool {
        false
    }
}

impl Inclusion<RangeFrom<Hedge>> for Empty {
    fn includes(&self, _other: &RangeFrom<Hedge>) -> bool {
        false
    }

    fn intersects(&self, _other: &RangeFrom<Hedge>) -> bool {
        false
    }
}

impl Inclusion<RangeInclusive<Hedge>> for Empty {
    fn includes(&self, other: &RangeInclusive<Hedge>) -> bool {
        other.start() > other.end()
    }

    fn intersects(&self, other: &RangeInclusive<Hedge>) -> bool {
        other.start() > other.end()
    }
}

/// A generic iterator that always yields `None`.
///
/// This is used as the `BaseIter` for the [`Empty`] subgraph type, as an empty
/// subgraph contains no hedges to iterate over.
///
/// # Type Parameters
/// - `T`: The type of item the iterator would theoretically yield (though it never does).
pub struct EmptyIter<T> {
    data: PhantomData<T>,
    // life: PhantomData<&'a ()>, // No lifetime needed if it holds no references
}

impl<T> EmptyIter<T> {
    pub fn new() -> Self {
        Self { data: PhantomData }
    }
}

impl<T> Default for EmptyIter<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Iterator for EmptyIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl SubGraph for Empty {
    type Base = Empty;
    type BaseIter<'a> = EmptyIter<Hedge>;
    fn nedges<E, V, H, N: NodeStorageOps<NodeData = V>>(
        &self,
        _graph: &HedgeGraph<E, V, H, N>,
    ) -> usize {
        self.nhedges() / 2
    }

    fn has_greater(&self, _hedge: Hedge) -> bool {
        false
    }

    fn size(&self) -> usize {
        self.size
    }

    fn join_mut(&mut self, other: Self) {
        self.size += other.size
    }

    fn included_iter(&self) -> Self::BaseIter<'_> {
        EmptyIter::new()
    }

    fn nhedges(&self) -> usize {
        0
    }

    fn empty(size: usize) -> Self {
        Empty { size }
    }

    fn included(&self) -> &Empty {
        self
    }

    fn string_label(&self) -> String {
        "∅".into()
    }
    fn is_empty(&self) -> bool {
        true
    }
}
