use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

use bitvec::vec::BitVec;

use crate::half_edge::{involution::Hedge, nodestore::NodeStorageOps, HedgeGraph};

use super::{Inclusion, SubGraph};

#[derive(Clone, Debug, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct FullOrEmpty {
    size: isize,
}

impl FullOrEmpty {
    pub fn full(size: usize) -> FullOrEmpty {
        FullOrEmpty {
            size: size as isize,
        }
    }
    pub fn empty(size: usize) -> FullOrEmpty {
        FullOrEmpty {
            size: -(size as isize),
        }
    }
}

impl Inclusion<Hedge> for FullOrEmpty {
    fn includes(&self, hedge_id: &Hedge) -> bool {
        (hedge_id.0 as isize) < self.size
    }

    fn intersects(&self, other: &Hedge) -> bool {
        (other.0 as isize) < self.size
    }
}

impl Inclusion<FullOrEmpty> for FullOrEmpty {
    fn includes(&self, other: &FullOrEmpty) -> bool {
        // true
        self.size == other.size
    }

    fn intersects(&self, other: &FullOrEmpty) -> bool {
        // true
        self.size == other.size
    }
}

impl Inclusion<BitVec> for FullOrEmpty {
    fn includes(&self, other: &BitVec) -> bool {
        self.size == other.size() as isize
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.size == other.size() as isize
    }
}

impl Inclusion<Range<Hedge>> for FullOrEmpty {
    fn includes(&self, other: &Range<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size >= other.end.0
        } else {
            other.start >= other.end
        }
    }

    fn intersects(&self, other: &Range<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size >= other.end.0
        } else {
            other.start >= other.end
        }
    }
}

impl Inclusion<RangeTo<Hedge>> for FullOrEmpty {
    fn includes(&self, other: &RangeTo<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size >= other.end.0
        } else {
            false
        }
    }

    fn intersects(&self, other: &RangeTo<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size >= other.end.0
        } else {
            false
        }
    }
}

impl Inclusion<RangeToInclusive<Hedge>> for FullOrEmpty {
    fn includes(&self, other: &RangeToInclusive<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.end.0
        } else {
            false
        }
    }

    fn intersects(&self, other: &RangeToInclusive<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.end.0
        } else {
            false
        }
    }
}

impl Inclusion<RangeFrom<Hedge>> for FullOrEmpty {
    fn includes(&self, other: &RangeFrom<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.start.0
        } else {
            false
        }
    }

    fn intersects(&self, other: &RangeFrom<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.start.0
        } else {
            false
        }
    }
}

impl Inclusion<RangeInclusive<Hedge>> for FullOrEmpty {
    fn includes(&self, other: &RangeInclusive<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.end().0
        } else {
            other.start() > other.end()
        }
    }

    fn intersects(&self, other: &RangeInclusive<Hedge>) -> bool {
        if let Ok(size) = usize::try_from(self.size) {
            size > other.end().0
        } else {
            other.start() > other.end()
        }
    }
}

pub struct RangeHedgeIter {
    iter: Range<usize>,
}

impl From<Range<Hedge>> for RangeHedgeIter {
    fn from(value: Range<Hedge>) -> Self {
        RangeHedgeIter {
            iter: value.start.0..value.end.0,
        }
    }
}

impl Iterator for RangeHedgeIter {
    type Item = Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(Hedge)
    }
}

impl SubGraph for FullOrEmpty {
    type Base = FullOrEmpty;
    type BaseIter<'a> = RangeHedgeIter;
    fn nedges<E, V, N: NodeStorageOps<NodeData = V>>(&self, graph: &HedgeGraph<E, V, N>) -> usize {
        let mut count = 0;
        for i in self.included_iter() {
            if i != graph.inv(i) && self.includes(&graph.inv(i)) {
                count += 1;
            }
        }
        count / 2
    }

    fn has_greater(&self, _hedge: Hedge) -> bool {
        false
    }

    fn size(&self) -> usize {
        self.size.unsigned_abs()
    }

    fn join_mut(&mut self, other: Self) {
        if self.is_empty() && other.is_empty() {
            self.size -= other.size
        } else if !self.is_empty() && !other.is_empty() {
            self.size += other.size
        }
    }

    fn included_iter(&self) -> Self::BaseIter<'_> {
        if self.is_empty() {
            (Hedge(1)..Hedge(0)).into()
        } else {
            (Hedge(0)..Hedge(self.size as usize)).into()
        }
    }

    fn nhedges(&self) -> usize {
        self.size.try_into().unwrap_or(0)
    }

    fn empty(size: usize) -> Self {
        FullOrEmpty {
            size: -(size as isize),
        }
    }

    fn included(&self) -> &FullOrEmpty {
        self
    }

    fn string_label(&self) -> String {
        if self.is_empty() {
            "∅".into()
        } else {
            "full".into()
        }
    }
    fn is_empty(&self) -> bool {
        self.size <= 0
    }
}
