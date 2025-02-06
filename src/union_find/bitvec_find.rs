use bitvec::vec::BitVec;

use super::{ParentPointer, UnionFind};

/// Derived Structure: UnionFindBitFilter
///
/// In this derived structure the associated data is a BitVec filter for the set’s contents.
/// Each element’s filter is initially a BitVec (from the bitvec crate) of length n (the number of elements)
/// with only its own bit set. When two sets merge the filters are merged via bitwise OR.
pub struct UnionFindBitFilter<T> {
    inner: UnionFind<T, BitVec>,
}

impl<T> UnionFindBitFilter<T> {
    /// Creates a new UnionFindBitFilter from a vector of base elements.
    /// Each element is initially in its own set and its filter is a BitVec of length n with only its own bit set.
    pub fn new(elements: Vec<T>) -> Self {
        let n = elements.len();
        let associated: Vec<BitVec> = (0..n)
            .map(|i| {
                let mut bv = BitVec::repeat(false, n);
                bv.set(i, true);
                bv
            })
            .collect();
        let inner = UnionFind::new(elements, associated);
        Self { inner }
    }

    /// Returns the number of base elements.
    pub fn len(&self) -> usize {
        self.inner.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.elements.is_empty()
    }

    /// Finds the representative of the set containing the element at ParentPointer x.
    pub fn find(&self, x: ParentPointer) -> ParentPointer {
        self.inner.find(x)
    }

    /// Returns a reference to the BitVec filter for the set containing the element at ParentPointer x.
    pub fn find_filter(&self, x: ParentPointer) -> &BitVec {
        self.inner.find_data(x)
    }

    /// Unions the sets containing x and y, merging their BitVec filters via bitwise OR.
    pub fn union(&mut self, x: ParentPointer, y: ParentPointer) -> ParentPointer {
        self.inner.union(x, y, |a, b| a | b)
    }

    /// Returns the base elements belonging to the set containing the element at ParentPointer x,
    /// as determined by the BitVec filter.
    pub fn elements_in_set(&self, x: ParentPointer) -> Vec<&T> {
        let filter = self.find_filter(x);
        filter
            .iter_ones()
            .map(|i| &self.inner.elements[i])
            .collect()
    }
}
