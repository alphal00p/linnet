use std::{collections::BTreeMap, fmt, ops::Index};

use crate::half_edge::{
    involution::Hedge,
    subgraph::{BaseSubgraph, InternalSubGraph, SubGraph, SubGraphOps},
    HedgeGraph, NodeIndex,
};
use ahash::AHashSet;
use bitvec::vec::BitVec;
use thiserror::Error;

use crate::half_edge::involution::Flow;

/// A permutation of `0..n`, with the ability to apply itself (or its inverse) to slices.
///
/// # Examples
///
/// ```
/// use linnet::permutation::Permutation;
///
/// // Create a permutation that maps 0->2, 1->0, 2->1, 3->3
/// let p = Permutation::from_map(vec![2, 0, 1, 3]);
///
/// // Apply the permutation to a slice
/// let data = vec![10, 20, 30, 40];
/// let permuted = p.apply_slice(&data);
/// assert_eq!(permuted, vec![30, 10, 20, 40]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Permutation {
    map: Vec<usize>,
    inv: Vec<usize>,
}

/// Implement ordering comparisons for permutations based on their `map` field.
impl PartialOrd for Permutation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.map.partial_cmp(&other.map)
    }
}

impl Permutation {
    // --------------------------------------------------------------------------------------------
    // Basic Constructors and Accessors
    // --------------------------------------------------------------------------------------------

    /// Creates the identity permutation of length `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::id(4);
    /// assert_eq!(p.apply_slice(&[10,20,30,40]), vec![10,20,30,40]);
    /// ```
    pub fn id(n: usize) -> Self {
        Permutation {
            map: (0..n).collect(),
            inv: (0..n).collect(),
        }
    }

    /// Creates a permutation from a mapping vector.
    /// The `map` vector states where index `i` is sent: `map[i]` is the image of `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.apply_slice(&[10,20,30]), vec![30,10,20]);
    /// ```
    pub fn from_map(map: Vec<usize>) -> Self {
        let mut inv = vec![0; map.len()];
        for (i, &j) in map.iter().enumerate() {
            inv[j] = i;
        }
        Permutation { map, inv }
    }

    /// Returns the internal mapping as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.map(), &[2, 0, 1]);
    /// ```
    // -- ADDED
    pub fn map(&self) -> &[usize] {
        &self.map
    }

    /// Returns the inverse mapping as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// assert_eq!(p.inv(), &[1, 2, 0]);
    /// ```
    // -- ADDED
    pub fn inv(&self) -> &[usize] {
        &self.inv
    }

    // --------------------------------------------------------------------------------------------
    // Basic Operations
    // --------------------------------------------------------------------------------------------

    /// Returns the inverse of the permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let inv = p.inverse();
    /// assert_eq!(inv.apply_slice(&[10,20,30]), vec![20,30,10]);
    /// ```
    pub fn inverse(&self) -> Self {
        Permutation {
            map: self.inv.clone(),
            inv: self.map.clone(),
        }
    }

    /// Applies `self` to a slice, returning a new `Vec<T>` in permuted order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let data = vec![10, 20, 30];
    /// assert_eq!(p.apply_slice(&data), vec![30, 10, 20]);
    /// ```
    pub fn apply_slice<T: Clone, S>(&self, slice: S) -> Vec<T>
    where
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        self.map.iter().map(|&idx| s[idx].clone()).collect()
    }

    /// Applies the inverse of `self` to a slice, returning a new `Vec<T>` in permuted order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let data = vec![10, 20, 30];
    /// assert_eq!(p.apply_slice_inv(&data), vec![20, 30, 10]);
    /// ```
    pub fn apply_slice_inv<T: Clone, S>(&self, slice: S) -> Vec<T>
    where
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        self.inv.iter().map(|&idx| s[idx].clone()).collect()
    }

    /// Applies `self` in-place to the provided slice by using transpositions
    /// derived from the cycle decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let mut data = vec![10, 20, 30];
    /// p.apply_slice_in_place(&mut data);
    /// assert_eq!(data, vec![20, 30, 10]);
    /// ```
    pub fn apply_slice_in_place<T: Clone, S>(&self, slice: &mut S)
    where
        S: AsMut<[T]>,
    {
        let transpositions = self.transpositions();
        for (i, j) in transpositions.iter().rev() {
            slice.as_mut().swap(*i, *j);
        }
    }

    /// Applies the inverse of `self` in-place to the provided slice by using transpositions
    /// derived from the cycle decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let mut data = vec![10, 20, 30];
    /// p.apply_slice_in_place_inv(&mut data);
    /// assert_eq!(data, vec![30, 10, 20]);
    /// ```
    pub fn apply_slice_in_place_inv<T, S>(&self, slice: &mut S)
    where
        S: AsMut<[T]>,
    {
        let transpositions = self.transpositions();
        for (i, j) in transpositions {
            slice.as_mut().swap(i, j);
        }
    }

    /// Composes `self` with another permutation `other`, returning a new permutation:
    /// `(self ◦ other)(i) = self.map[other.map[i]]`.
    pub fn compose(&self, other: &Self) -> Self {
        let map = other.map.iter().map(|&i| self.map[i]).collect();
        Self::from_map(map)
    }

    // --------------------------------------------------------------------------------------------
    // Sorting Utilities
    // --------------------------------------------------------------------------------------------

    /// Given a slice of items that implement `Ord`, returns the permutation that sorts them
    /// in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let data = vec![30, 10, 20, 40];
    /// let perm = Permutation::sort(&data);
    /// // perm.map should be [1, 2, 0, 3]
    /// assert_eq!(perm.apply_slice(&data), vec![10, 20, 30, 40]);
    /// ```
    pub fn sort<T, S>(slice: S) -> Permutation
    where
        T: Ord,
        S: AsRef<[T]>,
    {
        let s = slice.as_ref();
        let mut permutation: Vec<usize> = (0..s.len()).collect();
        permutation.sort_by_key(|&i| &s[i]);
        Self::from_map(permutation)
    }

    // --------------------------------------------------------------------------------------------
    // Cycles and Transpositions
    // --------------------------------------------------------------------------------------------

    /// Returns the cycle decomposition of `self` as a `Vec` of cycles,
    /// each cycle represented as a `Vec<usize>`.
    /// Each cycle lists the indices of a single cycle, e.g. `[0, 2, 1]` means `0->2, 2->1, 1->0`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1, 3]);
    /// let cycles = p.find_cycles();
    /// // cycles might be [[0, 2, 1], [3]]
    /// assert_eq!(cycles.len(), 2);
    /// ```
    pub fn find_cycles(&self) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.map.len()];
        let mut cycles = Vec::new();
        for i in 0..self.map.len() {
            if visited[i] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                cycle.push(j);
                j = self.map[j];
            }
            if !cycle.is_empty() {
                cycles.push(cycle);
            }
        }
        cycles
    }

    /// Converts a single cycle to a list of transpositions that produce that cycle.
    /// This is a helper method and typically not used directly.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let cycle = vec![0, 2, 1];
    /// let transpositions = Permutation::cycle_to_transpositions(&cycle);
    /// // cycle 0->2,2->1,1->0 can be built from swaps (0,1) and (0,2)
    /// assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    /// ```
    pub fn cycle_to_transpositions(cycle: &[usize]) -> Vec<(usize, usize)> {
        let mut transpositions = Vec::new();
        for i in (1..cycle.len()).rev() {
            transpositions.push((cycle[0], cycle[i]));
        }
        transpositions
    }

    /// Returns the list of transpositions for `self`, by decomposing it into cycles
    /// and then converting each cycle to transpositions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 0, 1]);
    /// let transpositions = p.transpositions();
    /// assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    /// ```
    pub fn transpositions(&self) -> Vec<(usize, usize)> {
        let cycles = self.find_cycles();
        let mut transpositions = Vec::new();
        for cycle in cycles {
            transpositions.extend(Self::cycle_to_transpositions(&cycle));
        }
        transpositions
    }

    // --------------------------------------------------------------------------------------------
    // Myrvold & Ruskey Ranking/Unranking
    // --------------------------------------------------------------------------------------------

    /// Computes the rank of the permutation in the Myrvold & Ruskey "Rank1" ordering.
    ///
    /// This is a recursive implementation. For permutations of size `n`, it removes
    /// the position of `n-1` from the permutation, multiplies the result by `n`,
    /// and adds the index of `n-1`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 1, 3, 0]);
    /// assert_eq!(p.myrvold_ruskey_rank1(), 12);
    /// ```
    pub fn myrvold_ruskey_rank1(mut self) -> usize {
        let n = self.map.len();
        if self.map.len() == 1 {
            return 0;
        }

        let s = self.map[n - 1];
        self.map.swap_remove(self.inv[n - 1]);
        self.inv.swap_remove(s);

        s + n * self.myrvold_ruskey_rank1()
    }

    /// Unranks a permutation of size `n` from its Myrvold & Ruskey "Rank1" index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::myrvold_ruskey_unrank1(4, 12);
    /// assert_eq!(p.map(), &[2, 1, 3, 0]);
    /// ```
    pub fn myrvold_ruskey_unrank1(n: usize, mut rank: usize) -> Self {
        let mut p = (0..n).collect::<Vec<_>>();
        for i in (1..=n).rev() {
            let j = rank % i;
            rank /= i;
            p.swap(i - 1, j);
        }
        Permutation::from_map(p)
    }

    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    /// Computes the rank of the permutation in the Myrvold & Ruskey "Rank2" ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![2, 1, 3, 0]);
    /// // Suppose it has rank = R. We can test we get p back by unranking R.
    /// let rank = p.clone().myrvold_ruskey_rank2();
    /// let q = Permutation::myrvold_ruskey_unrank2(4, rank);
    /// assert_eq!(q, p);
    /// ```
    pub fn myrvold_ruskey_rank2(mut self) -> usize {
        let n = self.map.len();
        if n == 1 {
            return 0;
        }
        let s = self.map[n - 1];
        self.map.swap_remove(self.inv[n - 1]);
        self.inv.swap_remove(s);
        s * Self::factorial(n - 1) + self.myrvold_ruskey_rank2()
    }

    /// Unranks a permutation of size `n` from its Myrvold & Ruskey "Rank2" index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::myrvold_ruskey_unrank2(4, 1);
    /// assert_eq!(p.map(), &[2, 1, 3, 0]);
    /// ```
    pub fn myrvold_ruskey_unrank2(n: usize, mut rank: usize) -> Self {
        let mut p = (0..n).collect::<Vec<_>>();
        for i in (1..=n).rev() {
            let s = rank / (Self::factorial(i - 1));
            p.swap(i - 1, s);
            rank %= Self::factorial(i - 1);
        }
        Permutation::from_map(p)
    }

    // --------------------------------------------------------------------------------------------
    // Additional Suggested Methods
    // --------------------------------------------------------------------------------------------

    /// Checks if this permutation is the identity permutation (i.e., does nothing).
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::id(4);
    /// assert!(p.is_identity());
    ///
    /// let q = Permutation::from_map(vec![1,0,2,3]);
    /// assert!(!q.is_identity());
    /// ```
    // -- ADDED
    pub fn is_identity(&self) -> bool {
        self.map.iter().enumerate().all(|(i, &m)| i == m)
    }

    /// Returns the sign (+1 or -1) of the permutation,
    /// indicating whether it is an even (+1) or odd (-1) permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![1,0,3,2]);
    /// assert_eq!(p.sign(), 1); // even
    ///
    /// let q = Permutation::from_map(vec![2,1,0]);
    /// assert_eq!(q.sign(), -1); // odd
    /// ```
    // -- ADDED
    pub fn sign(&self) -> i8 {
        // Count inversions or use transpositions
        let mut sign = 1i8;
        for cycle in self.find_cycles() {
            // Each cycle of length k contributes (k-1) to the total parity
            let k = cycle.len();
            if k > 1 && (k - 1) % 2 == 1 {
                sign = -sign;
            }
        }
        sign
    }

    /// Computes the k-th power of the permutation (composition with itself k times).
    /// For k = 0, it returns the identity of the same size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let p = Permutation::from_map(vec![1, 2, 0]);
    /// // p^2 maps 0->p(1)=2, 1->p(2)=0, 2->p(0)=1 => [2,0,1]
    /// let p2 = p.pow(2);
    /// assert_eq!(p2.map(), &[2, 0, 1]);
    /// ```
    // -- ADDED
    pub fn pow(&self, k: usize) -> Self {
        if k == 0 {
            return Permutation::id(self.map.len());
        }
        let mut result = Permutation::id(self.map.len());
        let mut base = self.clone();
        let mut exp = k;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.compose(&base);
            }
            base = base.compose(&base);
            exp /= 2;
        }
        result
    }
}

/// Specifies the direction for reading cycle compositions
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum CycleOrder {
    /// Apply rightmost cycles first: (a b)(c d) applies (c d) then (a b)
    LastFirst,
    /// Apply leftmost cycles first: (a b)(c d) applies (a b) then (c d)
    FirstFirst,
}

impl fmt::Display for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // First show cycle notation
        let cycles = self.find_cycles();
        let mut first = true;
        for cycle in cycles {
            if cycle.len() > 1 {
                // Only show non-trivial cycles
                if !first {
                    write!(f, " ")?;
                }
                write!(f, "(")?;
                for (i, &x) in cycle.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", x)?;
                }
                write!(f, ")")?;
                first = false;
            }
        }
        if first {
            // If no cycles were printed (identity permutation)
            write!(f, "()")?;
        }

        // Then show one-line notation
        write!(f, " [")?;
        for (i, &x) in self.map.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", x)?;
        }
        write!(f, "]")
    }
}

impl Permutation {
    /// Creates a permutation from a set of disjoint cycles.
    /// Each cycle should be a Vec<usize> representing indices in the cycle.
    /// The cycles must be disjoint (no shared elements).
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let cycles = vec![vec![0, 1, 2], vec![3, 4]];
    /// let p = Permutation::from_disjoint_cycles(&cycles).unwrap();
    /// assert_eq!(p.map(), &[1, 2, 0, 4, 3]);
    ///
    /// // Error if cycles are not disjoint
    /// let invalid = vec![vec![0, 1], vec![1, 2]];
    /// assert!(Permutation::from_disjoint_cycles(&invalid).is_err());
    /// ```
    pub fn from_disjoint_cycles(cycles: &[Vec<usize>]) -> Result<Self, String> {
        // Find the size of the permutation (maximum index + 1)
        let n = cycles
            .iter()
            .flat_map(|cycle| cycle.iter())
            .max()
            .map(|&max| max + 1)
            .unwrap_or(0);

        // Check for duplicates across cycles
        let mut seen = vec![false; n];
        for cycle in cycles {
            for &idx in cycle {
                if seen[idx] {
                    return Err("Cycles are not disjoint".to_string());
                }
                seen[idx] = true;
            }
        }

        // Create the permutation map
        let mut map = (0..n).collect::<Vec<_>>();
        for cycle in cycles {
            if cycle.len() <= 1 {
                continue;
            }

            // Map each element to the next element in the cycle
            for i in 0..cycle.len() {
                let from = cycle[i];
                let to = cycle[(i + 1) % cycle.len()];
                map[from] = to;
            }
        }

        Ok(Permutation::from_map(map))
    }
    /// Creates a permutation from any set of cycles with specified reading order.
    pub fn from_cycles_ordered(cycles: &[Vec<usize>], order: CycleOrder) -> Self {
        // Find the size of the permutation
        let n = cycles
            .iter()
            .flat_map(|cycle| cycle.iter())
            .max()
            .map(|&max| max + 1)
            .unwrap_or(0);

        // Start with identity permutation
        let mut result = Permutation::id(n);

        // Choose iteration order based on CycleOrder
        let cycle_iter: Box<dyn Iterator<Item = &Vec<usize>>> = match order {
            CycleOrder::LastFirst => Box::new(cycles.iter().rev()),
            CycleOrder::FirstFirst => Box::new(cycles.iter()),
        };

        // Apply cycles in specified order
        for cycle in cycle_iter {
            if cycle.len() <= 1 {
                continue;
            }

            // Create a single cycle permutation
            let mut cycle_map = (0..n).collect::<Vec<_>>();
            for i in 0..cycle.len() {
                let from = cycle[i];
                let to = cycle[(i + 1) % cycle.len()];
                cycle_map[from] = to;
            }
            let cycle_perm = Permutation::from_map(cycle_map);
            result = cycle_perm.compose(&result);
        }

        result
    }

    /// Creates a permutation from cycles using right-to-left reading order (default).
    /// Equivalent to `from_cycles_ordered(cycles, CycleOrder::RightToLeft)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linnet::permutation::Permutation;
    /// let cycles = vec![vec![0, 1, 2], vec![1, 2]];
    /// let p = Permutation::from_cycles(&cycles);
    /// assert_eq!(p.map(), &[1, 0, 2]);
    /// ```
    pub fn from_cycles(cycles: &[Vec<usize>]) -> Self {
        Self::from_cycles_ordered(cycles, CycleOrder::LastFirst)
    }

    pub fn length(&self) -> usize {
        self.map.len()
    }

    pub fn generate_all(generators: &[Permutation]) -> Result<Vec<Permutation>, PermutationError> {
        let size = if let Some(generator) = generators.first() {
            generator.length()
        } else {
            return Err(PermutationError::EmptyGenerators);
        };
        let mut all = AHashSet::new();
        let mut stack = vec![Permutation::id(size)];
        for g in generators {
            if g.length() != size {
                return Err(PermutationError::InvalidGeneratorLength);
            }
        }

        while let Some(current) = stack.pop() {
            for g in generators {
                let next = current.compose(g);
                if all.insert(next.clone()) {
                    stack.push(next);
                }
            }
        }

        Ok(all.drain().collect())
    }
}

#[derive(Error, Debug)]
pub enum PermutationError {
    #[error("Invalid generator length")]
    InvalidGeneratorLength,

    #[error("Empty generators")]
    EmptyGenerators,
}

pub trait HedgeGraphExt {
    fn hedges_between(&self, a: NodeIndex, b: NodeIndex) -> BitVec;

    fn permute_subgraph<S: SubGraph>(&self, subgraph: &S, hedge_perm: &Permutation) -> BitVec;

    fn orientation_ord(&self, hedge: Hedge) -> u8;
}

pub trait PermutationExt<Orderer: Ord = ()> {
    // fn from_disjoint_cycles(cycles: &[Vec<usize>]) -> Result<Self, String>;
    //
    type Output;
    type Edges;

    fn permute_vertices(
        &self,
        perm: &Permutation,
        ord: &impl Fn(&Self::Edges) -> Orderer,
    ) -> Vec<Self::Output>;

    fn sort_by(&self, hedge: Hedge, ord: &impl Fn(&Self::Edges) -> Orderer) -> impl Ord;

    fn sort_by_perm(
        &self,
        hedge: Hedge,
        perm: &Permutation,
        ord: &impl Fn(&Self::Edges) -> Orderer,
    ) -> impl Ord;
}

impl<E, V> HedgeGraphExt for HedgeGraph<E, V> {
    fn hedges_between(&self, a: NodeIndex, b: NodeIndex) -> BitVec {
        let a_ext = InternalSubGraph::cleaned_filter_optimist(
            BitVec::from_hedge_iter(self.hair_iter(a), self.n_hedges()),
            self,
        )
        .filter;
        let b_ext = InternalSubGraph::cleaned_filter_optimist(
            BitVec::from_hedge_iter(self.hair_iter(b), self.n_hedges()),
            self,
        )
        .filter;
        a_ext.intersection(&b_ext)
    }

    fn permute_subgraph<S: SubGraph>(&self, subgraph: &S, hedge_perm: &Permutation) -> BitVec {
        let mut permuted_subgraph: BitVec = self.empty_subgraph();

        for h in subgraph.included_iter() {
            permuted_subgraph.set(hedge_perm[h.0], true);
        }
        permuted_subgraph
    }

    fn orientation_ord(&self, hedge: Hedge) -> u8 {
        match self.superficial_hedge_orientation(hedge) {
            Some(Flow::Sink) => 1,
            Some(Flow::Source) => 2,
            None => 0,
        }
    }
}

impl<E, V, O: Ord> PermutationExt<O> for HedgeGraph<E, V> {
    type Output = Permutation;
    type Edges = E;

    fn permute_vertices(
        &self,
        perm: &Permutation,
        ord: &impl Fn(&Self::Edges) -> O,
    ) -> Vec<Self::Output> {
        // fn generator_pair(shift: usize, n: usize, total_size: usize) -> [Permutation; 2] {
        //     let mut map = (0..total_size).collect::<Vec<_>>();
        //     for i in 0..n {
        //         map[i + shift] = (i + 1) % n;
        //     }
        //     [
        //         Permutation::from_map(map),
        //         Permutation::from_cycles(&[vec![shift, shift + n - 1], vec![total_size]]),
        //     ]
        // }
        //
        fn generator_pair(n: usize) -> [Permutation; 2] {
            let mut map = Vec::new();
            for i in 0..n {
                map.push((i + 1) % n)
            }
            [
                Permutation::from_map(map),
                Permutation::from_cycles(&[vec![0, n - 1]]),
            ]
        }
        let transpositions = perm.map();
        let mut perms = Vec::new();

        let n = self.n_hedges();
        let mut map = (0..n).collect::<Vec<_>>();

        for (from, &to) in transpositions.iter().enumerate() {
            let from_hairs = self.hair_iter(NodeIndex(from));
            let mut from_hedges = from_hairs.collect::<Vec<_>>();
            let to_hairs = self.hair_iter(NodeIndex(to));
            let mut to_hedges = to_hairs.collect::<Vec<_>>();

            let mut to = BTreeMap::new();

            to_hedges.iter().for_each(|&hedge| {
                to.entry(self.sort_by_perm(hedge, perm, ord))
                    .or_insert_with(Vec::new)
                    .push(hedge);
            });
            to_hedges = to
                .values()
                .flat_map(|values| {
                    if values.len() > 1
                        && self.node_id(self.inv(values[0])).0 <= self.node_id(values[0]).0
                    //insure no double counting of inter-edge permutation
                    {
                        let gen_pair = generator_pair(values.len());

                        let cycle = gen_pair[0].apply_slice(values);
                        let trans = gen_pair[1].apply_slice(values);

                        // println!("{values:?}{cycle:?}{trans:?}");

                        let mut cycle_map = (0..n).collect::<Vec<_>>();
                        let mut trans_map = (0..n).collect::<Vec<_>>();
                        for i in 0..values.len() {
                            cycle_map[cycle[i].0] = values[i].0;
                            trans_map[trans[i].0] = values[i].0;
                        }

                        perms.push(Permutation::from_map(cycle_map));
                        perms.push(Permutation::from_map(trans_map));
                    }
                    values
                })
                .cloned()
                .collect();

            // let mut to = BTreeMap::new();

            // to_hedges.iter().for_each(|&hedge| {
            //     to.entry(self.sort_by(hedge, ord))
            //         .or_insert_with(Vec::new)
            //         .push(hedge);
            // });

            from_hedges.sort_by(|a, b| self.sort_by(*a, ord).cmp(&self.sort_by(*b, ord)));

            // to_hedges.sort_by(|a, b| {
            //     self.sort_by_perm(*a, perm, ord)
            //         .cmp(&self.sort_by_perm(*b, perm, ord))
            // });

            // println!("from:{from_hedges:?}");

            for (from_hedge, to_hedge) in from_hedges.iter().zip(to_hedges.iter()) {
                // println!("{from_hedge}->{to_hedge}");
                map[from_hedge.0] = to_hedge.0;
            }
        }
        let mut maps = vec![];
        let map = Permutation::from_map(map);
        // println!("Map:{map}");
        match Permutation::generate_all(&perms) {
            Ok(a) => {
                for p in a {
                    // println!("Permutation:{p}");
                    // println!("Composition:{}", map.compose(&p));
                    maps.push(map.compose(&p));
                }
            }
            Err(PermutationError::EmptyGenerators) => maps.push(map),
            Err(e) => panic!("Error generating permutations: {}", e),
        }
        maps
    }

    fn sort_by(&self, hedge: Hedge, ord: &impl Fn(&Self::Edges) -> O) -> impl Ord {
        (
            self.involved_node_id(hedge)
                .unwrap_or(self.node_id(hedge))
                .0,
            self.orientation_ord(hedge),
            ord(self.get_edge_data(hedge)),
        )
    }

    fn sort_by_perm(
        &self,
        hedge: Hedge,
        perm: &Permutation,
        ord: &impl Fn(&Self::Edges) -> O,
    ) -> impl Ord {
        (
            perm.inv()[self
                .involved_node_id(hedge)
                .unwrap_or(self.node_id(hedge))
                .0],
            self.orientation_ord(hedge),
            ord(self.get_edge_data(hedge)),
        )
    }
}

impl Index<usize> for Permutation {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.map()[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::half_edge::{builder::HedgeGraphBuilder, involution::Flow};

    use super::*;

    #[test]
    fn test_from_disjoint_cycles() {
        // Test disjoint cycles
        let cycles = vec![vec![0, 3, 2], vec![1, 4]];
        let p = Permutation::from_disjoint_cycles(&cycles).unwrap();
        assert_eq!(p.map(), &[3, 4, 0, 2, 1]);

        // Test single cycle
        let cycles = vec![vec![0, 1, 2]];
        let p = Permutation::from_disjoint_cycles(&cycles).unwrap();
        assert_eq!(p.map(), &[1, 2, 0]);

        // Test non-disjoint cycles
        let cycles = vec![vec![0, 1], vec![1, 2]];
        assert!(Permutation::from_disjoint_cycles(&cycles).is_err());

        // Test empty cycles
        let cycles: Vec<Vec<usize>> = vec![];
        let p = Permutation::from_disjoint_cycles(&cycles).unwrap();
        assert!(p.map().is_empty());

        // Test single element cycles
        let cycles = vec![vec![0]];
        let p = Permutation::from_disjoint_cycles(&cycles).unwrap();
        assert_eq!(p.map(), &[0]);
    }

    #[test]
    fn test_from_cycles_ordered() {
        let cycles = vec![vec![0, 1, 2], vec![1, 2]];

        // Right to left: (0 1 2)(1 2)
        // First (1 2): [0 2 1]
        // Then (0 1 2): [1 0 2]
        let p = Permutation::from_cycles_ordered(&cycles, CycleOrder::LastFirst);
        assert_eq!(p.map(), &[1, 0, 2]);

        // Left to right: (1 2)(0 1 2)
        // First (0 1 2): [1 2 0]
        // Then (1 2): [0 2 1]
        let p = Permutation::from_cycles_ordered(&cycles, CycleOrder::FirstFirst);
        assert_eq!(p.map(), &[2, 1, 0]);
    }

    #[test]
    fn test_cycle_composition_both_orders() {
        // Test (1 2)(0 1)
        let cycles = vec![vec![1, 2], vec![0, 1]];

        // Right to left: apply (0 1) then (1 2)
        let p = Permutation::from_cycles_ordered(&cycles, CycleOrder::LastFirst);
        assert_eq!(p.map(), &[2, 0, 1]);

        // Left to right: apply (1 2) then (0 1)
        let p = Permutation::from_cycles_ordered(&cycles, CycleOrder::FirstFirst);
        assert_eq!(p.map(), &[1, 2, 0]);
    }

    #[test]
    fn test_default_order() {
        let cycles = vec![vec![0, 1, 2], vec![1, 2]];
        let p1 = Permutation::from_cycles(&cycles);
        let p2 = Permutation::from_cycles_ordered(&cycles, CycleOrder::LastFirst);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_disjoint_cycles_order_invariant() {
        let cycles = vec![vec![0, 1], vec![2, 3]];
        let p1 = Permutation::from_cycles_ordered(&cycles, CycleOrder::LastFirst);
        let p2 = Permutation::from_cycles_ordered(&cycles, CycleOrder::FirstFirst);
        // Order shouldn't matter for disjoint cycles
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_from_cycles() {
        // Test single cycle
        let cycles = vec![vec![0, 1, 2]];
        let p = Permutation::from_cycles(&cycles);
        assert_eq!(p.map(), &[1, 2, 0]);

        // Test overlapping cycles
        let cycles = vec![vec![0, 1, 2], vec![1, 2]];
        let p = Permutation::from_cycles(&cycles);
        assert_eq!(p.map(), &[1, 0, 2]);

        // Test multiple disjoint cycles
        let cycles = vec![vec![0, 1], vec![2, 3]];
        let p = Permutation::from_cycles(&cycles);
        assert_eq!(p.map(), &[1, 0, 3, 2]);

        // Test composition order
        let cycles = vec![vec![0, 1], vec![1, 2]]; // (0 1)(1 2) = (0 1 2)
        let p = Permutation::from_cycles(&cycles);
        assert_eq!(p.map(), &[1, 2, 0]);
    }

    #[test]
    fn test_cycle_composition_properties() {
        // Test that (0 1 2)(1 2) = (0 1)()
        let p = Permutation::from_cycles(&[vec![0, 1, 2], vec![1, 2]]);
        let q = Permutation::from_cycles(&[vec![1, 0], vec![2]]);
        assert_eq!(p, q);

        // Test that (0 1)(1 2) = (0 1 2)
        let p = Permutation::from_cycles(&[vec![0, 1], vec![1, 2]]);
        let q = Permutation::from_cycles(&[vec![1, 2, 0]]);
        assert_eq!(p, q);
    }

    #[test]
    fn test_myrvold_ruskey_rank1() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        assert_eq!(p.myrvold_ruskey_rank1(), 12);
        for i in 0..=23 {
            assert_eq!(
                i,
                Permutation::myrvold_ruskey_rank1(Permutation::myrvold_ruskey_unrank1(4, i))
            );
        }
    }

    #[test]
    fn test_myrvold_ruskey_rank2() {
        let p = Permutation::myrvold_ruskey_unrank2(4, 1);
        assert_eq!(p.map, vec![2, 1, 3, 0]);
        for i in 0..=23 {
            assert_eq!(
                i,
                Permutation::myrvold_ruskey_rank2(Permutation::myrvold_ruskey_unrank2(4, i))
            );
        }
    }

    #[test]
    fn test_apply_slice() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        let data = vec![10, 20, 30, 40];
        let permuted = p.apply_slice(&data);
        assert_eq!(permuted, vec![30, 20, 40, 10]);
    }

    #[test]
    fn test_apply_slice_inv() {
        let p = Permutation::from_map(vec![2, 1, 3, 0]);
        let data = vec![10, 20, 30, 40];
        let permuted = p.apply_slice_inv(&data);
        assert_eq!(permuted, vec![40, 20, 10, 30]);
    }

    #[test]
    fn test_find_cycles() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let cycles = p.find_cycles();
        assert_eq!(cycles, vec![vec![0, 2, 1], vec![3]]);
    }

    #[test]
    fn test_cycle_to_transpositions() {
        let cycle = vec![0, 2, 1];
        let transpositions = Permutation::cycle_to_transpositions(&cycle);
        assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_transpositions() {
        let p = Permutation::sort([2, 1, 0, 3]);

        let transpositions = p.transpositions();
        println!("{:?}", transpositions.len());

        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let transpositions = p.transpositions();
        assert_eq!(transpositions, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_apply_slice_in_place() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let mut data = vec![10, 20, 30, 40];
        p.apply_slice_in_place(&mut data);
        assert_eq!(data, vec![20, 30, 10, 40]);
    }

    #[test]
    fn test_apply_slice_in_place_inv() {
        let p = Permutation::from_map(vec![2, 0, 1, 3]);
        let mut data = vec![10, 20, 30, 40];
        p.apply_slice_in_place_inv(&mut data);
        assert_eq!(data, vec![30, 10, 20, 40]);
    }

    #[test]
    fn test_sort() {
        let data = vec![30, 10, 20, 40];
        let perm = Permutation::sort(&data);
        assert_eq!(perm.map, vec![1, 2, 0, 3]);

        let sorted_data = perm.apply_slice(&data);
        assert_eq!(sorted_data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_sort_inverse() {
        let data = vec![30, 10, 20, 40];
        let perm = Permutation::sort(&data);
        let sorted_data = perm.apply_slice(&data);
        assert_eq!(sorted_data, vec![10, 20, 30, 40]);

        let inv_perm = perm.inverse();
        let original_data = inv_perm.apply_slice(&sorted_data);
        assert_eq!(original_data, data);
    }

    // -- ADDED TESTS

    #[test]
    fn test_is_identity() {
        let p = Permutation::id(5);
        assert!(p.is_identity());

        let q = Permutation::from_map(vec![1, 0, 2]);
        assert!(!q.is_identity());
    }

    #[test]
    fn test_sign() {
        // Even permutation
        let p = Permutation::from_map(vec![1, 0, 3, 2]);
        assert_eq!(p.sign(), 1);

        // Odd permutation
        let q = Permutation::from_map(vec![2, 1, 0]);
        assert_eq!(q.sign(), -1);
    }

    #[test]
    fn test_pow() {
        let p = Permutation::from_map(vec![1, 2, 0]);
        // p^1 = p
        assert_eq!(p.pow(1), p);

        // p^2 = 0->2,1->0,2->1 => [2,0,1]
        let p2 = p.pow(2);
        assert_eq!(p2.map(), &[2, 0, 1]);

        // p^3 = identity
        let p3 = p.pow(3);
        assert_eq!(p3, Permutation::id(3));
    }

    #[test]
    fn test_compose() {
        // p1: 0→1, 1→2, 2→0 (a cycle of length 3)
        let p1 = Permutation::from_map(vec![1, 2, 0]);

        // p2: 0→2, 1→0, 2→1 (the inverse of p1)
        let p2 = Permutation::from_map(vec![2, 0, 1]);

        // By definition in your code, compose(self, other) = x ↦ other(self(x)).
        //
        // So p2 ∘ p1 means apply p1 first, then p2. Since p2 is the inverse of p1,
        // their composition should be the identity permutation.
        let c1 = p1.compose(&p2);
        assert_eq!(c1, Permutation::id(3), "Expected p2 ∘ p1 = identity");

        // Likewise, p1 ∘ p2 = identity
        let c2 = p2.compose(&p1);
        assert_eq!(c2, Permutation::id(3), "Expected p1 ∘ p2 = identity");

        {
            let p1 = Permutation::from_map(vec![1, 0, 2]); // (0 1)
            let p2 = Permutation::from_map(vec![0, 2, 1]); // (1 2)

            let c1 = p1.compose(&p2); // (0 1)(1 2) = (0 1 2)
            let c2 = p2.compose(&p1); // (1 2)(0 1) = (0 2 1)

            assert_ne!(
                c1, c2,
                "Different order of composition should give different results"
            );
            assert_eq!(c1.map(), &[1, 2, 0]);
            assert_eq!(c2.map(), &[2, 0, 1]);
        }

        // Test associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)
        {
            let p1 = Permutation::from_map(vec![1, 2, 0]);
            let p2 = Permutation::from_map(vec![2, 0, 1]);
            let p3 = Permutation::from_map(vec![0, 2, 1]);

            let c1 = p1.compose(&p2).compose(&p3);
            let c2 = p1.compose(&p2.compose(&p3));

            assert_eq!(c1, c2, "Composition should be associative");
        }

        // Test composition of disjoint cycles
        {
            let p1 = Permutation::from_map(vec![1, 0, 2, 3]); // (0 1)
            let p2 = Permutation::from_map(vec![0, 1, 3, 2]); // (2 3)

            let c = p1.compose(&p2);
            assert_eq!(
                c.map(),
                &[1, 0, 3, 2],
                "Disjoint cycles should compose independently"
            );
        }

        // Test composition order with multiple elements
        {
            let p1 = Permutation::from_map(vec![1, 2, 3, 0]); // (0 1 2 3)
            let p2 = Permutation::from_map(vec![3, 2, 1, 0]); // (0 3)(1 2)

            let c = p1.compose(&p2);
            assert_eq!(
                c.map(),
                &[0, 3, 2, 1],
                "Complex composition should work correctly"
            );
        }
    }

    #[test]
    fn test_permute_graph() {
        let mut triangle = HedgeGraphBuilder::new();

        let a = triangle.add_node(());
        let b = triangle.add_node(());
        let c = triangle.add_node(());

        triangle.add_edge(a, b, (), false);
        triangle.add_edge(b, c, (), false);
        triangle.add_edge(c, a, (), false);

        let graph = triangle.build();
        let perm = Permutation::from_cycles(&[vec![0, 2], vec![1]]); //permutes a and c
        let perm2 = Permutation::from_cycles(&[vec![0, 1, 2]]); //permutes a and c

        let a_b_edge = graph.hedges_between(a, b);
        let b_c_edge = graph.hedges_between(b, c);
        let c_a_edge = graph.hedges_between(c, a);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let hedge_perm2 = graph.permute_vertices(&perm2, &|_| ());
        let permuted_b_c_edge = graph.permute_subgraph(&b_c_edge, &hedge_perm[0]);
        let permuted_b_c_edge2 = graph.permute_subgraph(&b_c_edge, &hedge_perm2[0]);
        let permuted_a_b_edge = graph.permute_subgraph(&a_b_edge, &hedge_perm[0]);
        let permuted_a_b_edge2 = graph.permute_subgraph(&a_b_edge, &hedge_perm2[0]);
        let permuted_c_a_edge = graph.permute_subgraph(&c_a_edge, &hedge_perm[0]);
        let permuted_c_a_edge2 = graph.permute_subgraph(&c_a_edge, &hedge_perm2[0]);

        assert_eq!(hedge_perm.len(), 1);

        println!(
            "//a-c\n{}\n//a-b\n{}\n//b-c\n{}",
            graph.dot(&c_a_edge),
            graph.dot(&a_b_edge),
            graph.dot(&b_c_edge)
        );
        println!(
            "//permuted a-b\n{}\n//permuted b-c\n{}\n//permuted c-a\n{}",
            graph.dot(&permuted_a_b_edge),
            graph.dot(&permuted_b_c_edge),
            graph.dot(&permuted_c_a_edge)
        );
        println!(
            "//permuted a-b\n{}\n//permuted b-c\n{}\n//permuted c-a\n{}",
            graph.dot(&permuted_a_b_edge2),
            graph.dot(&permuted_b_c_edge2),
            graph.dot(&permuted_c_a_edge2)
        );

        assert_eq!(a_b_edge, permuted_b_c_edge);
        assert_eq!(b_c_edge, permuted_a_b_edge);
        assert_eq!(c_a_edge, permuted_c_a_edge);

        // println!(
        //     "{}\n{}",
        //     graph.dot(&cleaned_subgraph),
        //     graph.dot(&permuted_subgraph)
        // )
        // let permuted_subgraph = hedgeperm.apply_slice(cleaned_subgraph.filter);
    }

    #[test]
    fn test_permute_graph_double_edges() {
        let mut triangle = HedgeGraphBuilder::new();

        let a = triangle.add_node(());
        let b = triangle.add_node(());
        let c = triangle.add_node(());

        triangle.add_edge(a, b, (), false);

        triangle.add_external_edge(a, (), false, Flow::Sink);
        triangle.add_edge(b, c, (), false);
        triangle.add_external_edge(c, (), false, Flow::Source);
        triangle.add_edge(c, a, (), false);
        triangle.add_edge(c, a, (), false);
        let graph = triangle.build();
        let perm = Permutation::from_cycles(&[vec![0, 2], vec![1]]); //permutes a and c

        let a_b_edge = graph.hedges_between(a, b);
        let b_c_edge = graph.hedges_between(b, c);
        let c_a_edge = graph.hedges_between(c, a);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let permuted_b_c_edge = graph.permute_subgraph(&b_c_edge, &hedge_perm[0]);
        let permuted_a_b_edge = graph.permute_subgraph(&a_b_edge, &hedge_perm[0]);
        let permuted_c_a_edge = graph.permute_subgraph(&c_a_edge, &hedge_perm[0]);

        assert_eq!(hedge_perm.len(), 2);
        assert_eq!(a_b_edge, permuted_b_c_edge);
        assert_eq!(b_c_edge, permuted_a_b_edge);
        assert_eq!(c_a_edge, permuted_c_a_edge);

        println!(
            "//a-c\n{}\n//a-b\n{}\n//b-c\n{}",
            graph.dot(&c_a_edge),
            graph.dot(&a_b_edge),
            graph.dot(&b_c_edge)
        );
        println!(
            "//permuted a-b\n{}\n//permuted b-c\n{}\n//permuted c-a\n{}",
            graph.dot(&permuted_a_b_edge),
            graph.dot(&permuted_b_c_edge),
            graph.dot(&permuted_c_a_edge)
        );
        // let permuted_subgraph = hedgeperm.apply_slice(a.filter);
    }

    #[test]
    fn test_permute_double_triangle() {
        let mut doubletriangle = HedgeGraphBuilder::new();

        let a = doubletriangle.add_node(());
        let b = doubletriangle.add_node(());
        let c = doubletriangle.add_node(());
        let d = doubletriangle.add_node(());

        doubletriangle.add_edge(a, b, (), false);

        doubletriangle.add_edge(b, c, (), false);
        doubletriangle.add_edge(c, d, (), false);
        doubletriangle.add_edge(d, a, (), false);
        doubletriangle.add_edge(c, a, (), false);
        let graph = doubletriangle.build();
        let perm = Permutation::from_cycles(&[vec![0, 2], vec![3]]); //permutes a and c

        let a_b_edge = graph.hedges_between(a, b);
        let b_c_edge = graph.hedges_between(b, c);
        let c_a_edge = graph.hedges_between(c, a);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let permuted_b_c_edge = graph.permute_subgraph(&b_c_edge, &hedge_perm[0]);
        let permuted_a_b_edge = graph.permute_subgraph(&a_b_edge, &hedge_perm[0]);
        let permuted_c_a_edge = graph.permute_subgraph(&c_a_edge, &hedge_perm[0]);

        assert_eq!(hedge_perm.len(), 1);
        assert_eq!(a_b_edge, permuted_b_c_edge);
        assert_eq!(b_c_edge, permuted_a_b_edge);
        assert_eq!(c_a_edge, permuted_c_a_edge);

        println!(
            "//a-c\n{}\n//a-b\n{}\n//b-c\n{}",
            graph.dot(&c_a_edge),
            graph.dot(&a_b_edge),
            graph.dot(&b_c_edge)
        );
        println!(
            "//permuted a-b\n{}\n//permuted b-c\n{}\n//permuted c-a\n{}",
            graph.dot(&permuted_a_b_edge),
            graph.dot(&permuted_b_c_edge),
            graph.dot(&permuted_c_a_edge)
        );
        // let permuted_subgraph = hedgeperm.apply_slice(a.filter);
    }

    #[test]
    fn cyclic_doubled_triangle() {
        let mut doubledtriangle = HedgeGraphBuilder::new();

        let a = doubledtriangle.add_node(());
        let b = doubledtriangle.add_node(());
        let c = doubledtriangle.add_node(());

        doubledtriangle.add_edge(a, b, (), false);
        doubledtriangle.add_edge(a, b, (), false);
        doubledtriangle.add_edge(b, c, (), false);
        doubledtriangle.add_edge(b, c, (), false);
        doubledtriangle.add_edge(c, a, (), false);
        doubledtriangle.add_edge(c, a, (), false);
        let graph = doubledtriangle.build();
        let perm = Permutation::from_cycles(&[vec![0, 1, 2]]); //single cycle

        let a_b_edge = graph.hedges_between(a, b);
        let b_c_edge = graph.hedges_between(b, c);
        let c_a_edge = graph.hedges_between(c, a);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let permuted_b_c_edge = graph.permute_subgraph(&b_c_edge, &hedge_perm[0]);
        let permuted_a_b_edge = graph.permute_subgraph(&a_b_edge, &hedge_perm[0]);
        let permuted_c_a_edge = graph.permute_subgraph(&c_a_edge, &hedge_perm[0]);

        assert_eq!(hedge_perm.len(), 8); //2^3

        println!(
            "//a-c\n{}\n//a-b\n{}\n//b-c\n{}",
            graph.dot(&c_a_edge),
            graph.dot(&a_b_edge),
            graph.dot(&b_c_edge)
        );
        println!(
            "//permuted a-b\n{}\n//permuted b-c\n{}\n//permuted c-a\n{}",
            graph.dot(&permuted_a_b_edge),
            graph.dot(&permuted_b_c_edge),
            graph.dot(&permuted_c_a_edge)
        );
        assert_eq!(a_b_edge, permuted_c_a_edge);
        assert_eq!(c_a_edge, permuted_b_c_edge);
        assert_eq!(b_c_edge, permuted_a_b_edge);
        // let permuted_subgraph = hedgeperm.apply_slice(a.filter);
    }

    #[test]
    fn cycle_permutation() {
        let mut cycle = HedgeGraphBuilder::new();

        let a = cycle.add_node(());
        let b = cycle.add_node(());

        cycle.add_edge(a, b, (), false);
        cycle.add_edge(b, a, (), false);

        let graph = cycle.build();
        let perm = Permutation::from_cycles(&[vec![0, 1]]); //permutes a and b

        let h = Hedge(0);
        let mut h_sub: BitVec = graph.empty_subgraph();
        h_sub.set(h.0, true);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let permuted_h = graph.permute_subgraph(&h_sub, &hedge_perm[0]);
        println!("n perms{}", hedge_perm.len());

        println!("//origninal:\n{}", graph.dot(&h_sub));
        println!("//permuted:\n{}", graph.dot(&permuted_h));
        // let permuted_subgraph = hedgeperm.apply_slice(a.filter);
    }

    #[test]
    fn triple_loop_two_nodes() {
        let mut cycle = HedgeGraphBuilder::new();

        let a = cycle.add_node(());
        let b = cycle.add_node(());

        cycle.add_edge(a, b, (), false);
        cycle.add_edge(b, a, (), false);
        cycle.add_edge(b, a, (), false);

        let graph = cycle.build();
        let perm = Permutation::from_cycles(&[vec![0, 1]]); //permutes a and b

        let h = Hedge(0);
        let mut h_sub: BitVec = graph.empty_subgraph();
        h_sub.set(h.0, true);

        let hedge_perm = graph.permute_vertices(&perm, &|_| ());
        let permuted_h = graph.permute_subgraph(&h_sub, &hedge_perm[0]);
        println!("n perms{}", hedge_perm.len());

        println!("//origninal:\n{}", graph.dot(&h_sub));
        println!("//permuted:\n{}", graph.dot(&permuted_h));
        // let permuted_subgraph = hedgeperm.apply_slice(a.filter);
    }

    #[test]
    fn generate_all() {
        fn generator_pair(n: usize) -> [Permutation; 2] {
            let mut map = Vec::new();
            for i in 0..n {
                map.push((i + 1) % n)
            }
            [
                Permutation::from_map(map),
                Permutation::from_cycles(&[vec![0, n - 1]]),
            ]
        }

        let all_2 = Permutation::generate_all(&generator_pair(2)).unwrap();

        let all_3 = Permutation::generate_all(&generator_pair(3)).unwrap();
        let all_4 = Permutation::generate_all(&generator_pair(4)).unwrap();
        assert_eq!(all_2.len(), 2);
        assert_eq!(all_3.len(), 6);
        assert_eq!(all_4.len(), 24);
    }
}
