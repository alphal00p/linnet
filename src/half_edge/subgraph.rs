use std::{
    hash::Hash,
    num::TryFromIntError,
    ops::{
        BitAndAssign, BitOrAssign, BitXorAssign, Range, RangeFrom, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

use ahash::AHashSet;
use bitvec::{bitvec, order::Lsb0, vec::BitVec};

use super::{
    involution::{Flow, HedgePair},
    nodestore::NodeStorageOps,
    GVEdgeAttrs, Hedge, HedgeGraph, NodeStorage, PowersetIterator,
};

const BASE62_ALPHABET: &[u8; 62] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

pub trait SubGraphOps<Other: SubGraph = Self>: SubGraph {
    // type Other = Self;

    fn all_pairwise_ops(
        left: &mut AHashSet<Self>,
        right: &[Self],
        op: &impl Fn(&Self, &Self) -> Self,
    ) -> bool {
        Self::all_pairwise_ops_filter_map(left, right, op, &|x| Some(x))
    }

    fn all_pairwise_ops_filter_map(
        left: &mut AHashSet<Self>,
        right: &[Self],
        op: &impl Fn(&Self, &Self) -> Self,
        filter_map: &impl Fn(Self) -> Option<Self>,
    ) -> bool {
        let mut added = false;
        let mut new = AHashSet::new();
        for l in left.iter() {
            for r in right {
                if let Some(n) = filter_map(op(l, r)) {
                    new.insert(n);
                }
            }
        }
        for n in new.drain() {
            if left.insert(n) {
                added = true;
            }
        }
        added
    }

    fn all_pairwise_unions(left: &mut AHashSet<Self>, right: &[Self]) -> bool {
        Self::all_pairwise_ops(left, right, &|l, r| l.union(r))
    }

    fn all_pairwise_sym_diff(left: &mut AHashSet<Self>, right: &[Self]) -> bool {
        Self::all_pairwise_ops(left, right, &|l, r| l.sym_diff(r))
    }

    fn all_ops_iterative_filter_map(
        set: &[Self],
        op: &impl Fn(&Self, &Self) -> Self,
        filter_map: &impl Fn(Self) -> Option<Self>,
    ) -> AHashSet<Self> {
        let mut s: AHashSet<_> = set.iter().cloned().collect();
        while Self::all_pairwise_ops_filter_map(&mut s, set, op, filter_map) {}
        s.drain().filter_map(filter_map).collect()
    }

    fn all_unions_iterative(set: &[Self]) -> AHashSet<Self> {
        Self::all_ops_iterative_filter_map(set, &|a, b| a.union(b), &|a| Some(a))
    }

    fn all_sym_diff_iterative(set: &[Self]) -> AHashSet<Self> {
        Self::all_ops_iterative_filter_map(set, &|a, b| a.sym_diff(b), &|a| Some(a))
    }

    fn all_op_powerset_filter_map(
        set: &[Self],
        op: impl Fn(&mut Self, &Self),
        filter_map: &impl Fn(Self) -> Option<Self>,
    ) -> Result<AHashSet<Self>, TryFromIntError> {
        let mut s = AHashSet::new();
        let mut pset = PowersetIterator::new(set.len().try_into()?);

        pset.next().unwrap(); //Skip the empty set

        for i in pset {
            let mut ones = i.iter_ones();

            let mut union = set[ones.next().unwrap()].clone();

            for o in ones {
                op(&mut union, &set[o]);
            }

            if let Some(union) = filter_map(union) {
                s.insert(union);
            }
        }

        Ok(s)
    }

    fn all_unions_powerset_filter_map(
        set: &[Self],
        filter_map: &impl Fn(Self) -> Option<Self>,
    ) -> Result<AHashSet<Self>, TryFromIntError> {
        Self::all_op_powerset_filter_map(set, |l, r| l.union_with(r), filter_map)
    }

    fn all_sym_diff_powerset(
        set: &[Self],
        filter_map: &impl Fn(Self) -> Option<Self>,
    ) -> Result<AHashSet<Self>, TryFromIntError> {
        Self::all_op_powerset_filter_map(set, |l, r| l.sym_diff_with(r), filter_map)
    }

    fn n_op(n: usize, set: &[Self], op: &impl Fn(&Self, &Self) -> Self) -> AHashSet<Self> {
        if n == 0 {
            AHashSet::new()
        } else {
            let mut s = Self::n_op(n - 1, set, op);
            Self::all_pairwise_ops(&mut s, set, op);
            s
        }
    }

    fn intersect_with(&mut self, other: &Self);
    fn union_with(&mut self, other: &Self);
    fn union_with_iter(&mut self, other: impl Iterator<Item = Hedge>);
    fn sym_diff_with(&mut self, other: &Self);
    fn empty_intersection(&self, other: &Self) -> bool;
    fn empty_union(&self, other: &Self) -> bool;
    fn intersection(&self, other: &Self) -> Self {
        let mut new = self.clone();
        new.intersect_with(other);
        new
    }
    fn union(&self, other: &Self) -> Self {
        let mut new = self.clone();
        new.union_with(other);

        new
    }
    fn sym_diff(&self, other: &Self) -> Self {
        let mut new = self.clone();
        new.sym_diff_with(other);
        new
    }

    fn subtract_with(&mut self, other: &Self);
    fn subtract(&self, other: &Self) -> Self {
        let mut new = self.clone();
        new.subtract_with(other);
        new
    }

    fn complement<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> Self;
}

pub trait Inclusion<T> {
    fn includes(&self, other: &T) -> bool;
    fn intersects(&self, other: &T) -> bool;
}

pub struct SubGraphHedgeIter<'a> {
    iter: std::iter::Map<bitvec::slice::IterOnes<'a, usize, Lsb0>, fn(usize) -> Hedge>,
}

impl Iterator for SubGraphHedgeIter<'_> {
    type Item = Hedge;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub trait BaseSubgraph: SubGraph + ModifySubgraph<Hedge> + ModifySubgraph<HedgePair> {
    fn from_filter<E, V, N: NodeStorageOps<NodeData = V>, F: FnMut(&E) -> bool>(
        graph: &HedgeGraph<E, V, N>,
        filter: F,
    ) -> Self;

    fn from_hedge_iter<I: Iterator<Item = Hedge>>(iter: I, len: usize) -> Self;
}

pub trait ModifySubgraph<Index> {
    fn add(&mut self, index: Index);

    fn sub(&mut self, index: Index);
}

pub trait SubGraph:
    Clone
    + Eq
    + Hash
    + Inclusion<Self>
    + Inclusion<Self::Base>
    + Inclusion<std::ops::Range<Hedge>>
    + Inclusion<std::ops::RangeToInclusive<Hedge>>
    + Inclusion<std::ops::RangeInclusive<Hedge>>
    + Inclusion<std::ops::RangeTo<Hedge>>
    + Inclusion<std::ops::RangeFrom<Hedge>>
    + Inclusion<Hedge>
{
    type Base: SubGraph;
    type BaseIter<'a>: Iterator<Item = Hedge>
    where
        Self: 'a;
    /// maximal graph that contains all nodes of the subgraph
    fn covers<E, V, N: NodeStorageOps<NodeData = V>>(&self, graph: &HedgeGraph<E, V, N>) -> BitVec {
        let mut covering = graph.empty_subgraph::<BitVec>();
        for i in self.included_iter() {
            covering.union_with_iter(graph.neighbors(i))
        }
        covering
    }

    /// Contains a half-edge with index >= hedge
    fn has_greater(&self, hedge: Hedge) -> bool {
        self.intersects(&(hedge..))
    }

    /// Contains a half-edge with index < hedge
    fn has_lesser(&self, hedge: Hedge) -> bool {
        self.intersects(&(..hedge))
    }

    ///joins two subgraphs into one, (this is not union)
    fn join(mut self, other: Self) -> Self {
        self.join_mut(other);
        self
    }

    /// Appends all incuded half-edges at the end of other
    fn join_mut(&mut self, other: Self);

    fn background_color(&self, hedge_pair: Option<HedgePair>) -> Option<String> {
        let color = "gray".to_string();

        if let Some(p) = hedge_pair {
            if let HedgePair::Split { .. } = p {
                Some(color)
            } else {
                None
            }
        } else {
            Some(color)
        }
    }

    fn string_label(&self) -> String;
    fn included_iter(&self) -> Self::BaseIter<'_>;
    // SubGraphHedgeIter {
    //     SubGraphHedgeIter {
    //         iter: self.included().iter_ones().map(Hedge),
    //     }
    // }
    /// Returns a simple Self::Base of all included hedges
    fn included(&self) -> &Self::Base;

    /// Number of half-edges in the graph this is a subgraph of
    fn size(&self) -> usize;
    /// Number of half-edges included in the subgraph
    fn nhedges(&self) -> usize;
    /// Number of full edges included in the subgraph
    fn nedges<E, V, N: NodeStorageOps<NodeData = V>>(&self, graph: &HedgeGraph<E, V, N>) -> usize; //not counting unpaired hedges

    fn dot_fmt<W: std::fmt::Write, E, V, N: NodeStorageOps<NodeData = V>, Str: AsRef<str>>(
        &self,
        writer: &mut W,
        graph: &HedgeGraph<E, V, N>,
        graph_info: Str,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> Result<(), std::fmt::Error> {
        writeln!(writer, "digraph {{")?;
        writeln!(
            writer,
            "  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\"; layout=\"neato\";",
        )?;
        writeln!(writer, "{}", graph_info.as_ref())?;

        for (n, _, v) in graph.iter_node_data(self) {
            if let Some(a) = node_attr(v) {
                writeln!(writer, "  {} [{}];", n.0, a)?;
            }
        }

        for (hedge_pair, _, data) in graph.iter_all_edges() {
            let subgraph_pair = hedge_pair.with_subgraph(self);

            let attr = GVEdgeAttrs {
                color: self.background_color(subgraph_pair),
                label: None,
                other: edge_attr(data.data),
            };
            write!(writer, "  ")?;
            if let Some(p) = subgraph_pair {
                let attr = p.fill_color(attr);
                p.dot_fmt(writer, graph, data.orientation, attr)?;
            } else {
                let attr = hedge_pair.fill_color(attr);
                hedge_pair.dot_fmt(writer, graph, data.orientation, attr)?;
            }
        }
        writeln!(writer, "}}")?;
        Ok(())
    }

    fn dot_io<W: std::io::Write, E, V, N: NodeStorageOps<NodeData = V>, Str: AsRef<str>>(
        &self,
        writer: &mut W,
        graph: &HedgeGraph<E, V, N>,
        graph_info: Str,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> Result<(), std::io::Error> {
        writeln!(writer, "digraph {{")?;
        writeln!(
            writer,
            "  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\"; layout=\"neato\";",
        )?;
        writeln!(writer, "{}", graph_info.as_ref())?;

        for (n, _, v) in graph.iter_node_data(self) {
            if let Some(a) = node_attr(v) {
                writeln!(writer, "  {} [{}];", n.0, a)?;
            }
        }

        for (hedge_pair, _, data) in graph.iter_all_edges() {
            let subgraph_pair = hedge_pair.with_subgraph(self);

            let attr = GVEdgeAttrs {
                color: self.background_color(subgraph_pair),
                label: None,
                other: edge_attr(data.data),
            };
            write!(writer, "  ")?;
            if let Some(p) = subgraph_pair {
                let attr = p.fill_color(attr);
                p.dot_io(writer, graph, data.orientation, attr)?;
            } else {
                let attr = hedge_pair.fill_color(attr);
                hedge_pair.dot_io(writer, graph, data.orientation, attr)?;
            }
        }
        writeln!(writer, "}}")?;
        Ok(())
    }

    fn hairs(&self, node: impl Iterator<Item = Hedge>) -> BitVec {
        let mut hairs = BitVec::empty(self.size());
        for h in node {
            if self.includes(&h) {
                hairs.add(h)
            }
        }
        hairs
    }
    fn empty(size: usize) -> Self;
    fn is_empty(&self) -> bool;
}

impl Inclusion<BitVec> for BitVec {
    fn includes(&self, other: &BitVec) -> bool {
        &self.intersection(other) == other
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.intersection(other).count_ones() > 0
    }
}

impl Inclusion<Range<Hedge>> for BitVec {
    fn includes(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &Range<Hedge>) -> bool {
        (other.start.0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeTo<Hedge>> for BitVec {
    fn includes(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeTo<Hedge>) -> bool {
        (0..other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeToInclusive<Hedge>> for BitVec {
    fn includes(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeToInclusive<Hedge>) -> bool {
        (0..=other.end.0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeFrom<Hedge>> for BitVec {
    fn includes(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeFrom<Hedge>) -> bool {
        (other.start.0..).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<RangeInclusive<Hedge>> for BitVec {
    fn includes(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).all(|a| self.includes(&Hedge(a)))
    }

    fn intersects(&self, other: &RangeInclusive<Hedge>) -> bool {
        (other.start().0..=other.end().0).any(|a| self.includes(&Hedge(a)))
    }
}

impl Inclusion<Hedge> for BitVec {
    fn includes(&self, other: &Hedge) -> bool {
        self[other.0]
    }

    fn intersects(&self, other: &Hedge) -> bool {
        self[other.0]
    }
}

impl ModifySubgraph<Hedge> for BitVec {
    fn add(&mut self, hedge: Hedge) {
        self.set(hedge.0, true);
    }

    fn sub(&mut self, hedge: Hedge) {
        self.set(hedge.0, false);
    }
}

impl ModifySubgraph<HedgePair> for BitVec {
    fn add(&mut self, index: HedgePair) {
        match index {
            HedgePair::Paired { source, sink } => {
                self.add(source);
                self.add(sink);
            }
            HedgePair::Split {
                source,
                sink,
                split,
            } => match split {
                Flow::Source => {
                    self.add(source);
                    self.sub(sink);
                }
                Flow::Sink => {
                    self.add(sink);
                    self.sub(source);
                }
            },
            HedgePair::Unpaired { hedge, .. } => {
                self.add(hedge);
            }
        }
    }

    fn sub(&mut self, index: HedgePair) {
        match index {
            HedgePair::Paired { source, sink } => {
                self.sub(source);
                self.sub(sink);
            }
            HedgePair::Split {
                source,
                sink,
                split,
            } => match split {
                Flow::Source => {
                    self.sub(source);
                    // self.sub(sink);
                }
                Flow::Sink => {
                    self.sub(sink);
                    // self.sub(source);
                }
            },
            HedgePair::Unpaired { hedge, .. } => {
                self.sub(hedge);
            }
        }
    }
}

impl BaseSubgraph for BitVec {
    fn from_filter<E, V, N: NodeStorageOps<NodeData = V>, F: FnMut(&E) -> bool>(
        graph: &HedgeGraph<E, V, N>,
        mut filter: F,
    ) -> Self {
        let mut empty: BitVec = graph.empty_subgraph();

        for (p, _, d) in graph.iter_all_edges() {
            if filter(d.data) {
                empty.add(p);
            }
        }

        empty
    }

    fn from_hedge_iter<I: Iterator<Item = Hedge>>(iter: I, len: usize) -> Self {
        let mut subgraph = BitVec::empty(len);

        for h in iter {
            subgraph.add(h);
        }

        subgraph
    }
}

impl SubGraph for BitVec {
    type Base = BitVec;
    type BaseIter<'a> = SubGraphHedgeIter<'a>;
    fn included(&self) -> &BitVec {
        self
    }

    fn size(&self) -> usize {
        self.len()
    }

    fn has_greater(&self, hedge: Hedge) -> bool {
        (hedge.0..self.len()).any(|h| self.includes(&Hedge(h)))
    }

    fn hairs(&self, node: impl Iterator<Item = Hedge>) -> BitVec {
        let mut hairs = BitVec::empty(self.size());

        for h in node {
            if self.includes(&h) {
                hairs.add(h);
            }
        }

        hairs
    }

    fn join_mut(&mut self, other: Self) {
        self.extend(other);
    }
    fn included_iter(&self) -> Self::BaseIter<'_> {
        SubGraphHedgeIter {
            iter: self.iter_ones().map(Hedge),
        }
    }
    fn nedges<E, V, N: NodeStorageOps<NodeData = V>>(&self, graph: &HedgeGraph<E, V, N>) -> usize {
        let mut count = 0;
        for i in self.included_iter() {
            if i != graph.inv(i) && self.includes(&graph.inv(i)) {
                count += 1;
            }
        }
        count / 2
    }

    fn nhedges(&self) -> usize {
        self.count_ones()
    }

    fn empty(size: usize) -> Self {
        bitvec![usize, Lsb0; 0; size]
    }

    fn string_label(&self) -> String {
        if self.is_empty() {
            return "0".to_string();
        }

        let mut digits = vec![0u8]; // Initialize with a single zero digit

        // Iterate over the bits from MSB to LSB
        for bit in self.iter().by_vals().rev() {
            let mut carry = 0u8;

            // Multiply existing digits by 2 (shift left)
            for digit in &mut digits {
                let temp = (*digit as u16) * 2 + carry as u16;
                *digit = (temp % 62) as u8;
                carry = (temp / 62) as u8;
            }

            if carry > 0 {
                digits.push(carry);
            }

            // Add the current bit (if it's 1)
            if bit {
                let mut carry = 1u8;
                for digit in &mut digits {
                    let temp = *digit as u16 + carry as u16;
                    *digit = (temp % 62) as u8;
                    carry = (temp / 62) as u8;

                    if carry == 0 {
                        break;
                    }
                }
                if carry > 0 {
                    digits.push(carry);
                }
            }
        }

        // Map digits to base62 characters and reverse the result
        let base62_string: String = digits
            .iter()
            .rev()
            .map(|&d| BASE62_ALPHABET[d as usize] as char)
            .collect();

        base62_string
    }

    fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }
}

impl SubGraphOps for BitVec {
    fn intersect_with(&mut self, other: &Self) {
        self.bitand_assign(other)
    }

    fn union_with_iter(&mut self, other: impl Iterator<Item = Hedge>) {
        for h in other {
            self.add(h)
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.bitor_assign(other)
    }

    fn sym_diff_with(&mut self, other: &Self) {
        self.bitxor_assign(other)
    }

    fn empty_union(&self, other: &Self) -> bool {
        self.union(other).count_ones() == 0
    }

    fn empty_intersection(&self, other: &Self) -> bool {
        self.intersection(other).count_ones() == 0
    }

    fn complement<E, V, N: NodeStorage<NodeData = V>>(&self, _graph: &HedgeGraph<E, V, N>) -> Self {
        !self.clone()
    }

    fn subtract_with(&mut self, other: &Self) {
        self.bitand_assign(!other.clone());
    }
}

pub mod contracted;
pub use contracted::ContractedSubGraph;
pub mod cut;
pub use cut::OrientedCut;
pub mod cycle;
pub use cycle::Cycle;
pub mod internal;
pub use internal::InternalSubGraph;
pub mod node;
pub use node::HedgeNode;
pub mod empty;
pub use empty::Empty;
pub mod full;
pub use full::FullOrEmpty;
