use crate::half_edge::hedgevec::Accessors;
use crate::half_edge::involution::{EdgeIndex, HedgePair};
use crate::half_edge::nodestorage::{NodeStorageOps, NodeStorageVec};
use crate::half_edge::EdgeAccessors;
use crate::half_edge::{
    involution::SignOrZero, EdgeData, Flow, Hedge, HedgeGraph, HedgeGraphBuilder,
    InvolutiveMapping, NodeStorage, Orientation, PowersetIterator,
};
use ahash::AHashMap;
use bitvec::vec::BitVec;
use std::{
    fmt::{Display, Formatter},
    hash::Hash,
};
use thiserror::Error;

use super::{Cycle, Inclusion, SubGraph, SubGraphHedgeIter, SubGraphOps};
#[cfg(feature = "layout")]
use crate::half_edge::layout::{
    LayoutEdge, LayoutIters, LayoutParams, LayoutSettings, LayoutVertex,
};

#[cfg(feature = "layout")]
use by_address::ByAddress;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrientedCut {
    ///gives the hedges actually "in" the cut
    pub reference: BitVec,
    ///gives the orientation of the hedges
    pub sign: BitVec,
}

impl Display for OrientedCut {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (i, c) in self.reference.iter().enumerate() {
            if *c {
                if self.sign[i] {
                    write!(f, "+{}", i)?;
                } else {
                    write!(f, "-{}", i)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CutError {
    #[error("Invalid edge")]
    InvalidEdge,
    #[error("Invalid orientation")]
    InvalidOrientation,
    #[error("Cut edge has already been set")]
    CutEdgeAlreadySet,
    #[error("Cut edge is identity")]
    CutEdgeIsIdentity,
}

impl OrientedCut {
    pub fn from_underlying_coerce<E, V, N: NodeStorageOps<NodeData = V>>(
        cut: BitVec,
        graph: &HedgeGraph<E, V, N>,
    ) -> Result<Self, CutError> {
        let mut reference = graph.empty_subgraph::<BitVec>();
        let mut sign = graph.empty_subgraph::<BitVec>();

        for i in cut.included_iter() {
            if reference.includes(&i) {
                return Err(CutError::CutEdgeAlreadySet);
            } else {
                reference.set(i.0, true);
            }
            match graph.edge_store.inv_full(i) {
                InvolutiveMapping::Source { .. } => {
                    sign.set(i.0, true);
                }
                InvolutiveMapping::Sink { source_idx } => {
                    sign.set(source_idx.0, true);
                }
                _ => {}
            }
        }
        Ok(OrientedCut { reference, sign })
    }

    pub fn from_underlying_strict<E, V, N: NodeStorageOps<NodeData = V>>(
        cut: BitVec,
        graph: &HedgeGraph<E, V, N>,
    ) -> Result<Self, CutError> {
        let mut reference = graph.empty_subgraph::<BitVec>();
        let mut sign = graph.empty_subgraph::<BitVec>();

        for i in cut.included_iter() {
            if reference.includes(&i) {
                return Err(CutError::CutEdgeAlreadySet);
            } else {
                reference.set(i.0, true);
            }
            match graph.edge_store.inv_full(i) {
                InvolutiveMapping::Identity { .. } => {
                    return Err(CutError::CutEdgeIsIdentity);
                }
                InvolutiveMapping::Source { .. } => {
                    sign.set(i.0, true);
                }
                InvolutiveMapping::Sink { source_idx } => {
                    sign.set(source_idx.0, true);
                }
            }
        }
        Ok(OrientedCut { reference, sign })
    }

    pub fn all_initial_state_cuts<E, V, N: NodeStorageOps<NodeData = V>>(
        graph: &HedgeGraph<E, V, N>,
    ) -> Vec<Self> {
        let mut all_cuts = Vec::new();

        for c in graph.non_cut_edges() {
            if c.count_ones() == 0 {
                continue;
            }
            let mut all_sources = graph.empty_subgraph::<BitVec>();

            for h in c.included_iter() {
                match graph.edge_store.inv_full(h) {
                    InvolutiveMapping::Identity { .. } => {
                        panic!("cut edge is identity")
                    }
                    InvolutiveMapping::Source { .. } => {
                        all_sources.set(h.0, true);
                    }
                    InvolutiveMapping::Sink { .. } => {}
                }
            }

            let n_cut_edges: u8 = all_sources.count_ones().try_into().unwrap();

            let pset = PowersetIterator::new(n_cut_edges); //.unchecked_sub(1)

            for i in pset {
                let mut cut_content = graph.empty_subgraph::<BitVec>();
                for (j, h) in all_sources.included_iter().enumerate() {
                    // if let Some(j) = j.checked_sub(1) {
                    if i[j] {
                        cut_content.set(graph.inv(h).0, true);
                    } else {
                        cut_content.set(h.0, true);
                    }
                    // } else {
                    // cut_content.set(h, true);
                    // }
                }
                all_cuts.push(Self {
                    sign: cut_content,
                    reference: all_sources.clone(),
                });
            }
        }

        all_cuts
    }

    pub fn winding_number(&self, cycle: &Cycle) -> i32 {
        let mut winding_number = 0;

        for h in cycle.filter.included_iter() {
            winding_number += SignOrZero::from(self.relative_orientation(h)) * 1;
        }

        winding_number
    }
    pub fn iter_edges<'a, E, V, N: NodeStorageOps<NodeData = V>>(
        &'a self,
        graph: &'a HedgeGraph<E, V, N>,
    ) -> impl Iterator<Item = (Orientation, EdgeData<&'a E>)> {
        self.reference
            .included_iter()
            .map(|i| (self.orientation(i, graph), graph.get_edge_data_full(i)))
    }

    pub fn iter_edges_relative<'a, E, V, N: NodeStorageOps<NodeData = V>>(
        &'a self,
        graph: &'a HedgeGraph<E, V, N>,
    ) -> impl Iterator<Item = (Orientation, EdgeData<&'a E>)> {
        self.reference
            .included_iter()
            .map(|i| (self.relative_orientation(i), graph.get_edge_data_full(i)))
    }

    pub fn iter_edges_idx_relative<'a>(
        &'a self,
    ) -> impl Iterator<Item = (Hedge, Orientation)> + 'a {
        self.reference
            .included_iter()
            .map(|i| (i, self.relative_orientation(i)))
    }

    /// If the cut and reference are aligned=> Default
    /// If the reference is included but not the cut, => Reversed
    /// If the cut is included but not the reference, => Undirected
    /// If neither the cut nor the reference is included, => Undirected
    pub fn relative_orientation(&self, i: Hedge) -> Orientation {
        match (self.sign.includes(&i), self.reference.includes(&i)) {
            (true, true) => Orientation::Default,
            (false, false) => Orientation::Undirected,
            (true, false) => Orientation::Undirected,
            (false, true) => Orientation::Reversed,
        }
    }

    pub fn set(&mut self, pair: HedgePair, flow: Flow) {
        if let HedgePair::Paired { source, sink } = pair {
            if !(self.reference.includes(&source) || self.reference.includes(&sink)) {
                self.sign.set(source.0, true);
                match flow {
                    Flow::Source => {
                        self.reference.set(source.0, true);
                    }
                    Flow::Sink => {
                        self.reference.set(sink.0, true);
                    }
                }
            }
        }
    }

    pub fn set_inv(&mut self, pair: HedgePair, flow: Flow) {
        if let HedgePair::Paired { source, sink } = pair {
            if !(self.reference.includes(&source) || self.reference.includes(&sink)) {
                self.reference.set(source.0, true);
                match flow {
                    Flow::Source => {
                        self.sign.set(source.0, true);
                    }
                    Flow::Sink => {
                        self.sign.set(sink.0, true);
                    }
                }
            }
        }
    }

    pub fn orientation<E, V, N: NodeStorage<NodeData = V>>(
        &self,
        i: Hedge,
        graph: &HedgeGraph<E, V, N>,
    ) -> Orientation {
        let orientation = match graph.edge_store.orientation(i) {
            Orientation::Undirected => Orientation::Default,
            o => o,
        };

        match (self.sign.includes(&i), self.reference.includes(&i)) {
            (true, true) => orientation,
            (false, true) => orientation.reverse(),
            _ => Orientation::Undirected,
        }
    }
}

impl Inclusion<Hedge> for OrientedCut {
    fn includes(&self, other: &Hedge) -> bool {
        self.reference.includes(other)
    }
    fn intersects(&self, other: &Hedge) -> bool {
        self.reference.intersects(other)
    }
}

impl Inclusion<BitVec> for OrientedCut {
    fn includes(&self, other: &BitVec) -> bool {
        self.reference.includes(other)
    }

    fn intersects(&self, other: &BitVec) -> bool {
        self.reference.intersects(other)
    }
}

impl Inclusion<OrientedCut> for OrientedCut {
    fn includes(&self, other: &OrientedCut) -> bool {
        self.reference.includes(&other.reference)
    }

    fn intersects(&self, other: &OrientedCut) -> bool {
        self.reference.intersects(&other.reference)
    }
}

impl SubGraph for OrientedCut {
    type Base = BitVec;

    type BaseIter<'a> = SubGraphHedgeIter<'a>;
    fn nedges<E, V, N: NodeStorage<NodeData = V>>(&self, _graph: &HedgeGraph<E, V, N>) -> usize {
        self.nhedges()
    }

    fn included(&self) -> &BitVec {
        self.reference.included()
    }

    fn included_iter(&self) -> Self::BaseIter<'_> {
        self.reference.included_iter()
    }

    fn nhedges(&self) -> usize {
        self.reference.nhedges()
    }
    fn empty(size: usize) -> Self {
        OrientedCut {
            reference: BitVec::empty(size),
            sign: BitVec::empty(size),
        }
    }

    fn dot<E, V, N: NodeStorage<NodeData = V>, Str: AsRef<str>>(
        &self,
        _graph: &crate::half_edge::HedgeGraph<E, V, N>,
        _graph_info: Str,
        _edge_attr: &impl Fn(&E) -> Option<String>,
        _node_attr: &impl Fn(&V) -> Option<String>,
    ) -> String {
        String::new()
    }

    fn hairs(&self, node: &super::HedgeNode) -> BitVec {
        self.reference.hairs(node)
    }

    fn is_empty(&self) -> bool {
        self.sign.count_ones() == 0
    }

    fn string_label(&self) -> String {
        self.sign.string_label()
    }
}

impl OrientedCut {
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[cfg(feature = "layout")]
    pub fn layout<'a, E, V>(
        self,
        graph: &'a HedgeGraph<E, V, NodeStorageVec<V>>,
        params: LayoutParams,
        iters: LayoutIters,
        edge: f64,
    ) -> HedgeGraph<
        LayoutEdge<(&'a E, Orientation, EdgeIndex)>,
        LayoutVertex<&'a V>,
        NodeStorageVec<LayoutVertex<&'a V>>,
    > {
        use indexmap::IndexMap;

        use crate::half_edge::involution::HedgePair;

        let mut left = vec![];
        let mut leftright_map = IndexMap::new();
        let mut right = vec![];

        let graph = self.to_owned_graph(graph);

        for (p, _, i) in graph.iter_all_edges() {
            if let HedgePair::Unpaired { hedge, flow } = p {
                let d = i.data.2;
                match flow {
                    Flow::Sink => {
                        leftright_map
                            .entry(d)
                            .or_insert_with(|| [Some(hedge), None])[0] = Some(hedge)
                    }
                    Flow::Source => {
                        leftright_map
                            .entry(d)
                            .or_insert_with(|| [None, Some(hedge)])[1] = Some(hedge)
                    }
                }
            }
        }

        for (_, [i, j]) in leftright_map {
            if let Some(i) = i {
                left.push(i);
            }
            if let Some(j) = j {
                right.push(j);
            }
        }

        let settings = LayoutSettings::left_right_square(&graph, params, iters, edge, left, right);

        // println!("{:?}", settings);
        graph.layout(settings)
    }

    /// Take the graph and split it along the cut, putting the cut orientation, and original edge index as additional data.
    pub fn to_owned_graph<'a, E, V, N: NodeStorageOps<NodeData = V>>(
        self,
        graph: &'a HedgeGraph<E, V, N>,
    ) -> HedgeGraph<(&'a E, Orientation, EdgeIndex), &'a V, N::OpStorage<&'a V>> {
        let mut new_graph = graph.map_data_ref(&|_, v, _| v, &|_, i, _, e| {
            e.map(|d| (d, Orientation::Undirected, i))
        });

        for (h, o) in self.iter_edges_idx_relative() {
            new_graph.edge_store.set_flow(h, Flow::Source);
            new_graph[[&h]].1 = o;
            let mut datainv = new_graph[[&h]].clone();
            datainv.1 = o;
            let supo = new_graph.orientation(h);
            let data = EdgeData::new(datainv, supo);
            let invh = new_graph.inv(h);
            new_graph.split_edge(invh, data).unwrap();
        }

        new_graph
    }
}
