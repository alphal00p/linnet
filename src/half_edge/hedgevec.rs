use std::ops::{Index, IndexMut, Neg};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use bitvec::vec::BitVec;

use super::{
    involution::{
        EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Involution, InvolutionError,
        InvolutiveMapping, Orientation,
    },
    subgraph::SubGraph,
    HedgeGraph, HedgeGraphError, NodeStorage,
};

#[derive(
    Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode,
)]
pub struct SmartHedgeVec<T> {
    pub(super) data: Vec<(T, HedgePair)>,
    involution: Involution,
}

impl<T> AsRef<Involution> for SmartHedgeVec<T> {
    fn as_ref(&self) -> &Involution {
        &self.involution
    }
}

pub trait Accessors<Index> {
    type Data;

    fn orientation(&self, index: Index) -> Orientation;
    fn set_orientation(&mut self, index: Index, orientation: Orientation);
    fn pair(&self, index: Index) -> HedgePair;
    // fn flow(&self, index: Index) -> Flow;
    // fn set_flow(&mut self, index: Index, flow: Flow);
    fn data(&self, index: Index) -> &Self::Data;
    fn data_mut(&mut self, index: Index) -> &mut Self::Data;
}

impl<T> Accessors<EdgeIndex> for SmartHedgeVec<T> {
    type Data = T;

    fn orientation(&self, index: EdgeIndex) -> Orientation {
        let h = self[&index].1.any_hedge();
        self.involution.orientation(h)
    }

    fn set_orientation(&mut self, index: EdgeIndex, orientation: Orientation) {
        let h = self[&index].1.any_hedge();
        self.involution.set_orientation(h, orientation);
    }

    fn pair(&self, index: EdgeIndex) -> HedgePair {
        self[&index].1
    }

    fn data(&self, index: EdgeIndex) -> &Self::Data {
        &self[index]
    }

    fn data_mut(&mut self, index: EdgeIndex) -> &mut Self::Data {
        &mut self[index]
    }
}

impl<T> Accessors<Hedge> for SmartHedgeVec<T> {
    type Data = T;

    fn orientation(&self, index: Hedge) -> Orientation {
        self.involution.orientation(index)
    }

    fn set_orientation(&mut self, index: Hedge, orientation: Orientation) {
        self.involution.set_orientation(index, orientation);
    }

    fn pair(&self, index: Hedge) -> HedgePair {
        let e = self[&index];
        self[&e].1
    }

    fn data(&self, index: Hedge) -> &Self::Data {
        let e = self[&index];
        &self[e]
    }

    fn data_mut(&mut self, index: Hedge) -> &mut Self::Data {
        let e = self[&index];
        &mut self[e]
    }
}

impl<T> Accessors<HedgePair> for SmartHedgeVec<T> {
    type Data = T;

    fn orientation(&self, index: HedgePair) -> Orientation {
        let index = index.any_hedge();
        self.involution.orientation(index)
    }

    fn set_orientation(&mut self, index: HedgePair, orientation: Orientation) {
        let index = index.any_hedge();
        self.involution.set_orientation(index, orientation);
    }

    fn pair(&self, index: HedgePair) -> HedgePair {
        index
    }

    fn data(&self, index: HedgePair) -> &Self::Data {
        let index = index.any_hedge();
        let e = self[&index];
        &self[e]
    }

    fn data_mut(&mut self, index: HedgePair) -> &mut Self::Data {
        let index = index.any_hedge();
        let e = self[&index];
        &mut self[e]
    }
}

impl<T> SmartHedgeVec<T> {
    pub fn new(involution: Involution<T>) -> Self {
        let mut data = Vec::new();

        let involution = involution.map_full(|a, d| {
            let new_data = d;
            let edgeid = EdgeIndex(data.len());
            data.push((new_data.data, a));
            EdgeData::new(edgeid, new_data.orientation)
        });
        SmartHedgeVec { data, involution }
    }

    pub(crate) fn fix_hedge_pairs(&mut self) {
        for (i, d) in self.involution.iter_edge_data() {
            let hedge_pair = self.involution.hedge_pair(i);
            self.data[d.data.0].1 = hedge_pair;
        }
    }

    pub fn n_dangling(&self) -> usize {
        self.involution
            .inv
            .iter()
            .filter(|e| e.is_identity())
            .count()
    }

    pub fn map<T2, F: Fn(HedgePair, EdgeData<&T>) -> EdgeData<T2>>(
        &self,
        f: &F,
    ) -> SmartHedgeVec<T2> {
        let mut data = Vec::new();
        let involution = self.involution.clone().map_full(|a, d| {
            let d = d.map(|i| &self.data[i.0].0);
            let new_data = f(a, d);
            let edgeid = EdgeIndex(data.len());
            data.push((new_data.data, a));
            EdgeData::new(edgeid, new_data.orientation)
        });

        SmartHedgeVec { data, involution }
    }

    pub fn new_hedgevec<T2>(
        &self,
        mut f: impl FnMut(&T, EdgeIndex, &HedgePair) -> T2,
    ) -> HedgeVec<T2> {
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, (e, pair))| f(e, EdgeIndex(i), pair))
            .collect();

        HedgeVec(data)
    }

    pub fn new_hedgevec_from_iter<T2, I: IntoIterator<Item = T2>>(
        &self,
        iter: I,
    ) -> Result<HedgeVec<T2>, HedgeGraphError> {
        let data: Vec<_> = iter.into_iter().collect();
        if data.len() != self.data.len() {
            return Err(HedgeGraphError::DataLengthMismatch);
        }

        Ok(HedgeVec(data))
    }

    pub fn map_data<T2, V, N: NodeStorage<NodeData = V>>(
        self,
        node_store: &N,
        mut edge_map: impl FnMut(&Involution<EdgeIndex>, &N, HedgePair, EdgeData<T>) -> EdgeData<T2>,
    ) -> SmartHedgeVec<T2> {
        let mut involution = self.involution.clone();
        SmartHedgeVec {
            data: self
                .data
                .into_iter()
                .map(|(e, h)| {
                    let new_data = edge_map(
                        &involution,
                        node_store,
                        h,
                        EdgeData::new(e, involution.orientation(h.any_hedge())),
                    );

                    involution.edge_data_mut(h.any_hedge()).orientation = new_data.orientation;
                    (new_data.data, h)
                })
                .collect(),
            involution,
        }
    }

    pub fn map_data_ref<'a, T2, V, N: NodeStorage<NodeData = V>>(
        &'a self,
        graph: &'a HedgeGraph<T, V, N>,
        edge_map: &impl Fn(
            &'a HedgeGraph<T, V, N>,
            EdgeIndex,
            HedgePair,
            EdgeData<&'a T>,
        ) -> EdgeData<T2>,
    ) -> SmartHedgeVec<T2> {
        let mut involution = self.involution.clone();
        SmartHedgeVec {
            data: self
                .data
                .iter()
                .enumerate()
                .map(|(i, (e, h))| {
                    let new_edgedata = edge_map(
                        graph,
                        EdgeIndex(i),
                        *h,
                        EdgeData::new(e, self.orientation(h.any_hedge())),
                    );

                    involution.edge_data_mut(h.any_hedge()).orientation = new_edgedata.orientation;
                    (new_edgedata.data, *h)
                })
                .collect(),
            involution,
        }
    }

    pub fn n_paired(&self) -> usize {
        self.involution.inv.iter().filter(|e| e.is_source()).count()
    }

    pub(crate) fn add_dangling_edge(
        self,
        data: T,
        flow: Flow,
        orientation: impl Into<Orientation>,
    ) -> (Self, Hedge) {
        let mut involution = self.involution;
        let o = orientation.into();

        let mut edge_data = self.data;
        let edge_index = EdgeIndex(edge_data.len());
        let hedge = involution.add_identity(edge_index, o, flow);
        edge_data.push((data, HedgePair::Unpaired { hedge, flow }));

        (
            SmartHedgeVec {
                data: edge_data,
                involution,
            },
            hedge,
        )
    }

    pub(crate) fn add_paired(
        self,
        data: T,
        orientation: impl Into<Orientation>,
    ) -> (Self, Hedge, Hedge) {
        let mut involution = self.involution;
        let o = orientation.into();

        let mut edge_data = self.data;
        let edge_index = EdgeIndex(edge_data.len());
        let (source, sink) = involution.add_pair(edge_index, o);
        edge_data.push((data, HedgePair::Paired { source, sink }));

        (
            SmartHedgeVec {
                data: edge_data,
                involution,
            },
            source,
            sink,
        )
    }

    fn connect_identities(
        &mut self,
        source: Hedge,
        sink: Hedge,
        merge_fn: impl Fn(Flow, EdgeData<T>, Flow, EdgeData<T>) -> (Flow, EdgeData<T>),
    ) {
        let g = self;
        let source_edge_id = g.involution[source];
        let sink_edge_id = g.involution[sink];
        let last = g.data.len().checked_sub(1).unwrap();
        let second_last = g.data.len().checked_sub(2).unwrap();

        let mut remaps: [Option<(EdgeIndex, EdgeIndex)>; 2] = [None, None];

        // If sink_edge_id.0 is already the last, swap it to second-last first.
        if sink_edge_id.0 == last {
            g.data.swap(sink_edge_id.0, second_last); // swap last and second last

            if source_edge_id.0 != second_last {
                g.data.swap(source_edge_id.0, last);

                // now we need to remap any pointers to second_last, to source_edge_id
                remaps[0] = Some((EdgeIndex(second_last), source_edge_id));
            }
        } else {
            g.data.swap(source_edge_id.0, last);
            g.data.swap(sink_edge_id.0, second_last);

            if source_edge_id.0 == second_last {
                remaps[0] = Some((EdgeIndex(last), sink_edge_id));
            } else {
                remaps[0] = Some((EdgeIndex(last), source_edge_id));
                remaps[1] = Some((EdgeIndex(second_last), sink_edge_id));
            }
        }

        let source_data = EdgeData::new(g.data.pop().unwrap().0, g.involution.orientation(source));

        let sink_data = EdgeData::new(g.data.pop().unwrap().0, g.involution.orientation(sink));
        let (merge_flow, merge_data) = merge_fn(
            g.involution.flow(source),
            source_data,
            g.involution.flow(sink),
            sink_data,
        );

        let new_edge_data = EdgeData::new(EdgeIndex(g.data.len()), merge_data.orientation);
        let pair = match merge_flow {
            Flow::Sink => HedgePair::Paired {
                source: sink,
                sink: source,
            },
            Flow::Source => HedgePair::Paired { source, sink },
        };

        g.data.push((merge_data.data, pair));

        for (_, d) in g.involution.iter_mut() {
            match d {
                InvolutiveMapping::Source { data, .. } => {
                    if let Some((old, new)) = &remaps[0] {
                        if data.data == *old {
                            data.data = *new;
                        }
                    }
                    if let Some((old, new)) = &remaps[1] {
                        if data.data == *old {
                            data.data = *new;
                        }
                    }
                }
                InvolutiveMapping::Identity { data, .. } => {
                    if let Some((old, new)) = &remaps[0] {
                        if data.data == *old {
                            data.data = *new;
                        }
                    }
                    if let Some((old, new)) = &remaps[1] {
                        if data.data == *old {
                            data.data = *new;
                        }
                    }
                }
                _ => {}
            }
        }
        g.involution
            .connect_identities(source, sink, |_, _, _, _| (merge_flow, new_edge_data))
            .unwrap();
    }

    pub(crate) fn sew(
        &mut self,
        matching_fn: impl Fn(Flow, EdgeData<&T>, Flow, EdgeData<&T>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<T>, Flow, EdgeData<T>) -> (Flow, EdgeData<T>),
    ) -> Result<(), HedgeGraphError> {
        let nhedges = self.hedge_len();

        let g = self;

        let mut found_match = true;

        while found_match {
            let mut matching_ids = None;

            for i in (0..nhedges).map(Hedge) {
                if let InvolutiveMapping::Identity {
                    data: datas,
                    underlying: underlyings,
                } = g.involution.hedge_data(i)
                {
                    for j in ((i.0 + 1)..nhedges).map(Hedge) {
                        if let InvolutiveMapping::Identity { data, underlying } =
                            &g.involution.inv[j.0]
                        {
                            if matching_fn(
                                *underlyings,
                                datas.as_ref().map(|a| &g.data[a.0].0),
                                *underlying,
                                data.as_ref().map(|a| &g.data[a.0].0),
                            ) {
                                matching_ids = Some((i, j));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((source, sink)) = matching_ids {
                g.connect_identities(source, sink, &merge_fn);
            } else {
                found_match = false;
            }
        }
        Ok(())
    }

    pub(crate) fn join(
        self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&T>, Flow, EdgeData<&T>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<T>, Flow, EdgeData<T>) -> (Flow, EdgeData<T>),
    ) -> Result<Self, HedgeGraphError> {
        let self_empty_filter = BitVec::empty(self.hedge_len());
        let mut full_self = !self_empty_filter.clone();
        let other_empty_filter = BitVec::empty(other.hedge_len());
        let mut full_other = self_empty_filter.clone();
        full_self.extend(other_empty_filter.clone());
        full_other.extend(!other_empty_filter.clone());

        let self_inv_shift = self.hedge_len();
        let mut edge_data = self.data;
        let edge_data_shift = edge_data.len();
        edge_data.extend(other.data);

        let involution = Involution {
            inv: self
                .involution
                .into_iter()
                .chain(other.involution.into_iter().map(|(i, m)| {
                    (
                        i,
                        match m {
                            InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink {
                                source_idx: Hedge(source_idx.0 + self_inv_shift),
                            },
                            InvolutiveMapping::Source { data, sink_idx } => {
                                InvolutiveMapping::Source {
                                    data: data.map(|e| EdgeIndex(e.0 + edge_data_shift)),
                                    sink_idx: Hedge(sink_idx.0 + self_inv_shift),
                                }
                            }
                            InvolutiveMapping::Identity { data, underlying } => {
                                InvolutiveMapping::Identity {
                                    data: data.map(|e| EdgeIndex(e.0 + edge_data_shift)),
                                    underlying,
                                }
                            }
                        },
                    )
                }))
                .map(|(_, m)| m)
                .collect(),
        };

        let mut found_match = true;

        let mut g = Self {
            data: edge_data,
            involution,
        };

        g.fix_hedge_pairs();

        while found_match {
            let mut matching_ids = None;

            for i in full_self.included_iter() {
                if let InvolutiveMapping::Identity {
                    data: datas,
                    underlying: underlyings,
                } = g.involution.hedge_data(i)
                {
                    for j in full_other.included_iter() {
                        if let InvolutiveMapping::Identity { data, underlying } =
                            &g.involution.inv[j.0]
                        {
                            if matching_fn(
                                *underlyings,
                                datas.as_ref().map(|a| &g.data[a.0].0),
                                *underlying,
                                data.as_ref().map(|a| &g.data[a.0].0),
                            ) {
                                matching_ids = Some((i, j));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((source, sink)) = matching_ids {
                g.connect_identities(source, sink, &merge_fn);
            } else {
                found_match = false;
            }
        }

        Ok(g)
    }

    pub(crate) fn split_edge(
        &mut self,
        hedge: Hedge,
        data: EdgeData<T>,
    ) -> Result<(), InvolutionError> {
        let new_data = EdgeData::new(EdgeIndex(self.edge_len()), data.orientation);

        let invh = self.inv(hedge);
        let flow = self.flow(hedge);
        self.involution.split_edge(hedge, new_data)?;

        self.data
            .push((data.data, HedgePair::Unpaired { hedge, flow }));

        let invh_edge_id = self.involution[invh];
        self[&invh_edge_id].1 = HedgePair::Unpaired {
            hedge: invh,
            flow: -flow,
        };
        Ok(())
    }

    pub fn inv(&self, hedge: Hedge) -> Hedge {
        self.involution.inv(hedge)
    }

    pub fn flow(&self, hedge: Hedge) -> Flow {
        self.involution.flow(hedge)
    }

    pub fn set_flow(&mut self, hedge: Hedge, flow: Flow) {
        if self.flow(hedge) != flow {
            let edge_id = self[&hedge];
            self[&edge_id].1.swap();
            match flow {
                Flow::Source => {
                    self.involution.set_as_source(hedge);
                }
                Flow::Sink => {
                    self.involution.set_as_sink(hedge);
                }
            }
        }
    }

    pub fn superficial_hedge_orientation(&self, hedge: Hedge) -> Option<Flow> {
        match self.involution.hedge_data(hedge) {
            InvolutiveMapping::Identity { data, underlying } => {
                data.orientation.relative_to(*underlying).try_into().ok()
            }
            InvolutiveMapping::Sink { source_idx } => self
                .superficial_hedge_orientation(*source_idx)
                .map(Neg::neg),
            InvolutiveMapping::Source { data, .. } => data.orientation.try_into().ok(),
        }
    }

    pub fn underlying_hedge_orientation(&self, hedge: Hedge) -> Flow {
        match self.involution.hedge_data(hedge) {
            InvolutiveMapping::Identity { underlying, .. } => *underlying,
            InvolutiveMapping::Sink { .. } => Flow::Sink,
            InvolutiveMapping::Source { .. } => Flow::Source,
        }
    }

    pub fn inv_full(&self, hedge: Hedge) -> &InvolutiveMapping<EdgeIndex> {
        self.involution.hedge_data(hedge)
    }

    pub fn edge_len(&self) -> usize {
        self.data.len()
    }

    pub fn iter_edges<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&'a T>)> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.involution.smart_data(i, subgraph).map(|d| {
                (
                    HedgePair::from_half_edge_with_subgraph(i, &self.involution, subgraph).unwrap(),
                    d.data,
                    d.as_ref().map(|&a| &self[a]),
                )
            })
        })
    }

    pub fn iter_all_edges(&self) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&T>)> {
        self.involution
            .iter_edge_data()
            .map(move |(i, d)| (self[&self[&i]].1, d.data, d.as_ref().map(|&a| &self[a])))
    }

    pub fn iter_internal_edge_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a T>> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.involution
                .smart_data(i, subgraph)
                .map(|d| d.as_ref().map(|&a| &self[a]))
        })
    }

    pub fn n_internals<S: SubGraph>(&self, subgraph: &S) -> usize {
        self.involution.n_internals(subgraph)
    }

    pub fn hedge_len(&self) -> usize {
        self.involution.len()
    }

    pub fn align_underlying_to_superficial(&mut self) {
        for i in self.involution.iter_idx() {
            let orientation = self.involution.edge_data(i).orientation;
            if let Orientation::Reversed = orientation {
                self.involution.flip_underlying(i);
            }
        }

        self.fix_hedge_pairs(); //EXPENSIVE TODO FIX
    }

    pub fn mapped_from_involution<E>(
        involution: &Involution<E>,
        f: &impl Fn(HedgePair, EdgeData<&E>) -> EdgeData<T>,
    ) -> Self {
        let mut data = Vec::new();

        let involution = involution.as_ref().map_full(|a, d| {
            let new_data = f(a, d);
            let edgeid = EdgeIndex(data.len());
            data.push((new_data.data, a));
            EdgeData::new(edgeid, new_data.orientation)
        });
        SmartHedgeVec { data, involution }
    }
}

impl<T> IntoIterator for SmartHedgeVec<T> {
    type Item = (EdgeIndex, T, HedgePair);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::vec::IntoIter<(T, HedgePair)>>,
        fn((usize, (T, HedgePair))) -> (EdgeIndex, T, HedgePair),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.data
            .into_iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), t.0, t.1))
    }
}

impl<'a, T> IntoIterator for &'a SmartHedgeVec<T> {
    type Item = (EdgeIndex, &'a T, HedgePair);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::slice::Iter<'a, (T, HedgePair)>>,
        fn((usize, &(T, HedgePair))) -> (EdgeIndex, &T, HedgePair),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.data
            .iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), &t.0, t.1))
    }
}

// impl<T> AsRef<Involution> for SmartHedgeVec<T> {
//     fn as_ref(&self) -> &Involution {
//         self.edge_store.as_ref()
//     }
// }

impl<T> IndexMut<EdgeIndex> for SmartHedgeVec<T> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.data[index.0].0
    }
}
impl<T> Index<EdgeIndex> for SmartHedgeVec<T> {
    type Output = T;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.data[index.0].0
    }
}

impl<T> Index<&EdgeIndex> for SmartHedgeVec<T> {
    type Output = (T, HedgePair);
    fn index(&self, index: &EdgeIndex) -> &Self::Output {
        &self.data[index.0]
    }
}

impl<T> IndexMut<&EdgeIndex> for SmartHedgeVec<T> {
    fn index_mut(&mut self, index: &EdgeIndex) -> &mut Self::Output {
        &mut self.data[index.0]
    }
}

impl<T> Index<Hedge> for SmartHedgeVec<T> {
    type Output = T;
    fn index(&self, hedge: Hedge) -> &Self::Output {
        let eid = self.involution[hedge];
        &self[eid]
    }
}

impl<T> IndexMut<Hedge> for SmartHedgeVec<T> {
    fn index_mut(&mut self, hedge: Hedge) -> &mut Self::Output {
        let eid = self.involution[hedge];
        &mut self[eid]
    }
}

impl<T> Index<&Hedge> for SmartHedgeVec<T> {
    type Output = EdgeIndex;
    fn index(&self, hedge: &Hedge) -> &Self::Output {
        &self.involution[*hedge]
    }
}

// Data stored once per edge (pair of half-edges or external edge)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HedgeVec<T>(pub(super) Vec<T>);

impl<T> IntoIterator for HedgeVec<T> {
    type Item = (EdgeIndex, T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::vec::IntoIter<T>>,
        fn((usize, T)) -> (EdgeIndex, T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.0
            .into_iter()
            .enumerate()
            .map(|(u, t)| (EdgeIndex(u), t))
    }
}

impl<'a, T> IntoIterator for &'a HedgeVec<T> {
    type Item = (EdgeIndex, &'a T);
    type IntoIter = std::iter::Map<
        std::iter::Enumerate<std::slice::Iter<'a, T>>,
        fn((usize, &T)) -> (EdgeIndex, &T),
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().enumerate().map(|(u, t)| (EdgeIndex(u), t))
    }
}

impl<T> IndexMut<EdgeIndex> for HedgeVec<T> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.0[index.0]
    }
}
impl<T> Index<EdgeIndex> for HedgeVec<T> {
    type Output = T;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.0[index.0]
    }
}

impl<T> HedgeVec<T> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn map_ref<O>(&self, f: &impl Fn(&T) -> O) -> HedgeVec<O> {
        HedgeVec(self.0.iter().map(f).collect())
    }

    pub fn get_raw(self) -> Vec<T> {
        self.0
    }
}
