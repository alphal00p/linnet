use std::ops::{Index, IndexMut, Neg};

use bitvec::vec::BitVec;

use crate::permutation::Permutation;

use super::{
    involution::{
        EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Involution, InvolutionError,
        InvolutiveMapping, Orientation,
    },
    subgraph::SubGraph,
    HedgeGraph, HedgeGraphError, NodeStorage,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// A specialized vector-like structure for storing edge data along with its
/// topological [`Involution`].
///
/// `SmartHedgeVec` is a core component for managing edges in a [`HedgeGraph`].
/// It stores the custom edge data (`T`) and the [`HedgePair`] for each edge.
/// The associated `Involution` within `SmartHedgeVec` then stores `EdgeIndex` values,
/// which point back into this `SmartHedgeVec`'s `data` vector. This design
/// separates the topological mapping of half-edges (handled by `Involution`)
/// from the storage of the actual edge data and pairing information.
///
/// # Type Parameters
///
/// - `T`: The type of custom data associated with each edge.
pub struct SmartHedgeVec<T> {
    /// A vector where each element is a tuple containing the custom edge data (`T`)
    /// and the [`HedgePair`] that describes the topological state of the edge
    /// (e.g., paired, unpaired, split). The index into this vector serves as
    /// an `EdgeIndex`.
    pub(super) data: Vec<(T, HedgePair)>,
    /// The [`Involution`] structure that manages the topological relationships of half-edges.
    /// In this context, the `Involution` stores `EdgeIndex` as its data, pointing back
    /// to the `data` vector of this `SmartHedgeVec`.
    involution: Involution,
}

impl<T> AsRef<Involution> for SmartHedgeVec<T> {
    fn as_ref(&self) -> &Involution {
        &self.involution
    }
}

/// A trait providing a generic interface for accessing properties of edges
/// (like orientation, data, and pairing information) using various index types.
///
/// This allows functions to be generic over how they access edge details,
/// whether by [`Hedge`], [`EdgeIndex`], or [`HedgePair`].
pub trait Accessors<Index> {
    /// The type of the primary data associated with the edge.
    type Data;

    /// Returns the [`Orientation`] of the edge identified by `index`.
    fn orientation(&self, index: Index) -> Orientation;
    /// Sets the [`Orientation`] of the edge identified by `index`.
    fn set_orientation(&mut self, index: Index, orientation: Orientation);
    /// Returns the [`HedgePair`] describing the pairing status of the edge/half-edge identified by `index`.
    fn pair(&self, index: Index) -> HedgePair;
    // fn flow(&self, index: Index) -> Flow; // Example of other potential accessors
    // fn set_flow(&mut self, index: Index, flow: Flow);
    /// Returns a reference to the custom data of the edge identified by `index`.
    fn data(&self, index: Index) -> &Self::Data;
    /// Returns a mutable reference to the custom data of the edge identified by `index`.
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
        mut edge_map: impl FnMut(
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

    pub fn map_data_ref_mut<'a, T2>(
        &'a mut self,
        mut edge_map: impl FnMut(EdgeIndex, HedgePair, EdgeData<&'a mut T>) -> EdgeData<T2>,
    ) -> SmartHedgeVec<T2> {
        let mut involution = self.involution.clone();
        SmartHedgeVec {
            data: self
                .data
                .iter_mut()
                .enumerate()
                .map(|(i, (e, h))| {
                    let new_edgedata = edge_map(
                        EdgeIndex(i),
                        *h,
                        EdgeData::new(e, self.involution.orientation(h.any_hedge())),
                    );

                    involution.edge_data_mut(h.any_hedge()).orientation = new_edgedata.orientation;
                    (new_edgedata.data, *h)
                })
                .collect(),
            involution,
        }
    }

    pub fn map_data_ref_result<'a, T2, V, N: NodeStorage<NodeData = V>, Er>(
        &'a self,
        graph: &'a HedgeGraph<T, V, N>,
        mut edge_map: impl FnMut(
            &'a HedgeGraph<T, V, N>,
            EdgeIndex,
            HedgePair,
            EdgeData<&'a T>,
        ) -> Result<EdgeData<T2>, Er>,
    ) -> Result<SmartHedgeVec<T2>, Er> {
        let mut involution = self.involution.clone();
        let data: Result<Vec<_>, Er> = self
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

                match new_edgedata {
                    Ok(new_edgedata) => {
                        involution.edge_data_mut(h.any_hedge()).orientation =
                            new_edgedata.orientation;
                        Ok((new_edgedata.data, *h))
                    }
                    Err(err) => Err(err),
                }
            })
            .collect();
        Ok(SmartHedgeVec {
            data: data?,
            involution,
        })
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

    pub fn delete<S: SubGraph>(&mut self, graph: &S) {
        let mut left = Hedge(0);
        let mut extracted = Hedge(self.involution.len());
        while left < extracted {
            if !graph.includes(&left) {
                //left is in the right place
                left.0 += 1;
            } else {
                //left needs to be swapped
                extracted.0 -= 1;
                if !graph.includes(&extracted) {
                    // println!("{extracted}<=>{left}");
                    //only with an extracted that is in the wrong spot
                    self.swap_hedges(left, extracted);
                    left.0 += 1;
                }
            }
        }

        for i in 0..(left.0) {
            if self.inv(Hedge(i)) >= left {
                let flow = if self.involution.set_as_source(Hedge(i)) {
                    Flow::Sink
                } else {
                    Flow::Source
                };

                // println!("Split:{i}");
                self.data[self.involution[Hedge(i)].0].1 = HedgePair::Unpaired {
                    hedge: Hedge(i),
                    flow,
                };

                self.involution
                    .source_to_identity_impl(Hedge(i), flow, flow == Flow::Sink);
            }
        }
        // println!("left:{left}");
        // println!("{}", self.involution.display());

        // The self involution is good and valid, but we now need to split the data carrying vec.

        let mut split_at = 0;

        let mut extracted = self.edge_len();

        while split_at < extracted {
            if self.data[split_at].1.any_hedge() < left {
                //left is in the right place
                split_at += 1;
            } else {
                //left needs to be swapped
                extracted -= 1;
                if self.data[extracted].1.any_hedge() < left {
                    // println!("{extracted}<=>{split_at}");
                    //only with an extracted that is in the wrong spot
                    self.swap_edges(EdgeIndex(split_at), EdgeIndex(extracted));
                    // println!("{}", self.involution.display());

                    split_at += 1;
                }
            }
        }

        // println!("{}", self.involution.display());

        let _ = self.involution.inv.split_off(left.0);
        let _ = self.data.split_off(split_at);
        // self.fix_hedge_pairs();
    }

    fn swap_edges(&mut self, e1: EdgeIndex, e2: EdgeIndex) {
        if e1 != e2 {
            let a = &mut self
                .involution
                .edge_data_mut(self.data[e1.0].1.any_hedge())
                .data;

            // println!("{a}{e1}");
            *a = e1;

            // = e1;
            self.involution
                .edge_data_mut(self.data[e2.0].1.any_hedge())
                .data = e1;
            self.data.swap(e1.0, e2.0);
        }
    }

    fn swap_hedges(&mut self, e1: Hedge, e2: Hedge) {
        if e1 != e2 {
            if e1 == self.inv(e1) {
                match &mut self.data[self.involution[e1].0].1 {
                    HedgePair::Paired { source, sink } => {
                        std::mem::swap(source, sink);
                    }
                    HedgePair::Split { source, sink, .. } => {
                        std::mem::swap(source, sink);
                    }
                    _ => {}
                };
            } else {
                match &mut self.data[self.involution[e1].0].1 {
                    HedgePair::Split { source, sink, .. } | HedgePair::Paired { source, sink } => {
                        if *source == e1 {
                            *source = e1;
                        } else if *source == e2 {
                            *source = e2;
                        }
                        if *sink == e1 {
                            *sink = e1;
                        } else if *sink == e2 {
                            *sink = e2;
                        }
                    }
                    HedgePair::Unpaired { hedge, .. } => {
                        if *hedge == e1 {
                            *hedge = e1;
                        } else if *hedge == e2 {
                            *hedge = e2;
                        }
                    }
                };

                match &mut self.data[self.involution[e2].0].1 {
                    HedgePair::Split { source, sink, .. } | HedgePair::Paired { source, sink } => {
                        if *source == e1 {
                            *source = e1;
                        } else if *source == e2 {
                            *source = e2;
                        }
                        if *sink == e1 {
                            *sink = e1;
                        } else if *sink == e2 {
                            *sink = e2;
                        }
                    }
                    HedgePair::Unpaired { hedge, .. } => {
                        if *hedge == e1 {
                            *hedge = e1;
                        } else if *hedge == e2 {
                            *hedge = e2;
                        }
                    }
                };
            }
            self.involution.swap(e1, e2);
            self.fix_hedge_pairs();
        }
    }

    pub fn extract<S: SubGraph, O>(
        &mut self,
        graph: &S,
        mut split_edge_fn: impl FnMut(EdgeData<&T>) -> EdgeData<O>,
        mut internal_data: impl FnMut(EdgeData<T>) -> EdgeData<O>,
    ) -> SmartHedgeVec<O> {
        let mut new_id_data = vec![];
        let mut extracted = self.involution.extract(
            graph,
            |a| {
                let new_id = -(1 + new_id_data.len() as i64);
                let new_data = split_edge_fn(EdgeData::new(&self.data[a.data.0].0, a.orientation));
                new_id_data.push((
                    new_data.data,
                    HedgePair::Unpaired {
                        hedge: Hedge(0),
                        flow: Flow::Sink,
                    },
                ));
                EdgeData::new(new_id, new_data.orientation)
            },
            |a| a.map(|e| e.0 as i64),
        );

        // The self involution is good and valid, but we now need to split the data carrying vec.

        let mut split_at = 0;
        let mut other_len = 0;
        let data_pos: Vec<_> = self
            .involution
            .iter_edge_data_mut()
            .map(|data| {
                let old_data = data.data.0;
                data.data.0 = split_at;
                split_at += 1;
                old_data
            })
            .chain(extracted.iter_edge_data_mut().filter_map(|i| {
                if i.data >= 0 {
                    let old_data = i.data as usize;
                    i.data = other_len;
                    other_len += 1;
                    Some(old_data)
                } else {
                    None
                }
            }))
            .collect();

        let perm = Permutation::from_map(data_pos);
        perm.apply_slice_in_place_inv(&mut self.data);
        let mut new_data: Vec<Option<T>> = self
            .data
            .split_off(split_at)
            .into_iter()
            .map(|(d, _)| Some(d))
            .collect();
        let shift = new_data.len();
        let mut new_data_mapped = vec![];

        extracted.iter_edge_data_mut().for_each(|d| {
            if d.data >= 0 {
                let new_data = internal_data(EdgeData::new(
                    new_data[d.data as usize].take().unwrap(),
                    d.orientation,
                ));

                new_data_mapped.push((
                    new_data.data,
                    HedgePair::Unpaired {
                        hedge: Hedge(0),
                        flow: Flow::Sink,
                    },
                ));
                d.orientation = new_data.orientation;
            } else {
                d.data = shift as i64 + d.data.abs() - 1;
                // println!("{}", d.data);
            }
        });
        new_data_mapped.extend(new_id_data);
        let new_inv = extracted.map_data(&|e| EdgeIndex(e as usize));
        for (i, d) in new_inv.iter_edge_data() {
            let hedge_pair = new_inv.hedge_pair(i);
            // println!("{}", d.data.0);
            new_data_mapped[d.data.0].1 = hedge_pair;
        }
        self.fix_hedge_pairs();
        SmartHedgeVec {
            data: new_data_mapped,
            involution: new_inv,
        }
    }

    pub(crate) fn join_mut(
        &mut self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&T>, Flow, EdgeData<&T>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<T>, Flow, EdgeData<T>) -> (Flow, EdgeData<T>),
    ) -> Result<(), HedgeGraphError> {
        let self_empty_filter = BitVec::empty(self.hedge_len());
        let mut full_self = !self_empty_filter.clone();
        let other_empty_filter = BitVec::empty(other.hedge_len());
        let mut full_other = self_empty_filter.clone();
        full_self.extend(other_empty_filter.clone());
        full_other.extend(!other_empty_filter.clone());

        let self_inv_shift = self.hedge_len();
        let edge_data = &mut self.data;
        let edge_data_shift = edge_data.len();
        edge_data.extend(other.data);

        self.involution
            .inv
            .extend(other.involution.into_iter().map(|(_, m)| match m {
                InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink {
                    source_idx: Hedge(source_idx.0 + self_inv_shift),
                },
                InvolutiveMapping::Source { data, sink_idx } => InvolutiveMapping::Source {
                    data: data.map(|e| EdgeIndex(e.0 + edge_data_shift)),
                    sink_idx: Hedge(sink_idx.0 + self_inv_shift),
                },
                InvolutiveMapping::Identity { data, underlying } => InvolutiveMapping::Identity {
                    data: data.map(|e| EdgeIndex(e.0 + edge_data_shift)),
                    underlying,
                },
            }));

        let mut found_match = true;

        self.fix_hedge_pairs();

        while found_match {
            let mut matching_ids = None;

            for i in full_self.included_iter() {
                if let InvolutiveMapping::Identity {
                    data: datas,
                    underlying: underlyings,
                } = self.involution.hedge_data(i)
                {
                    for j in full_other.included_iter() {
                        if let InvolutiveMapping::Identity { data, underlying } =
                            &self.involution.inv[j.0]
                        {
                            if matching_fn(
                                *underlyings,
                                datas.as_ref().map(|a| &self.data[a.0].0),
                                *underlying,
                                data.as_ref().map(|a| &self.data[a.0].0),
                            ) {
                                matching_ids = Some((i, j));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((source, sink)) = matching_ids {
                self.connect_identities(source, sink, &merge_fn);
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
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// A simple wrapper around a `Vec<T>` for storing data associated with each edge,
/// where edges are identified by an [`EdgeIndex`].
///
/// This provides a more direct way to store per-edge data compared to
/// [`SmartHedgeVec`], if the detailed pairing information (`HedgePair`) and
/// tight integration with `Involution`'s internal structure are not needed
/// directly alongside the data.
///
/// # Type Parameters
///
/// - `T`: The type of data to be stored for each edge.
pub struct HedgeVec<T>(
    /// The underlying vector storing the edge data. The index in this vector
    /// corresponds to an `EdgeIndex`.
    pub(super) Vec<T>,
);

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

    pub fn from_raw(data: Vec<T>) -> Self {
        HedgeVec(data)
    }
}
