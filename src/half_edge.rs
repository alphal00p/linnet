use core::panic;
use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::Hash;
use std::num::TryFromIntError;
use std::ops::{Index, IndexMut};

use ahash::{AHashMap, AHashSet};

use bitvec::prelude::*;
use bitvec::{slice::IterOnes, vec::BitVec};
use builder::HedgeGraphBuilder;
use derive_more::{From, Into};
use hedgevec::{Accessors, HedgeVec, SmartHedgeVec};
use indexmap::IndexSet;
use involution::{
    EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Involution, InvolutionError, InvolutiveMapping,
    Orientation,
};
use itertools::Itertools;
use nodestore::{NodeStorage, NodeStorageOps, NodeStorageVec};
use rand::{rngs::SmallRng, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, From, Into, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct NodeIndex(pub usize);

impl std::fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Iterator over the powerset of a bitvec, of size n < 64.
pub struct PowersetIterator {
    size: usize,
    current: usize,
}

impl PowersetIterator {
    pub fn new(n_elements: u8) -> Self {
        PowersetIterator {
            size: 1 << n_elements,
            current: 0,
        }
    }
}

impl Iterator for PowersetIterator {
    type Item = BitVec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.size {
            let out = BitVec::<_, Lsb0>::from_element(self.current);
            self.current += 1;
            Some(out)
        } else {
            None
        }
    }
}
pub mod involution;

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct GVEdgeAttrs {
    pub label: Option<String>,
    pub color: Option<String>,
    pub other: Option<String>,
}

impl std::fmt::Display for GVEdgeAttrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = [
            ("label=", self.label.as_ref()),
            (
                "color=",
                self.color
                    .as_ref()
                    .map(|str| format!("\"{}\"", str))
                    .as_ref(),
            ),
            ("", self.other.as_ref()),
        ]
        .iter()
        .filter_map(|(prefix, x)| x.map(|s| format!("{}{}", prefix, s)))
        .join(",")
        .to_string();
        write!(f, "{}", out)
    }
}
pub mod builder;
pub mod nodestore;
pub mod subgraph;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HedgeGraph<E, V, S: NodeStorage<NodeData = V> = NodeStorageVec<V>> {
    edge_store: SmartHedgeVec<E>,
    pub node_store: S,
}

#[cfg(feature = "bincode")]
impl<E, V, S: NodeStorage<NodeData = V>> ::bincode::Encode for HedgeGraph<E, V, S>
where
    E: ::bincode::Encode,
    V: ::bincode::Encode,
    S: ::bincode::Encode,
{
    fn encode<__E: ::bincode::enc::Encoder>(
        &self,
        encoder: &mut __E,
    ) -> core::result::Result<(), ::bincode::error::EncodeError> {
        ::bincode::Encode::encode(&self.edge_store, encoder)?;
        ::bincode::Encode::encode(&self.node_store, encoder)?;
        core::result::Result::Ok(())
    }
}

#[cfg(feature = "bincode")]
impl<E, V, S: NodeStorage<NodeData = V>, __Context> ::bincode::Decode<__Context>
    for HedgeGraph<E, V, S>
where
    E: ::bincode::Decode<__Context>,
    V: ::bincode::Decode<__Context>,
    S: ::bincode::Decode<__Context>,
{
    fn decode<__D: ::bincode::de::Decoder<Context = __Context>>(
        decoder: &mut __D,
    ) -> core::result::Result<Self, ::bincode::error::DecodeError> {
        core::result::Result::Ok(Self {
            edge_store: ::bincode::Decode::decode(decoder)?,
            node_store: ::bincode::Decode::decode(decoder)?,
        })
    }
}

#[cfg(feature = "bincode")]
impl<'__de, E, V, S: NodeStorage<NodeData = V>, __Context> ::bincode::BorrowDecode<'__de, __Context>
    for HedgeGraph<E, V, S>
where
    E: ::bincode::de::BorrowDecode<'__de, __Context>,
    V: ::bincode::de::BorrowDecode<'__de, __Context>,
    S: ::bincode::de::BorrowDecode<'__de, __Context>,
{
    fn borrow_decode<__D: ::bincode::de::BorrowDecoder<'__de, Context = __Context>>(
        decoder: &mut __D,
    ) -> core::result::Result<Self, ::bincode::error::DecodeError> {
        core::result::Result::Ok(Self {
            edge_store: ::bincode::BorrowDecode::<'_, __Context>::borrow_decode(decoder)?,
            node_store: ::bincode::BorrowDecode::<'_, __Context>::borrow_decode(decoder)?,
        })
    }
}

impl<E, V, S: NodeStorage<NodeData = V>> AsRef<Involution> for HedgeGraph<E, V, S> {
    fn as_ref(&self) -> &Involution {
        self.edge_store.as_ref()
    }
}

impl<E, V: Default, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    /// Creates a random graph with the given number of nodes and edges.
    pub fn random(nodes: usize, edges: usize, seed: u64) -> HedgeGraph<(), V, N>
    where
        N::Neighbors: BaseSubgraph + SubGraphOps,
    {
        let inv: Involution<()> = Involution::<()>::random(edges, seed);

        let mut rng = SmallRng::seed_from_u64(seed);

        let mut externals = Vec::new();
        let mut sources = Vec::new();
        let mut sinks = Vec::new();

        for (i, e) in inv.iter() {
            let mut nodeid = N::Neighbors::empty(inv.len());
            nodeid.add(i);
            match e {
                InvolutiveMapping::Identity { .. } => externals.push(nodeid),
                InvolutiveMapping::Source { .. } => sources.push(nodeid),
                InvolutiveMapping::Sink { .. } => sinks.push(nodeid),
            }
        }

        assert_eq!(sources.len(), sinks.len());

        while !externals.is_empty() {
            if rng.gen_bool(0.5) {
                let source_i = rng.gen_range(0..sources.len());

                sources[source_i].union_with(&externals.pop().unwrap());
            } else {
                let sink_i = rng.gen_range(0..sinks.len());

                sinks[sink_i].union_with(&externals.pop().unwrap());
            }
        }

        let mut lengthone = false;

        while sources.len() + sinks.len() > nodes {
            if rng.gen_bool(0.5) {
                if sources.len() <= 1 {
                    if lengthone {
                        break;
                    }
                    lengthone = true;
                    continue;
                }

                let idx1 = rng.gen_range(0..sources.len());
                let idx2 = rng.gen_range(0..sources.len() - 1);

                let n_i = sources.swap_remove(idx1);
                sources[idx2].union_with(&n_i);
            } else {
                if sinks.len() <= 1 {
                    if lengthone {
                        break;
                    }
                    lengthone = true;
                    continue;
                }
                let idx1 = rng.gen_range(0..sinks.len());
                let idx2 = rng.gen_range(0..sinks.len() - 1);
                let n_i = sinks.swap_remove(idx1);
                sinks[idx2].union_with(&n_i);
            }
        }

        HedgeGraph {
            node_store: N::random(&sources, &sinks),
            edge_store: SmartHedgeVec::new(inv),
        }
    }
}

impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn delete_hedges<S: SubGraph<Base = N::Base>>(&mut self, subgraph: &S) {
        self.edge_store.delete(subgraph);
        self.node_store.delete(subgraph);
    }

    pub fn concretize<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> HedgeGraph<&'a E, &'a V, N::OpStorage<&'a V>> {
        let mut builder = HedgeGraphBuilder::new();

        let mut node_map = AHashMap::new();

        for (n, _, d) in self.iter_node_data(subgraph) {
            node_map.insert(n, builder.add_node(d));
        }

        for (pair, _, d) in self.iter_edges(subgraph) {
            match pair {
                HedgePair::Paired { source, sink } => {
                    let src = node_map[&self.node_id(source)];
                    let dst = node_map[&self.node_id(sink)];

                    builder.add_edge(src, dst, d.data, d.orientation);
                }
                HedgePair::Unpaired { hedge, flow } => {
                    let src = node_map[&self.node_id(hedge)];

                    builder.add_external_edge(src, d.data, d.orientation, flow);
                }
                HedgePair::Split {
                    source,
                    sink,
                    split,
                } => match split {
                    Flow::Sink => {
                        let src = node_map[&self.node_id(sink)];
                        builder.add_external_edge(src, d.data, d.orientation, split);
                    }
                    Flow::Source => {
                        let src = node_map[&self.node_id(source)];
                        builder.add_external_edge(src, d.data, d.orientation, split);
                    }
                },
            }
        }

        builder.build()
    }

    pub fn extract<O, V2, S: SubGraph<Base = N::Base>>(
        &mut self,
        subgraph: &S,
        split_edge_fn: impl FnMut(EdgeData<&E>) -> EdgeData<O>,
        internal_data: impl FnMut(EdgeData<E>) -> EdgeData<O>,
        split_node: impl FnMut(&V) -> V2,
        owned_node: impl FnMut(V) -> V2,
    ) -> HedgeGraph<O, V2, N::OpStorage<V2>> {
        let new_edge_store = self
            .edge_store
            .extract(subgraph, split_edge_fn, internal_data);

        let new_node_store = self.node_store.extract(subgraph, split_node, owned_node);
        HedgeGraph {
            node_store: new_node_store,
            edge_store: new_edge_store,
        }
    }

    /// Gives the involved hedge.
    /// If the hedge is a source, it will return the sink, and vice versa.
    /// If the hedge is an identity, it will return Itself.
    pub fn inv(&self, hedge: Hedge) -> Hedge {
        self.edge_store.inv(hedge)
    }

    /// Splits the edge that hedge is a part of into two dangling hedges, adding the data to the side given by hedge.
    /// The underlying orientation of the new edges is the same as the original edge, i.e. the source will now have `Flow::Source` and the sink will have `Flow::Sink`.
    /// The superficial orientation has to be given knowing this.
    pub fn split_edge(&mut self, hedge: Hedge, data: EdgeData<E>) -> Result<(), HedgeGraphError> {
        Ok(self.edge_store.split_edge(hedge, data)?)
    }

    /// Joins two graphs together, matching edges with the given function and merging them with the given function.
    /// The function `matching_fn` should return true if the two dangling half edges should be matched.
    /// The function `merge_fn` should return the new data for the merged edge, given the data of the two edges being merged.
    pub fn join(
        self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&E>, Flow, EdgeData<&E>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<E>, Flow, EdgeData<E>) -> (Flow, EdgeData<E>),
    ) -> Result<Self, HedgeGraphError> {
        let mut g = HedgeGraph {
            node_store: self.node_store.extend(other.node_store),
            edge_store: self
                .edge_store
                .join(other.edge_store, matching_fn, merge_fn)?,
        };
        g.node_store.check_and_set_nodes()?;

        Ok(g)
    }

    pub fn join_mut(
        &mut self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&E>, Flow, EdgeData<&E>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<E>, Flow, EdgeData<E>) -> (Flow, EdgeData<E>),
    ) -> Result<(), HedgeGraphError> {
        self.node_store.extend_mut(other.node_store);
        self.edge_store
            .join_mut(other.edge_store, matching_fn, merge_fn)?;

        self.node_store.check_and_set_nodes()?;
        Ok(())
    }

    /// Sews dangling edges internal to the graph, matching edges with the given function and merging them with the given function.
    /// The function `matching_fn` should return true if the two dangling half edges should be matched.
    /// The function `merge_fn` should return the new data for the merged edge, given the data of the two edges being merged.
    pub fn sew(
        &mut self,
        matching_fn: impl Fn(Flow, EdgeData<&E>, Flow, EdgeData<&E>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<E>, Flow, EdgeData<E>) -> (Flow, EdgeData<E>),
    ) -> Result<(), HedgeGraphError> {
        self.edge_store.sew(matching_fn, merge_fn)
    }

    /// Adds a dangling edge to the specified node with the given data and superficial orientation.
    pub fn add_dangling_edge(
        self,
        source: NodeIndex,
        data: E,
        flow: Flow,
        orientation: impl Into<Orientation>,
    ) -> Result<(Hedge, Self), HedgeGraphError> {
        let (edge_store, hedge) = self.edge_store.add_dangling_edge(data, flow, orientation);
        let mut g = HedgeGraph {
            edge_store,
            node_store: self.node_store.add_dangling_edge(source)?,
        };

        g.node_store.check_and_set_nodes()?;

        Ok((hedge, g))
    }

    /// Adds a paired edge to the specified nodes with the given data and superficial orientation.
    pub fn add_pair(
        self,
        source: NodeIndex,
        sink: NodeIndex,
        data: E,
        orientation: impl Into<Orientation>,
    ) -> Result<(Hedge, Hedge, Self), HedgeGraphError> {
        let (edge_store, sourceh, sinkh) = self.edge_store.add_paired(data, orientation);
        let mut g = HedgeGraph {
            edge_store,
            node_store: self
                .node_store
                .add_dangling_edge(source)?
                .add_dangling_edge(sink)?,
        };

        g.node_store.check_and_set_nodes()?;

        Ok((sourceh, sinkh, g))
    }

    /// Is the graph connected?
    pub fn is_connected<S: SubGraph>(&self, subgraph: &S) -> bool {
        let n_edges = subgraph.nedges(self);
        if let Some(start) = subgraph.included_iter().next() {
            SimpleTraversalTree::depth_first_traverse(self, subgraph, &self.node_id(start), None)
                .unwrap()
                .covers(subgraph)
                .nedges(self)
                == n_edges
        } else {
            true
        }
    }

    pub fn cut_branches(&self, subgraph: &mut HedgeNode) {
        let nodes = AHashSet::<NodeIndex>::from_iter(
            subgraph
                .internal_graph
                .included_iter()
                .map(|i| self.node_id(i)),
        );
        self.remove_externals(subgraph);

        let mut has_branch = true;
        while has_branch {
            has_branch = false;

            for n in &nodes {
                let int = self.sub_iter_crown(*n, subgraph).collect::<Vec<_>>();
                let first = int.first();
                let next = int.get(1);

                if let Some(first) = first {
                    if next.is_none() {
                        subgraph.internal_graph.filter.set(first.0, false);
                        subgraph
                            .internal_graph
                            .filter
                            .set(self.inv(*first).0, false);
                        has_branch = true;
                    }
                }
            }
        }

        self.nesting_node_fix(subgraph);
    }

    // fn _set_hedge_data(&mut self, hedge: Hedge, nodeid: NodeIndex) {
    //     self.node_store.set_hedge_data(hedge, nodeid);
    // }
}

// Subgraphs
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn paired_filter_from_pos(&self, pos: &[Hedge]) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.n_hedges()];

        for &i in pos {
            filter.set(i.0, true);
            filter.set(self.inv(i).0, true);
        }

        filter
    }

    /// Bitvec subgraph of all external (identity) hedges
    pub fn external_filter(&self) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.n_hedges()];

        for (i, _, _) in self.iter_all_edges() {
            if i.is_unpaired() {
                filter.set(i.any_hedge().0, true);
            }
        }

        filter
    }

    /// Bitvec subgraph of all hedges
    pub fn full_filter(&self) -> BitVec {
        bitvec![usize, Lsb0; 1; self.n_hedges()]
    }

    /// Get a hairless subgraph, deleting all hairs from the subgraph
    pub fn clean_subgraph(&self, filter: BitVec) -> InternalSubGraph {
        InternalSubGraph::cleaned_filter_pessimist(filter, self)
    }

    /// Get a HedgeNode subgraph that covers the whole graph
    pub fn full_node(&self) -> HedgeNode {
        self.nesting_node_from_subgraph(self.full_graph())
    }

    /// Get a internal subgraph that covers the whole internal edges of the graph
    pub fn full_graph(&self) -> InternalSubGraph {
        InternalSubGraph::cleaned_filter_optimist(self.full_filter(), self)
    }

    /// Get a generic empty subgraph
    pub fn empty_subgraph<S: SubGraph>(&self) -> S {
        S::empty(self.n_hedges())
    }

    pub fn from_filter<S: BaseSubgraph>(&self, filter: impl FnMut(&E) -> bool) -> S {
        S::from_filter(self, filter)
    }

    pub fn nesting_node_from_subgraph(&self, internal_graph: InternalSubGraph) -> HedgeNode {
        let mut hairs = bitvec![usize, Lsb0; 0; self.n_hedges()];

        if !internal_graph.valid::<E, V, N>(self) {
            panic!("Invalid subgraph")
        }

        for i in internal_graph.included_iter() {
            hairs.union_with_iter(self.neighbors(i));
        }

        HedgeNode {
            hairs: !(!hairs | &internal_graph.filter),
            internal_graph,
        }
    }

    pub fn remove_internal_hedges(&self, subgraph: &BitVec) -> BitVec {
        let mut hairs = subgraph.clone();
        for i in subgraph.included_iter() {
            if subgraph.includes(&self.inv(i)) {
                hairs.set(i.0, false);
                hairs.set(self.inv(i).0, false);
            }
        }
        hairs
    }

    pub(crate) fn split_hairs_and_internal_hedges(
        &self,
        mut subgraph: BitVec,
    ) -> (BitVec, InternalSubGraph) {
        let mut internal: InternalSubGraph = self.empty_subgraph();
        for i in subgraph.included_iter() {
            let invh = self.inv(i);
            if subgraph.includes(&invh) {
                internal.filter.set(i.0, true);
                internal.filter.set(invh.0, true);
            }
        }
        for i in internal.filter.included_iter() {
            subgraph.set(i.0, false);
        }
        (subgraph, internal)
    }

    fn nesting_node_fix(&self, node: &mut HedgeNode) {
        let mut externalhedges = bitvec![usize, Lsb0; 0; self.n_hedges()];

        for i in node.internal_graph.filter.included_iter() {
            externalhedges.union_with_iter(self.neighbors(i));
        }

        node.hairs = !(!externalhedges | &node.internal_graph.filter);
    }

    fn remove_externals(&self, subgraph: &mut HedgeNode) {
        let externals = self.external_filter();

        subgraph.internal_graph.filter &= !externals;
    }
}

// Counts
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn count_internal_edges<S: SubGraph>(&self, subgraph: &S) -> usize {
        let mut internal_edge_count = 0;
        // Iterate over all half-edges in the subgraph
        for hedge_index in subgraph.included_iter() {
            let inv_hedge_index = self.inv(hedge_index);

            // Check if the involuted half-edge is also in the subgraph
            if subgraph.includes(&inv_hedge_index) {
                // To avoid double-counting, only count when hedge_index < inv_hedge_index
                if hedge_index < inv_hedge_index {
                    internal_edge_count += 1;
                }
            }
        }
        internal_edge_count
    }

    pub fn n_hedges(&self) -> usize {
        self.edge_store.hedge_len()
    }

    pub fn n_nodes(&self) -> usize {
        self.node_store.node_len()
    }

    pub fn n_externals(&self) -> usize {
        self.edge_store.n_dangling()
    }

    pub fn n_internals(&self) -> usize {
        self.edge_store.n_paired()
    }

    // pub fn n_base_nodes(&self) -> usize {
    //     self.nodes.iter().filter(|n| n.is_node()).count()
    // }

    pub fn number_of_nodes_in_subgraph<S: SubGraph>(&self, subgraph: &S) -> usize {
        self.iter_node_data(subgraph).count()
    }

    pub fn node_degrees_in_subgraph(
        &self,
        subgraph: &InternalSubGraph,
    ) -> AHashMap<NodeIndex, usize> {
        let mut degrees = AHashMap::new();

        for (_, node, _) in self.iter_node_data(subgraph) {
            let node_pos = self.id_from_crown(node).unwrap();

            // Count the number of edges in the subgraph incident to this node
            let incident_edges =
                BitVec::from_hedge_iter(self.sub_iter_crown(node_pos, subgraph), subgraph.size());
            let degree = incident_edges.count_ones();

            degrees.insert(node_pos, degree);
        }

        degrees
    }
}

pub trait EdgeAccessors<Index> {
    fn orientation(&self, index: Index) -> Orientation;

    fn set_orientation(&mut self, index: Index, orientation: Orientation);
}

impl<E, V, N: NodeStorageOps<NodeData = V>> EdgeAccessors<Hedge> for HedgeGraph<E, V, N> {
    fn orientation(&self, index: Hedge) -> Orientation {
        self.edge_store.orientation(index)
    }

    fn set_orientation(&mut self, index: Hedge, orientation: Orientation) {
        self.edge_store.set_orientation(index, orientation);
    }
}

impl<E, V, N: NodeStorageOps<NodeData = V>> EdgeAccessors<HedgePair> for HedgeGraph<E, V, N> {
    fn orientation(&self, index: HedgePair) -> Orientation {
        self.edge_store.orientation(index)
    }

    fn set_orientation(&mut self, index: HedgePair, orientation: Orientation) {
        self.edge_store.set_orientation(index, orientation);
    }
}

impl<E, V, N: NodeStorageOps<NodeData = V>> EdgeAccessors<EdgeIndex> for HedgeGraph<E, V, N> {
    fn orientation(&self, index: EdgeIndex) -> Orientation {
        self.edge_store.orientation(index)
    }

    fn set_orientation(&mut self, index: EdgeIndex, orientation: Orientation) {
        self.edge_store.set_orientation(index, orientation);
    }
}

// Accessors
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    /// including pos
    pub fn owned_neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> BitVec {
        subgraph.hairs(self.neighbors(pos))
    }

    pub fn connected_neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> Option<BitVec> {
        Some(subgraph.hairs(self.involved_node_crown(pos)?))
    }
    pub fn get_edge_data(&self, edge: Hedge) -> &E {
        &self[self[&edge]]
    }

    pub fn hedge_pair(&self, hedge: Hedge) -> HedgePair {
        self.edge_store[&self.edge_store[&hedge]].1
    }

    pub fn get_edge_data_full(&self, hedge: Hedge) -> EdgeData<&E> {
        let orientation = self.edge_store.orientation(hedge);
        EdgeData::new(&self[self[&hedge]], orientation)
    }

    /// Gives the underlying orientation of this half-edge.
    pub fn flow(&self, hedge: Hedge) -> Flow {
        self.edge_store.flow(hedge)
    }

    pub fn superficial_hedge_orientation(&self, hedge: Hedge) -> Option<Flow> {
        self.edge_store.superficial_hedge_orientation(hedge)
    }

    pub fn underlying_hedge_orientation(&self, hedge: Hedge) -> Flow {
        self.edge_store.underlying_hedge_orientation(hedge)
    }

    pub fn neighbors(&self, hedge: Hedge) -> N::NeighborsIter<'_> {
        self.iter_crown(self.node_id(hedge))
    }

    pub fn iter_crown(&self, id: NodeIndex) -> N::NeighborsIter<'_> {
        self.node_store.get_neighbor_iterator(id)
    }

    pub fn sub_iter_crown<'a, S: SubGraph>(
        &'a self,
        id: NodeIndex,
        subgraph: &'a S,
    ) -> impl Iterator<Item = Hedge> + 'a {
        self.iter_crown(id).filter(|i| subgraph.includes(i))
    }

    pub fn id_from_crown<'a>(&'a self, mut neighbors: N::NeighborsIter<'a>) -> Option<NodeIndex> {
        let e = neighbors.next()?;
        Some(self.node_id(e))
    }

    pub fn involved_node_crown(&self, hedge: Hedge) -> Option<N::NeighborsIter<'_>> {
        self.involved_node_id(hedge).map(|id| self.iter_crown(id))
    }

    pub fn involved_node_id(&self, hedge: Hedge) -> Option<NodeIndex> {
        let invh = self.inv(hedge);
        if invh == hedge {
            return None;
        }
        Some(self.node_id(invh))
    }

    pub fn node_id(&self, hedge: Hedge) -> NodeIndex {
        self.node_store.node_id_ref(hedge)
    }

    pub fn is_self_loop(&self, hedge: Hedge) -> bool {
        !self.is_dangling(hedge) && self.node_id(hedge) == self.node_id(self.inv(hedge))
    }

    pub fn is_dangling(&self, hedge: Hedge) -> bool {
        self.inv(hedge) == hedge
    }

    /// Collect all nodes in the subgraph (all nodes that the hedges are connected to)
    pub fn nodes<S: SubGraph>(&self, subgraph: &S) -> Vec<NodeIndex> {
        let mut nodes = IndexSet::new();
        for i in subgraph.included_iter() {
            let node = self.node_id(i);
            nodes.insert(node);
        }

        nodes.into_iter().collect()
    }

    pub fn set_flow(&mut self, hedge: Hedge, flow: Flow) {
        self.edge_store.set_flow(hedge, flow);
    }

    ///Permutes nodes not pointing to any root anymore to end of nodestore and then extract it
    pub fn forget_identification_history(&mut self) -> Vec<(V, Hedge)> {
        self.node_store.forget_identification_history()
    }

    ///Retains the NodeIndex ordering and just appends a new node.
    pub fn identify_nodes(&mut self, nodes: &[NodeIndex], node_data_merge: V) -> NodeIndex {
        self.node_store.identify_nodes(nodes, node_data_merge)
    }

    ///Retains the NodeIndex ordering and just appends a new node.
    pub fn identify_nodes_without_self_edges<S: BaseSubgraph>(
        &mut self,
        nodes: &[NodeIndex],
        node_data_merge: V,
    ) -> (NodeIndex, S) {
        let mut self_edges: S = self.empty_subgraph();
        for n in nodes {
            for h in self.iter_crown(*n) {
                if self.is_self_loop(h) {
                    self_edges.add(h);
                }
            }
        }
        let n = self.node_store.identify_nodes(nodes, node_data_merge);

        for h in self.iter_crown(n) {
            if self.is_self_loop(h) {
                if self_edges.includes(&h) {
                    self_edges.sub(h);
                } else {
                    self_edges.add(h);
                }
            }
        }

        // self_edges

        (n, self_edges)
    }

    /// Collect all edges in the subgraph
    /// (This is without double counting, i.e. if two half-edges are part of the same edge, only one `EdgeIndex` will be collected)
    pub fn edges<S: SubGraph>(&self, subgraph: &S) -> Vec<EdgeIndex> {
        self.iter_edges(subgraph).map(|(_, i, _)| i).collect()
    }
}

pub enum NodeKind<Data> {
    Internal(Data),
    External { data: Data, flow: Flow },
}

pub enum DanglingMatcher<Data> {
    Actual { hedge: Hedge, data: Data },
    Internal { data: Data },
    Saturator { hedge: Hedge },
}

impl<Data> DanglingMatcher<Data> {
    fn new(pair: HedgePair, data: Data) -> Self {
        match pair {
            HedgePair::Unpaired { hedge, .. } => DanglingMatcher::Actual { hedge, data },
            HedgePair::Paired { .. } => DanglingMatcher::Internal { data },
            _ => panic!("Split"),
        }
    }
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (
                DanglingMatcher::Actual { hedge: h1, .. },
                DanglingMatcher::Saturator { hedge: h2 },
            ) => h1 == h2,
            _ => false,
        }
    }

    pub fn unwrap(self) -> Data {
        match self {
            DanglingMatcher::Actual { data, .. } => data,
            DanglingMatcher::Internal { data } => data,
            DanglingMatcher::Saturator { .. } => panic!("Cannot unwrap a saturator"),
        }
    }
}

// Mapping
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn saturate_dangling<V2>(
        self,
        node_map: impl FnMut(&Involution, NodeIndex, V) -> V2,
        mut dangling_map: impl FnMut(&Involution, &N, Hedge, Flow, EdgeData<&E>) -> V2,
    ) -> HedgeGraph<E, V2, <<N as NodeStorageOps>::OpStorage<V2> as NodeStorageOps>::OpStorage<V2>>
    {
        let ext = self.external_filter();

        let mut saturator = HedgeGraphBuilder::new();
        for i in ext.included_iter() {
            let flow = self.flow(i);
            let d = &self[[&i]];
            let orientation = self.orientation(i);
            let new_node_data = dangling_map(
                self.edge_store.as_ref(),
                &self.node_store,
                i,
                flow,
                EdgeData::new(d, orientation),
            );

            let n = saturator.add_node(new_node_data);

            saturator.add_external_edge(
                n,
                DanglingMatcher::Saturator { hedge: i },
                orientation,
                -flow,
            );
        }

        let saturator = saturator.build();
        let new_graph = self.map(node_map, |_, _, p, e| e.map(|d| DanglingMatcher::new(p, d)));
        new_graph
            .join(
                saturator,
                |_, dl, _, dr| dl.data.matches(dr.data),
                |fl, dl, _, _| (fl, dl),
            )
            .unwrap()
            .map(|_, _, v| v, |_, _, _, e| e.map(|d| d.unwrap()))
    }

    pub fn map_data_ref<'a, E2, V2>(
        &'a self,
        node_map: impl FnMut(&'a Self, N::NeighborsIter<'a>, &'a V) -> V2,
        edge_map: impl FnMut(&'a Self, EdgeIndex, HedgePair, EdgeData<&'a E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2, N::OpStorage<V2>> {
        HedgeGraph {
            node_store: self.node_store.map_data_ref_graph(self, node_map),
            edge_store: self.edge_store.map_data_ref(self, edge_map),
        }
    }

    pub fn map_data_ref_mut<'a, E2, V2>(
        &'a mut self,
        node_map: impl FnMut(N::NeighborsIter<'a>, &'a mut V) -> V2,
        edge_map: impl FnMut(EdgeIndex, HedgePair, EdgeData<&'a mut E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2, N::OpStorage<V2>> {
        HedgeGraph {
            node_store: self.node_store.map_data_ref_mut_graph(node_map),
            edge_store: self.edge_store.map_data_ref_mut(edge_map),
        }
    }

    pub fn map_data_ref_result<'a, E2, V2, Er>(
        &'a self,
        node_map: impl FnMut(&'a Self, N::NeighborsIter<'a>, &'a V) -> Result<V2, Er>,
        edge_map: impl FnMut(
            &'a Self,
            EdgeIndex,
            HedgePair,
            EdgeData<&'a E>,
        ) -> Result<EdgeData<E2>, Er>,
    ) -> Result<HedgeGraph<E2, V2, N::OpStorage<V2>>, Er> {
        Ok(HedgeGraph {
            node_store: self.node_store.map_data_ref_graph_result(self, node_map)?,
            edge_store: self.edge_store.map_data_ref_result(self, edge_map)?,
        })
    }

    pub fn just_structure(&self) -> HedgeGraph<(), (), N::OpStorage<()>> {
        self.map_data_ref(|_, _, _| (), |_, _, _, d| d.map(|_| ()))
    }

    pub fn map_nodes_ref<'a, V2>(
        &'a self,
        f: impl FnMut(&'a Self, N::NeighborsIter<'a>, &'a V) -> V2,
    ) -> HedgeGraph<&'a E, V2, N::OpStorage<V2>> {
        HedgeGraph {
            node_store: self.node_store.map_data_ref_graph(self, f),
            edge_store: self.edge_store.map_data_ref(self, &|_, _, _, e| e),
        }
    }
    pub fn map<E2, V2>(
        self,
        f: impl for<'b> FnMut(&Involution, NodeIndex, V) -> V2,
        g: impl FnMut(&Involution, &N, HedgePair, EdgeData<E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2, N::OpStorage<V2>> {
        let edge_store = self.edge_store.map_data(&self.node_store, g);
        HedgeGraph {
            node_store: self.node_store.map_data_graph(edge_store.as_ref(), f),
            edge_store,
        }
    }
    pub fn new_smart_hedgevec<T>(
        &self,
        f: &impl Fn(HedgePair, EdgeData<&E>) -> EdgeData<T>,
    ) -> SmartHedgeVec<T> {
        self.edge_store.map(f)
    }

    pub fn new_hedgevec<T>(&self, f: impl FnMut(&E, EdgeIndex, &HedgePair) -> T) -> HedgeVec<T> {
        self.edge_store.new_hedgevec(f)
    }

    pub fn new_hedgevec_from_iter<T, I: IntoIterator<Item = T>>(
        &self,
        iter: I,
    ) -> Result<HedgeVec<T>, HedgeGraphError> {
        self.edge_store.new_hedgevec_from_iter(iter)
    }
}

// Cuts
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    fn non_cut_edges_impl(
        &self,
        connected_components: usize,
        cyclotomatic_number: usize,
        start: Hedge,
        current: &mut BitVec,
        set: &mut AHashSet<BitVec>,
    ) {
        if current.count_ones() > 2 * cyclotomatic_number {
            return;
        }

        let complement = current.complement(self);

        if current.count_ones() > 0
            && self.count_connected_components(&complement) == connected_components
            && complement.covers(self) == self.full_filter()
        {
            // println!("//inserted with {con_comp}");
            set.insert(current.clone());
        }

        for i in (start.0..self.n_hedges()).map(Hedge) {
            let j = self.inv(i);
            if i > j {
                current.set(i.0, true);
                current.set(j.0, true);
                self.non_cut_edges_impl(
                    connected_components,
                    cyclotomatic_number,
                    Hedge(i.0 + 1),
                    current,
                    set,
                );
                current.set(i.0, false);
                current.set(j.0, false);
            }
        }
    }

    /// all sets of full edges that do not disconnect the graph/ increase its connected components
    pub fn non_cut_edges(&self) -> AHashSet<BitVec> {
        let connected_components = self.count_connected_components(&self.full_filter());

        let cyclotomatic_number = self.cyclotomatic_number(&self.full_node().internal_graph);

        let mut current = self.empty_subgraph::<BitVec>();
        let mut set = AHashSet::new();

        self.non_cut_edges_impl(
            connected_components,
            cyclotomatic_number,
            Hedge(0),
            &mut current,
            &mut set,
        );

        set
    }

    // pub fn backtracking_cut_set(
    //     &self,
    //     source: HedgeNode,
    //     target: HedgeNode,
    // ) -> Vec<InternalSubGraph> {
    // }

    pub fn non_bridges(&self) -> BitVec {
        let (c, _) = self.cycle_basis();
        let mut cycle_cover: BitVec = self.empty_subgraph();
        for cycle in c {
            cycle_cover.union_with(&cycle.filter);
        }

        cycle_cover
    }

    pub fn bridges(&self) -> BitVec {
        self.non_bridges().complement(self)
    }

    pub fn combine_to_single_hedgenode(&self, source: &[NodeIndex]) -> HedgeNode {
        let s: BitVec =
            source
                .iter()
                .map(|a| self.iter_crown(*a))
                .fold(self.empty_subgraph(), |mut acc, e| {
                    acc.union_with_iter(e);
                    acc
                });

        let (hairs, internal_graph) = self.split_hairs_and_internal_hedges(s);

        HedgeNode {
            internal_graph,
            hairs,
        }
    }

    pub fn all_cuts_from_ids(
        &self,
        source: &[NodeIndex],
        target: &[NodeIndex],
    ) -> Vec<(BitVec, OrientedCut, BitVec)>
    where
        N: NodeStorageOps,
    {
        let source = self.combine_to_single_hedgenode(source);
        let target = self.combine_to_single_hedgenode(target);
        self.all_cuts(source, target)
    }

    pub fn tadpoles(&self, externals: &[NodeIndex]) -> Vec<BitVec> {
        let mut identified: HedgeGraph<(), (), N::OpStorage<()>> =
            self.map_data_ref(|_, _, _| (), |_, _, _, d| d.map(|_| ()));

        let n = identified.identify_nodes(externals, ());
        let hairs = identified.iter_crown(n).next().unwrap();

        let non_bridges = identified.non_bridges();
        identified
            .connected_components(&non_bridges)
            .into_iter()
            .filter_map(|mut a| {
                if !a.includes(&hairs) {
                    let full = a.covers(self);

                    for i in full.included_iter() {
                        a.set(i.0, true);
                        a.set(self.inv(i).0, true);
                    }
                    Some(a)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn all_cuts(
        &self,
        source: HedgeNode,
        target: HedgeNode,
    ) -> Vec<(BitVec, OrientedCut, BitVec)>
    where
        N: NodeStorageOps,
    {
        // println!("//Source\n{}", self.dot(&source.hairs));
        // println!("//Target\n{}", self.dot(&target.hairs));

        let full_source = source.internal_and_hairs();
        let full_target = target.internal_and_hairs();
        let s_connectivity = self.count_connected_components(&full_source);

        let t_connectivity = self.count_connected_components(&full_target);

        let augmented: HedgeGraph<(), (), N::OpStorage<()>> =
            self.map_data_ref(|_, _, _| (), |_, _, _, d| d.map(|_| ()));
        let s_nodes = self
            .iter_node_data(&source)
            .map(|a| self.id_from_crown(a.1).unwrap())
            .collect::<Vec<_>>();
        let t_nodes = self
            .iter_node_data(&target)
            .map(|a| self.id_from_crown(a.1).unwrap())
            .collect::<Vec<_>>();

        let t_node = t_nodes[0];
        let s_node = s_nodes[0];

        let augmented = s_nodes.iter().fold(augmented, |aug, n| {
            let (_, _, augmented) = aug.add_pair(*n, t_node, (), false).unwrap();
            augmented
        });

        let augmented = t_nodes.iter().fold(augmented, |aug, n| {
            let (_, _, augmented) = aug.add_pair(s_node, *n, (), false).unwrap();
            augmented
        });

        let mut non_bridges = augmented.non_bridges();

        for _ in &s_nodes {
            non_bridges.pop();
            non_bridges.pop();
        }
        for _ in &t_nodes {
            non_bridges.pop();
            non_bridges.pop();
        }

        // println!("//non_bridges:\n{}", self.dot(&non_bridges));

        non_bridges.union_with(&source.hairs);
        non_bridges.union_with(&target.hairs);

        // println!("//non_bridges:\n{}", self.dot(&non_bridges));

        let mut regions = AHashSet::new();
        self.all_s_t_cuts_impl(
            &non_bridges,
            s_connectivity,
            source,
            &target,
            t_connectivity,
            &mut regions,
        );

        let mut cuts = vec![];
        let bridges = non_bridges.complement(self);

        for mut r in regions.drain() {
            // let disconnected = r.complement(self);

            // let mut s_side_covers = if s.hairs.intersects(&r) {
            //     s.hairs.clone()
            // } else {
            //     SimpleTraversalTree::depth_first_traverse(self, &disconnected, &source, None)
            //         .unwrap()
            //         .covers()
            // };
            let cut = OrientedCut::from_underlying_coerce(r.hairs.clone(), self).unwrap();
            r.add_all_hairs(self);
            let mut s_side_covers = r.internal_and_hairs();
            for i in bridges.included_iter() {
                if s_side_covers.includes(&self.inv(i)) {
                    s_side_covers.set(i.0, true);
                }
            }

            let t_side_covers = s_side_covers.complement(self);

            // let internal = InternalSubGraph::cleaned_filter_pessimist(t_side_covers, self);
            // let mut t_side = self.nesting_node_from_subgraph(internal);
            // t_side.hairs.union_with(&t.hairs);
            //

            cuts.push((s_side_covers, cut, t_side_covers));
        }

        cuts
    }

    pub fn all_s_t_cuts_impl<S: SubGraph<Base = BitVec>>(
        &self,
        subgraph: &S,
        s_connectivity: usize,
        s: HedgeNode, // will grow
        t: &HedgeNode,
        t_connectivity: usize,
        regions: &mut AHashSet<HedgeNode>,
    ) {
        // println!("regions size:{}", regions.len());
        //

        let hairy = s.internal_graph.filter.union(&s.hairs);
        let mut complement = hairy.complement(self);
        complement.intersect_with(subgraph.included());
        if !complement.includes(&t.hairs) {
            return;
        }

        let t_count = self.count_connected_components(&complement);
        let s_count = self.count_connected_components(&hairy);

        if t_count <= t_connectivity && s_count <= s_connectivity && regions.get(&s).is_none() {
            for h in s.hairs.included_iter() {
                let invh = self.inv(h);

                if invh != h && !t.hairs.includes(&invh) && subgraph.includes(&invh) {
                    let mut new_node = s.clone();

                    new_node.hairs.union_with_iter(self.neighbors(invh));
                    new_node.hairs.intersect_with(subgraph.included());

                    new_node.fix(self);
                    self.all_s_t_cuts_impl(
                        subgraph,
                        s_connectivity,
                        new_node,
                        t,
                        t_connectivity,
                        regions,
                    );
                }
            }
            regions.insert(s);
        }
    }
}

// Cycles
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn cyclotomatic_number<S: SubGraph>(&self, subgraph: &S) -> usize {
        let n_hedges = self.count_internal_edges(subgraph);
        // println!("n_hedges: {}", n_hedges);
        let n_nodes = self.number_of_nodes_in_subgraph(subgraph);
        // println!("n_nodes: {}", n_nodes);
        let n_components = self.count_connected_components(subgraph);

        n_hedges + n_components - n_nodes
    }

    pub fn cycle_basis(&self) -> (Vec<Cycle>, SimpleTraversalTree) {
        self.paton_cycle_basis(&self.full_graph(), &self.node_id(Hedge(0)), None)
            .unwrap()
    }

    pub fn order_basis(&self, basis: &[HedgeNode]) -> Vec<Vec<InternalSubGraph>> {
        let mut seen = vec![basis[0].internal_graph.clone()];
        let mut partitions = vec![seen.clone()];

        for cycle in basis.iter() {
            if seen
                .iter()
                .any(|p| !p.empty_intersection(&cycle.internal_graph))
            {
                partitions
                    .last_mut()
                    .unwrap()
                    .push(cycle.internal_graph.clone());
            } else {
                for p in partitions.last().unwrap() {
                    seen.push(p.clone());
                }
                partitions.push(vec![cycle.internal_graph.clone()]);
            }
        }

        partitions
    }

    pub fn all_cycles(&self) -> Vec<Cycle> {
        Cycle::all_sum_powerset_filter_map(&self.cycle_basis().0, &|mut c| {
            if c.is_circuit(self) {
                c.loop_count = Some(1);
                Some(c)
            } else {
                None
            }
        })
        .unwrap()
        .into_iter()
        .collect()
    }

    pub fn all_cycle_sym_diffs(&self) -> Result<Vec<InternalSubGraph>, TryFromIntError> {
        Cycle::all_sum_powerset_filter_map(&self.cycle_basis().0, &Some)
            .map(|a| a.into_iter().map(|c| c.internal_graph(self)).collect())
    }

    pub fn all_cycle_unions(&self) -> AHashSet<InternalSubGraph> {
        InternalSubGraph::all_unions_iterative(&self.all_cycle_sym_diffs().unwrap())
    }
    pub fn paton_count_loops(
        &self,
        subgraph: &InternalSubGraph,
        start: &NodeIndex,
    ) -> Result<usize, HedgeGraphError> {
        let tree = SimpleTraversalTree::depth_first_traverse(self, subgraph, start, None)?;

        let cuts = subgraph.subtract(&tree.tree_subgraph(self));
        Ok(self.edge_store.n_internals(&cuts))
    }

    pub fn paton_cycle_basis<S: SubGraph<Base = BitVec>>(
        &self,
        subgraph: &S,
        start: &NodeIndex,
        included_hedge: Option<Hedge>,
    ) -> Result<(Vec<Cycle>, SimpleTraversalTree), HedgeGraphError> {
        let tree =
            SimpleTraversalTree::depth_first_traverse(self, subgraph, start, included_hedge)?;

        let cuts = subgraph.included().subtract(&tree.tree_subgraph);

        let mut cycle_basis = Vec::new();

        for c in cuts.included_iter() {
            if c > self.inv(c) && subgraph.includes(&self.inv(c)) {
                cycle_basis.push(tree.get_cycle(c, self).unwrap());
            }
        }

        Ok((cycle_basis, tree))
    }

    pub fn all_spinneys_with_basis(&self, basis: &[&InternalSubGraph]) -> AHashSet<HedgeNode> {
        let mut spinneys = AHashSet::new();
        let mut base_cycle: InternalSubGraph = self.empty_subgraph();

        for cycle in basis {
            base_cycle.sym_diff_with(cycle);
        }

        spinneys.insert(self.nesting_node_from_subgraph(base_cycle.clone()));

        if basis.len() == 1 {
            return spinneys;
        }

        for i in 0..basis.len() {
            for s in self.all_spinneys_with_basis(
                &basis
                    .iter()
                    .enumerate()
                    .filter_map(|(j, s)| if j != i { Some(*s) } else { None })
                    .collect_vec(),
            ) {
                spinneys
                    .insert(self.nesting_node_from_subgraph(s.internal_graph.union(&base_cycle)));
                spinneys.insert(s);
            }
        }

        spinneys
    }

    pub fn all_spinneys_rec(&self, spinneys: &mut AHashSet<HedgeNode>, cycle_sums: Vec<HedgeNode>) {
        let _len = spinneys.len();

        let mut pset = PowersetIterator::new(cycle_sums.len() as u8);

        pset.next(); //Skip empty set

        for (ci, cj) in cycle_sums.iter().tuple_combinations() {
            let _union = ci.internal_graph.union(&cj.internal_graph);

            // spinneys.insert(union);
        }
    }

    pub fn all_spinneys(
        &self,
    ) -> AHashMap<InternalSubGraph, Vec<(InternalSubGraph, Option<InternalSubGraph>)>> {
        let basis_cycles = self.cycle_basis().0;

        let mut all_combinations = PowersetIterator::new(basis_cycles.len() as u8);
        all_combinations.next(); //Skip empty set

        let mut spinneys: AHashMap<
            InternalSubGraph,
            Vec<(InternalSubGraph, Option<InternalSubGraph>)>,
        > = AHashMap::new();

        let mut cycles: Vec<InternalSubGraph> = Vec::new();
        for p in all_combinations {
            let mut base_cycle: InternalSubGraph = self.empty_subgraph();

            for i in p.iter_ones() {
                base_cycle.sym_diff_with(&basis_cycles[i].clone().internal_graph(self));
            }

            cycles.push(base_cycle);
        }

        for (ci, cj) in cycles.iter().tuple_combinations() {
            let union = ci.union(cj);

            if let Some(v) = spinneys.get_mut(&union) {
                v.push((ci.clone(), Some(cj.clone())));
            } else {
                spinneys.insert(union, vec![(ci.clone(), Some(cj.clone()))]);
            }
        }

        for c in cycles {
            spinneys.insert(c.clone(), vec![(c.clone(), None)]);
        }
        spinneys
    }

    pub fn all_spinneys_alt(&self) -> AHashSet<InternalSubGraph> {
        let mut spinneys = AHashSet::new();
        let cycles = self.all_cycles();

        let mut pset = PowersetIterator::new(cycles.len() as u8);
        pset.next(); //Skip empty set

        for p in pset {
            let mut union: InternalSubGraph = self.empty_subgraph();

            for i in p.iter_ones() {
                union.union_with(&cycles[i].clone().internal_graph(self));
            }

            spinneys.insert(union);
        }

        for c in cycles {
            spinneys.insert(c.internal_graph(self));
        }
        spinneys
    }
}

// Traversal Trees
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn count_connected_components<S: SubGraph>(&self, subgraph: &S) -> usize {
        self.connected_components(subgraph).len()
    }

    pub fn connected_components<S: SubGraph>(&self, subgraph: &S) -> Vec<BitVec> {
        let mut visited_edges: BitVec = self.empty_subgraph();

        let mut components = vec![];

        // Iterate over all edges in the subgraph
        for hedge_index in subgraph.included_iter() {
            if !visited_edges.includes(&hedge_index) {
                // Perform DFS to find all reachable edges from this edge

                //
                let root_node = self.node_id(hedge_index);
                let reachable_edges =
                    SimpleTraversalTree::depth_first_traverse(self, subgraph, &root_node, None)
                        .unwrap()
                        .covers(subgraph);

                visited_edges.union_with(&reachable_edges);

                components.push(reachable_edges);
            }
        }
        components
    }
    pub fn align_underlying_to_superficial(&mut self) {
        self.edge_store.align_underlying_to_superficial();
    }

    /// aligns the underlying orientation of the graph to the tree, such that all tree edges are oriented towards the root, and all others point towards the leaves
    ///
    /// Relies on the tree being tremaux (i.e. the tree-order is total)
    /// This is the case for depth-first traversal.
    pub fn align_underlying_to_tree<P: ForestNodeStore<NodeData = ()>>(
        &mut self,
        tree: &SimpleTraversalTree<P>,
    ) {
        for (h, tt, i) in tree.iter_hedges() {
            // println!("hedge: {}, tt: {:?}, i: {:?}\n", h, tt, i);
            match tt {
                tree::TTRoot::Root => {
                    if i.is_some() {
                        self.edge_store.set_flow(h, Flow::Sink);
                    } else {
                        self.edge_store.set_flow(h, Flow::Source);
                    }
                }
                tree::TTRoot::Child => {
                    if let Some(root_pointer) = i {
                        if root_pointer == h {
                            self.edge_store.set_flow(h, Flow::Source);
                        } else {
                            self.edge_store.set_flow(h, Flow::Sink);
                        }
                    } else {
                        let current_node_id = tree.node_id(h);
                        let involved_node_id = tree.node_id(self.inv(h));

                        let order =
                            tree.tree_order(current_node_id, involved_node_id, &self.edge_store);
                        match order {
                            Some(Ordering::Equal) => {
                                // self.edge_store.set_flow(h);
                            }
                            Some(Ordering::Less) => {
                                //the path to the root from the current node, passes through the involved node
                                self.edge_store.set_flow(h, Flow::Sink);
                            }
                            Some(Ordering::Greater) => {
                                self.edge_store.set_flow(h, Flow::Source);
                            }
                            None => {}
                        }
                    }
                }
                tree::TTRoot::None => {}
            }
        }
    }

    /// aligns the superficial orientation of the graph to the tree,
    ///
    /// such that all tree edges are oriented towards the root, and all others are unoriented.
    pub fn align_superficial_to_tree<P: ForestNodeStore<NodeData = ()>>(
        &mut self,
        tree: &SimpleTraversalTree<P>,
    ) {
        for (h, tt, i) in tree.iter_hedges() {
            match tt {
                tree::TTRoot::Root => {
                    if i.is_some() {
                        let flow = self.edge_store.flow(h);
                        match flow {
                            Flow::Source => {
                                self.edge_store.set_orientation(h, Orientation::Reversed);
                            }
                            Flow::Sink => {
                                self.edge_store.set_orientation(h, Orientation::Default);
                            }
                        }
                    } else {
                        self.edge_store.set_orientation(h, Orientation::Undirected);
                    }
                }
                tree::TTRoot::Child => {
                    if let Some(root_pointer) = i {
                        let flow = self.edge_store.flow(h);
                        if root_pointer == h {
                            match flow {
                                Flow::Source => {
                                    self.edge_store.set_orientation(h, Orientation::Default);
                                }
                                Flow::Sink => {
                                    self.edge_store.set_orientation(h, Orientation::Reversed);
                                }
                            }
                        } else {
                            match flow {
                                Flow::Source => {
                                    self.edge_store.set_orientation(h, Orientation::Reversed);
                                }
                                Flow::Sink => {
                                    self.edge_store.set_orientation(h, Orientation::Default);
                                }
                            }

                            // self.edge_store.involution.set_as_sink(h);
                        }
                    } else {
                        self.edge_store.set_orientation(h, Orientation::Undirected);
                    }
                }
                tree::TTRoot::None => {}
            }
        }
    }

    // pub fn align_to_tree_superficial(&mut self, tree: &TraversalTree) {
    //     for (i, p) in tree.parent_iter() {
    //         match self.edge_store.involution.hedge_data_mut(i) {
    //             InvolutiveMapping::Identity { data, .. } => match p {
    //                 Parent::Root => {}
    //                 Parent::Hedge { hedge_to_root, .. } => {
    //                     if *hedge_to_root == i {
    //                         data.orientation = Orientation::Default;
    //                     } else {
    //                         data.orientation = Orientation::Reversed;
    //                     }
    //                 }
    //                 Parent::Unset => {}
    //             },
    //             InvolutiveMapping::Source { data, .. } => match p {
    //                 Parent::Root => {}
    //                 Parent::Hedge { hedge_to_root, .. } => {
    //                     if *hedge_to_root == i {
    //                         data.orientation = Orientation::Default;
    //                     } else {
    //                         data.orientation = Orientation::Reversed;
    //                     }
    //                 }
    //                 Parent::Unset => {}
    //             },
    //             _ => {}
    //         }
    //     }
    // }
}

// Iterators
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn iter_nodes(&self) -> impl Iterator<Item = (N::NeighborsIter<'_>, NodeIndex, &V)> {
        self.node_store.iter_nodes()
    }

    pub fn iter_nodes_mut(
        &mut self,
    ) -> impl Iterator<Item = (N::NeighborsIter<'_>, NodeIndex, &mut V)> {
        self.node_store.iter_nodes_mut()
    }

    pub fn base_nodes_iter(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.node_store.iter_node_id()
    }

    pub fn iter_edge_id<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> EdgeIter<'a, E, V, S, N, S::BaseIter<'a>> {
        EdgeIter::new(self, subgraph)
    }

    pub fn iter_edges<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&'a E>)> + 'a {
        self.edge_store.iter_edges(subgraph)
    }

    pub fn iter_all_edges(&self) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&E>)> {
        self.edge_store.iter_all_edges()
    }

    pub fn iter_internal_edge_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a E>> + 'a {
        self.edge_store.iter_internal_edge_data(subgraph)
    }

    pub fn iter_egde_node<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = N::NeighborsIter<'a>> + 'a {
        subgraph.included_iter().map(|i| self.neighbors(i))
    }

    pub fn iter_node_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (NodeIndex, N::NeighborsIter<'a>, &'a V)>
    where
        N::NeighborsIter<'a>: Clone,
    {
        NodeIterator {
            graph: self,
            edges: subgraph.included_iter(),
            seen: bitvec![usize, Lsb0; 0; self.node_store.node_len()],
        }
    }
}

// Display
impl<E, V, N: NodeStorageOps<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn dot_impl_fmt<S: SubGraph, Str1: AsRef<str>>(
        &self,
        writer: &mut impl std::fmt::Write,
        subgraph: &S,
        graph_info: Str1,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> Result<(), std::fmt::Error> {
        subgraph.dot_fmt(writer, self, graph_info, edge_attr, node_attr)
    }
    pub fn dot_impl_io<S: SubGraph, Str1: AsRef<str>>(
        &self,
        writer: &mut impl std::io::Write,
        subgraph: &S,
        graph_info: Str1,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> Result<(), std::io::Error> {
        subgraph.dot_io(writer, self, graph_info, edge_attr, node_attr)
    }

    pub fn dot_impl<S: SubGraph, Str1: AsRef<str>>(
        &self,
        subgraph: &S,
        graph_info: Str1,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> String {
        let mut output = String::new();
        subgraph
            .dot_fmt(&mut output, self, graph_info, edge_attr, node_attr)
            .unwrap();
        output
    }

    pub fn dot<S: SubGraph>(&self, node_as_graph: &S) -> String {
        let mut output = String::new();
        self.dot_impl_fmt(&mut output, node_as_graph, "start=2;\n", &|_| None, &|_| {
            None
        })
        .unwrap();
        output
    }

    pub fn dot_display<S: SubGraph>(&self, node_as_graph: &S) -> String
    where
        E: Display,
        V: Display,
    {
        let mut output = String::new();
        self.dot_impl_fmt(
            &mut output,
            node_as_graph,
            "start=2;\n",
            &|a| Some(format!("{}", a)),
            &|v| Some(format!("{}", v)),
        )
        .unwrap();
        output
    }

    pub fn dot_label<S: SubGraph>(&self, node_as_graph: &S) -> String
    where
        E: Display,
        V: Display,
    {
        let mut output = String::new();
        self.dot_impl_fmt(
            &mut output,
            node_as_graph,
            "start=2;\n",
            &|a| Some(format!("label=\"{}\"", a)),
            &|v| Some(format!("label=\"{}\"", v)),
        )
        .unwrap();
        output
    }

    pub fn base_dot(&self) -> String {
        self.dot(&self.full_filter())
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> Index<&Hedge> for HedgeGraph<E, V, N> {
    type Output = EdgeIndex;
    fn index(&self, index: &Hedge) -> &Self::Output {
        &self.edge_store[index]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> Index<[&Hedge; 1]> for HedgeGraph<E, V, N> {
    type Output = E;
    fn index(&self, index: [&Hedge; 1]) -> &Self::Output {
        &self[self[index[0]]]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> IndexMut<[&Hedge; 1]> for HedgeGraph<E, V, N> {
    // type Output = E;
    fn index_mut(&mut self, index: [&Hedge; 1]) -> &mut Self::Output {
        let edgeid = self[index[0]];
        &mut self[edgeid]
    }
}

impl<E, V, N: NodeStorageOps<NodeData = V>> Index<NodeIndex> for HedgeGraph<E, V, N> {
    type Output = V;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.node_store.get_node_data(index)
    }
}
impl<E, V, N: NodeStorageOps<NodeData = V>> IndexMut<NodeIndex> for HedgeGraph<E, V, N> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.node_store.get_node_data_mut(index)
    }
}
// impl<E, V, N: NodeStorageOps<NodeData = V>> Index<&NodeIndex> for HedgeGraph<E, V, N> {
//     type Output = HedgeNode;
//     fn index(&self, index: &NodeIndex) -> &Self::Output {
//         self.node_store.get_neighbor_iterator(*index)
//     }
// // }
// impl<E, V, N: NodeStorageOps<NodeData = V>> Index<&HedgeNode> for HedgeGraph<E, V, N> {
//     type Output = V;
//     fn index(&self, index: &HedgeNode) -> &Self::Output {
//         let id = self.id_from_hairs(index).unwrap();
//         &self[id]
//     }
// }

// impl<E, V, N: NodeStorageOps<NodeData = V>> IndexMut<&HedgeNode> for HedgeGraph<E, V, N> {
//     fn index_mut(&mut self, index: &HedgeNode) -> &mut Self::Output {
//         let id = self.id_from_hairs(index).unwrap();
//         &mut self[id]
//     }
// }

impl<E, V, N: NodeStorage<NodeData = V>> Index<EdgeIndex> for HedgeGraph<E, V, N> {
    type Output = E;
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edge_store[index]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> Index<&EdgeIndex> for HedgeGraph<E, V, N> {
    type Output = (E, HedgePair);
    fn index(&self, index: &EdgeIndex) -> &Self::Output {
        &self.edge_store[index]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> IndexMut<EdgeIndex> for HedgeGraph<E, V, N> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edge_store[index]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> IndexMut<&EdgeIndex> for HedgeGraph<E, V, N> {
    fn index_mut(&mut self, index: &EdgeIndex) -> &mut Self::Output {
        &mut self.edge_store[index]
    }
}

pub struct NodeIterator<'a, E, V, N: NodeStorage<NodeData = V>, I = IterOnes<'a, usize, Lsb0>> {
    graph: &'a HedgeGraph<E, V, N>,
    edges: I,
    seen: BitVec,
}

impl<'a, E, V, I: Iterator<Item = Hedge>, N: NodeStorageOps<NodeData = V>> Iterator
    for NodeIterator<'a, E, V, N, I>
where
    N::NeighborsIter<'a>: Clone,
{
    type Item = (NodeIndex, N::NeighborsIter<'a>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.edges.next() {
            let node = self.graph.neighbors(next);
            let node_pos = self.graph.id_from_crown(node.clone()).unwrap();

            if self.seen[node_pos.0] {
                self.next()
            } else {
                self.seen.set(node_pos.0, true);
                Some((node_pos, node, &self.graph[node_pos]))
            }
        } else {
            None
        }
    }
}

#[cfg(feature = "symbolica")]
pub mod symbolica_interop;

use subgraph::{
    BaseSubgraph, Cycle, HedgeNode, Inclusion, InternalSubGraph, ModifySubgraph, OrientedCut,
    SubGraph, SubGraphOps,
};

use thiserror::Error;
use tree::SimpleTraversalTree;

use crate::tree::ForestNodeStore;

#[derive(Error, Debug)]
pub enum HedgeError {
    #[error("Invalid start node")]
    InvalidStart,
}

pub struct EdgeIter<'a, E, V, S, N: NodeStorage<NodeData = V>, I: Iterator<Item = Hedge> + 'a> {
    graph: &'a HedgeGraph<E, V, N>,
    included_iter: I,
    subgraph: &'a S,
}
impl<'a, E, V, S, N: NodeStorage<NodeData = V>> EdgeIter<'a, E, V, S, N, S::BaseIter<'a>>
where
    S: SubGraph,
{
    pub fn new(graph: &'a HedgeGraph<E, V, N>, subgraph: &'a S) -> Self {
        EdgeIter {
            graph,
            subgraph,
            included_iter: subgraph.included_iter(),
        }
    }
}

impl<'a, E, V, S, N: NodeStorage<NodeData = V>> Iterator
    for EdgeIter<'a, E, V, S, N, S::BaseIter<'a>>
where
    S: SubGraph,
{
    type Item = (HedgePair, EdgeData<&'a E>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.included_iter.next()?;
        let orientation = self.graph.edge_store.orientation(i);
        let data = &self.graph[self.graph[&i]];
        if let Some(e) =
            HedgePair::from_source_with_subgraph(i, &self.graph.edge_store, self.subgraph)
        {
            Some((e, EdgeData::new(data, orientation)))
        } else {
            self.next()
        }
    }
}

#[derive(Debug, Error)]
pub enum HedgeGraphError {
    #[error("Nodes do not partition: {0}")]
    NodesDoNotPartition(String),
    #[error("Invalid node")]
    NoNode,
    #[error("Invalid hedge {0}")]
    InvalidHedge(Hedge),
    #[error("External hedge as included: {0}")]
    ExternalHedgeIncluded(Hedge),

    #[error("Included hedge {0} is not in node {1}")]
    NotInNode(Hedge, NodeIndex),

    #[error("Traversal Root node not in subgraph {0}")]
    RootNodeNotInSubgraph(NodeIndex),
    #[error("Invalid node {0}")]
    InvalidNode(NodeIndex),
    #[error("Invalid edge")]
    NoEdge,
    #[error("Dangling Half edge present")]
    HasIdentityHedge,
    #[error("SymbolicaError: {0}")]
    SymbolicaError(&'static str),
    #[error("InvolutionError: {0}")]
    InvolutionError(#[from] InvolutionError),
    #[error("Data length mismatch")]
    DataLengthMismatch,
    // #[error("From file error: {0}")]
    // FromFileError(#[from] GraphFromFileError),
    // #[error("Parse error: {0}")]
    // ParseError(#[from] PestError),
}

pub mod hedgevec;
pub mod tree;

#[cfg(feature = "drawing")]
pub mod drawing;
#[cfg(feature = "drawing")]
pub mod layout;
#[cfg(test)]
mod test_graphs;
#[cfg(test)]
mod tests;
