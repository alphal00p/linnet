use core::panic;
use std::hash::Hash;
use std::num::TryFromIntError;
use std::ops::{Index, IndexMut, Neg};

use ahash::{AHashMap, AHashSet};
use bitvec::prelude::*;
use bitvec::{slice::IterOnes, vec::BitVec};
use builder::HedgeGraphBuilder;
use derive_more::{From, Into};
use hedgevec::{HedgeVec, SmartHedgeVec};
use indexmap::IndexSet;
use involution::{
    EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Involution, InvolutionError, InvolutiveMapping,
    Orientation,
};
use itertools::Itertools;
use nodestorage::{NodeStorage, NodeStorageVec};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use ref_ops::RefMutNot;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, From, Into)]
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

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GVEdgeAttrs {
    pub label: Option<String>,
    pub color: Option<String>,
    pub other: Option<String>,
}

impl std::fmt::Display for GVEdgeAttrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = [
            ("label=", self.label.as_ref()),
            ("color=", self.color.as_ref()),
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
pub mod nodestorage;
pub mod subgraph;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HedgeGraph<E, V, S: NodeStorage<NodeData = V> = NodeStorageVec<V>> {
    edge_store: SmartHedgeVec<E>,
    node_store: S,
}

impl<E, V: Default, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    /// Creates a random graph with the given number of nodes and edges.
    pub fn random(nodes: usize, edges: usize, seed: u64) -> HedgeGraph<(), V, N> {
        let inv: Involution<()> = Involution::<()>::random(edges, seed);

        let mut rng = SmallRng::seed_from_u64(seed);

        let mut externals = Vec::new();
        let mut sources = Vec::new();
        let mut sinks = Vec::new();

        for (i, e) in inv.iter() {
            let nodeid = HedgeNode::node_from_pos(&[i.0], inv.len());
            match e {
                InvolutiveMapping::Identity { .. } => externals.push(nodeid),
                InvolutiveMapping::Source { .. } => sources.push(nodeid),
                InvolutiveMapping::Sink { .. } => sinks.push(nodeid),
            }
        }

        assert_eq!(sources.len(), sinks.len());
        let n_internals = sources.len();

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

impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
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

    /// Is the graph connected?
    pub fn is_connected<S: SubGraph>(&self, subgraph: &S) -> bool {
        let n_edges = subgraph.nedges(self);
        if let Some(start) = subgraph.included_iter().next() {
            TraversalTree::dfs(self, subgraph, self.node_hairs(start), None)
                .covers()
                .nedges(self)
                == n_edges
        } else {
            true
        }
    }

    pub fn cut_branches(&self, subgraph: &mut HedgeNode) {
        let nodes = AHashSet::<&HedgeNode>::from_iter(
            subgraph
                .internal_graph
                .included_iter()
                .map(|i| self.node_hairs(i)),
        );
        self.remove_externals(subgraph);

        let mut has_branch = true;
        while has_branch {
            has_branch = false;

            for n in &nodes {
                let int = n.hairs.clone() & &subgraph.internal_graph.filter;

                if int.count_ones() == 1 {
                    subgraph
                        .internal_graph
                        .filter
                        .set(int.first_one().unwrap(), false);
                    subgraph
                        .internal_graph
                        .filter
                        .set(self.inv(Hedge(int.first_one().unwrap())).0, false);
                    has_branch = true;
                }
            }
        }

        self.nesting_node_fix(subgraph);
    }

    fn set_hedge_data(&mut self, hedge: Hedge, nodeid: NodeIndex) {
        self.node_store.set_hedge_data(hedge, nodeid);
    }
}

// Subgraphs
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
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

        for (i, edge) in self.edge_store.involution.inv.iter().enumerate() {
            if edge.is_identity() {
                filter.set(i, true);
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

    pub fn hairy_from_filter(&self, filter: BitVec) -> HedgeNode {
        self.nesting_node_from_subgraph(InternalSubGraph::cleaned_filter_pessimist(filter, self))
    }

    pub fn nesting_node_from_subgraph(&self, internal_graph: InternalSubGraph) -> HedgeNode {
        let mut hairs = bitvec![usize, Lsb0; 0; self.n_hedges()];

        if !internal_graph.valid::<E, V, N>(self) {
            panic!("Invalid subgraph")
        }

        for i in internal_graph.included_iter() {
            hairs |= &self.node_hairs(i).hairs;
        }

        HedgeNode {
            hairs: !(!hairs | &internal_graph.filter),
            internal_graph,
        }
    }

    fn nesting_node_fix(&self, node: &mut HedgeNode) {
        let mut externalhedges = bitvec![usize, Lsb0; 0; self.n_hedges()];

        for i in node.internal_graph.filter.included_iter() {
            externalhedges |= &self.node_hairs(i).hairs;
        }

        node.hairs = !(!externalhedges | &node.internal_graph.filter);
    }

    fn remove_externals(&self, subgraph: &mut HedgeNode) {
        let externals = self.external_filter();

        subgraph.internal_graph.filter &= !externals;
    }
}

// Counts
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn count_internal_edges(&self, subgraph: &InternalSubGraph) -> usize {
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

    pub fn number_of_nodes_in_subgraph(&self, subgraph: &InternalSubGraph) -> usize {
        self.iter_node_data(subgraph).count()
    }

    pub fn node_degrees_in_subgraph(&self, subgraph: &InternalSubGraph) -> AHashMap<usize, usize> {
        let mut degrees = AHashMap::new();

        for (node, _) in self.iter_node_data(subgraph) {
            let node_pos = self.id_from_hairs(node).unwrap().0;

            // Count the number of edges in the subgraph incident to this node
            let incident_edges = node.hairs.clone() & &subgraph.filter;
            let degree = incident_edges.count_ones();

            degrees.insert(node_pos, degree);
        }

        degrees
    }
}

// Accessors
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    /// including pos
    pub fn neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> BitVec {
        subgraph.hairs(self.node_hairs(pos))
    }

    pub fn connected_neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> Option<BitVec> {
        Some(subgraph.hairs(self.involved_node_hairs(pos)?))
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

    pub fn orientation(&self, hedge_pair: HedgePair) -> Orientation {
        self.edge_store.orientation(hedge_pair.any_hedge())
    }

    /// Gives the underlying orientation of this half-edge.
    pub fn flow(&self, hedge: Hedge) -> Flow {
        self.edge_store.flow(hedge)
    }

    pub fn superficial_hedge_orientation(&self, hedge: Hedge) -> Option<Flow> {
        match self.edge_store.involution.hedge_data(hedge) {
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
        match self.edge_store.involution.hedge_data(hedge) {
            InvolutiveMapping::Identity { underlying, .. } => *underlying,
            InvolutiveMapping::Sink { .. } => Flow::Sink,
            InvolutiveMapping::Source { .. } => Flow::Source,
        }
    }

    pub fn node_hairs(&self, hedge: Hedge) -> &HedgeNode {
        self.hairs_from_id(self.node_id(hedge))
    }

    pub fn hairs_from_id(&self, id: NodeIndex) -> &HedgeNode {
        &self[&id]
    }

    pub fn id_from_hairs(&self, id: &HedgeNode) -> Option<NodeIndex> {
        let e = id.hairs.included_iter().next()?;
        Some(self.node_id(e))
    }

    pub fn involved_node_hairs(&self, hedge: Hedge) -> Option<&HedgeNode> {
        self.involved_node_id(hedge)
            .map(|id| self.hairs_from_id(id))
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

    /// Collect all nodes in the subgraph (all nodes that the hedges are connected to)
    pub fn nodes<S: SubGraph>(&self, subgraph: &S) -> Vec<NodeIndex> {
        let mut nodes = IndexSet::new();
        for i in subgraph.included_iter() {
            let node = self.node_id(i);
            nodes.insert(node);
        }

        nodes.into_iter().collect()
    }

    /// Collect all edges in the subgraph
    /// (This is without double counting, i.e. if two half-edges are part of the same edge, only one `EdgeIndex` will be collected)
    pub fn edges<S: SubGraph>(&self, subgraph: &S) -> Vec<EdgeIndex> {
        self.iter_edges(subgraph).map(|(_, i, _)| i).collect()
    }
}

// Mapping
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn map_data_ref<'a, E2, V2>(
        &'a self,
        node_map: &impl Fn(&'a Self, &'a V, &'a HedgeNode) -> V2,
        edge_map: &impl Fn(&'a Self, HedgePair, EdgeData<&'a E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2, N::Storage<V2>> {
        HedgeGraph {
            node_store: self.node_store.map_data_ref_graph(self, node_map),
            edge_store: self.edge_store.map_data_ref(self, edge_map),
        }
    }

    pub fn map_nodes_ref<V2>(
        &self,
        f: &impl Fn(&Self, &V, &HedgeNode) -> V2,
    ) -> HedgeGraph<&E, V2, N::Storage<V2>> {
        self.map_data_ref(f, &|_, _, e| e)
    }
    pub fn map<E2, V2>(
        self,
        f: impl FnMut(&Involution<EdgeIndex>, &HedgeNode, NodeIndex, V) -> V2,
        g: impl FnMut(&Involution<EdgeIndex>, &N, HedgePair, EdgeData<E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2, N::Storage<V2>> {
        let edge_store = self.edge_store.map_data(&self.node_store, g);
        HedgeGraph {
            node_store: self.node_store.map_data_graph(&edge_store.involution, f),
            edge_store,
        }
    }
    pub fn new_smart_hedgevec<T>(
        &self,
        f: &impl Fn(HedgePair, EdgeData<&E>) -> EdgeData<T>,
    ) -> SmartHedgeVec<T> {
        self.edge_store.map(f)
    }

    pub fn new_hedgevec<T>(&self, f: &impl Fn(&E, EdgeIndex) -> T) -> HedgeVec<T> {
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
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
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

    pub fn all_cuts(
        &self,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Vec<(InternalSubGraph, OrientedCut, InternalSubGraph)> {
        let s = self.hairs_from_id(source);
        let t = self.hairs_from_id(target);
        let mut regions = AHashSet::new();
        self.all_s_t_cuts_impl(s, t, &mut regions);

        let mut cuts = vec![];

        for r in regions.drain() {
            let hairs = self.nesting_node_from_subgraph(r.clone()).hairs;
            let complement = r.complement(self);

            let cut = OrientedCut::from_underlying_coerce(hairs, self).unwrap();
            cuts.push((r, cut, complement));
        }

        cuts
    }

    pub fn all_s_t_cuts_impl(
        &self,
        s: &HedgeNode,
        t: &HedgeNode,
        regions: &mut AHashSet<InternalSubGraph>,
    ) {
        let mut new_internals = vec![];
        for h in s.hairs.included_iter() {
            let invh = self.inv(h);

            if h > invh && s.hairs.includes(&self.inv(h)) {
                new_internals.push(h);
            }
        }

        let mut new_node = s.clone();

        for h in new_internals {
            new_node.hairs.set(h.0, false);
            new_node.hairs.set(self.inv(h).0, false);
            new_node.internal_graph.filter.set(h.0, true);
            new_node.internal_graph.filter.set(self.inv(h).0, true);
        }

        let hairy = new_node.internal_graph.filter.union(&new_node.hairs);
        let complement = hairy.complement(self);
        let count = self.count_connected_components(&complement);

        if count == 1 && !regions.insert(new_node.internal_graph.clone()) {
            return;
        }

        for h in new_node.included_iter() {
            let invh = self.inv(h);

            if invh != h && !t.hairs.includes(&invh) {
                let mut new_node = s.clone();
                new_node.hairs.union_with(&self.node_hairs(invh).hairs);

                new_node.hairs.set(h.0, false);
                new_node.hairs.set(invh.0, false);
                new_node.internal_graph.filter.set(h.0, true);
                new_node.internal_graph.filter.set(invh.0, true);
                self.all_s_t_cuts_impl(&new_node, t, regions);
            }
        }
    }
}

// Cycles
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn cyclotomatic_number(&self, subgraph: &InternalSubGraph) -> usize {
        let n_hedges = self.count_internal_edges(subgraph);
        // println!("n_hedges: {}", n_hedges);
        let n_nodes = self.number_of_nodes_in_subgraph(subgraph);
        // println!("n_nodes: {}", n_nodes);
        let n_components = self.count_connected_components(subgraph);
        // println!("n_components: {}", n_components);

        n_hedges - n_nodes + n_components
    }

    pub fn cycle_basis(&self) -> (Vec<Cycle>, TraversalTree) {
        self.paton_cycle_basis(&self.full_graph(), self.node_hairs(Hedge(0)), None)
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
        start: &HedgeNode,
    ) -> Result<usize, HedgeError> {
        let tree = TraversalTree::dfs(self, subgraph, start, None);

        let cuts = subgraph.subtract(&tree.tree);
        Ok(self.edge_store.involution.n_internals(&cuts))
    }

    pub fn paton_cycle_basis(
        &self,
        subgraph: &InternalSubGraph,
        start: &HedgeNode,
        included_hedge: Option<Hedge>,
    ) -> Result<(Vec<Cycle>, TraversalTree), HedgeError> {
        if !subgraph.intersects(&start.hairs) {
            return Err(HedgeError::InvalidStart);
        }

        let tree = TraversalTree::dfs(self, subgraph, start, included_hedge);

        let cuts = subgraph.subtract(&tree.tree);

        let mut cycle_basis = Vec::new();

        for c in cuts.included_iter() {
            if c > self.inv(c) {
                cycle_basis.push(tree.cycle(c).unwrap());
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
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn count_connected_components<S: SubGraph>(&self, subgraph: &S) -> usize {
        self.connected_components(subgraph).len()
    }

    pub fn connected_components<S: SubGraph>(&self, subgraph: &S) -> Vec<BitVec> {
        let mut visited_edges = self.empty_subgraph::<BitVec>();

        let mut components = vec![];

        // Iterate over all edges in the subgraph
        for hedge_index in subgraph.included_iter() {
            if !visited_edges.includes(&hedge_index) {
                // Perform DFS to find all reachable edges from this edge

                //
                let root_node = self.node_hairs(hedge_index);
                let reachable_edges = TraversalTree::dfs(self, subgraph, root_node, None).covers();

                visited_edges.union_with(&reachable_edges);

                components.push(reachable_edges);
            }
        }
        components
    }
    pub fn align_superficial_to_underlying(&mut self) {
        for i in self.edge_store.involution.iter_idx() {
            let orientation = self.edge_store.involution.edge_data(i).orientation;
            if let Orientation::Reversed = orientation {
                self.edge_store.involution.flip_underlying(i);
            }
        }
    }

    pub fn align_to_tree_underlying(&mut self, tree: &TraversalTree) {
        for (i, p) in tree.parent_iter() {
            match p {
                Parent::Root => {
                    if tree.tree.includes(&i) {
                        self.edge_store.involution.set_as_sink(i)
                    } else {
                        self.edge_store.involution.set_as_source(i)
                    }
                }
                Parent::Hedge {
                    hedge_to_root,
                    traversal_order,
                } => {
                    if tree.tree.includes(&i) {
                        if *hedge_to_root == i {
                            self.edge_store.involution.set_as_source(i)
                        } else {
                            self.edge_store.involution.set_as_sink(i)
                        }
                    } else {
                        let tord = traversal_order;
                        if let Parent::Hedge {
                            traversal_order, ..
                        } = tree.connected_parent(i)
                        {
                            if tord > traversal_order {
                                self.edge_store.involution.set_as_sink(i);
                            }
                        }
                    }
                }
                Parent::Unset => {
                    println!("unset{i}");
                }
            }
        }
    }

    pub fn align_to_tree_superficial(&mut self, tree: &TraversalTree) {
        for (i, p) in tree.parent_iter() {
            match self.edge_store.involution.hedge_data_mut(i) {
                InvolutiveMapping::Identity { data, .. } => match p {
                    Parent::Root => {}
                    Parent::Hedge { hedge_to_root, .. } => {
                        if *hedge_to_root == i {
                            data.orientation = Orientation::Default;
                        } else {
                            data.orientation = Orientation::Reversed;
                        }
                    }
                    Parent::Unset => {}
                },
                InvolutiveMapping::Source { data, .. } => match p {
                    Parent::Root => {}
                    Parent::Hedge { hedge_to_root, .. } => {
                        if *hedge_to_root == i {
                            data.orientation = Orientation::Default;
                        } else {
                            data.orientation = Orientation::Reversed;
                        }
                    }
                    Parent::Unset => {}
                },
                _ => {}
            }
        }
    }
}

// Iterators
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &V)> {
        self.node_store.iter_nodes()
    }

    pub fn base_nodes_iter<'a>(&'a self) -> impl Iterator<Item = NodeIndex> + 'a {
        self.node_store.iter_node_id()
    }

    pub fn iter_edge_id<'a, S: SubGraph>(&'a self, subgraph: &'a S) -> EdgeIter<'a, E, V, S, N> {
        EdgeIter::new(self, subgraph)
    }

    pub fn iter_edges<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&'a E>)> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.edge_store.involution.smart_data(i, subgraph).map(|d| {
                (
                    HedgePair::from_half_edge_with_subgraph(
                        i,
                        &self.edge_store.involution,
                        subgraph,
                    )
                    .unwrap(),
                    d.data,
                    d.as_ref().map(|&a| &self[a]),
                )
            })
        })
    }

    pub fn iter_all_edges(&self) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&E>)> {
        self.edge_store
            .involution
            .iter_edge_data()
            .map(move |(i, d)| (self.hedge_pair(i), d.data, d.as_ref().map(|&a| &self[a])))
    }

    pub fn iter_internal_edge_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a E>> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.edge_store
                .involution
                .smart_data(i, subgraph)
                .map(|d| d.as_ref().map(|&a| &self[a]))
        })
    }

    pub fn iter_egde_node<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = &'a HedgeNode> + 'a {
        subgraph.included_iter().map(|i| self.node_hairs(i))
    }

    pub fn iter_node_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (&'a HedgeNode, &'a V)> {
        NodeIterator {
            graph: self,
            edges: subgraph.included_iter(),
            seen: bitvec![usize, Lsb0; 0; self.node_store.node_len()],
        }
    }
}

// Display
impl<E, V, N: NodeStorage<NodeData = V>> HedgeGraph<E, V, N> {
    pub fn dot_impl<S: SubGraph>(
        &self,
        node_as_graph: &S,
        graph_info: String,
        edge_attr: &impl Fn(&E) -> Option<String>,
        node_attr: &impl Fn(&V) -> Option<String>,
    ) -> String {
        node_as_graph.dot(self, graph_info, edge_attr, node_attr)
    }

    pub fn dot<S: SubGraph>(&self, node_as_graph: &S) -> String {
        self.dot_impl(node_as_graph, "start=2;\n".to_string(), &|_| None, &|_| {
            None
        })
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
impl<E, V, N: NodeStorage<NodeData = V>> Index<NodeIndex> for HedgeGraph<E, V, N> {
    type Output = V;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.node_store.get_node_data(index)
    }
}
impl<E, V, N: NodeStorage<NodeData = V>> IndexMut<NodeIndex> for HedgeGraph<E, V, N> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.node_store.get_node_data_mut(index)
    }
}
impl<E, V, N: NodeStorage<NodeData = V>> Index<&NodeIndex> for HedgeGraph<E, V, N> {
    type Output = HedgeNode;
    fn index(&self, index: &NodeIndex) -> &Self::Output {
        self.node_store.get_node(*index)
    }
}
impl<E, V, N: NodeStorage<NodeData = V>> Index<&HedgeNode> for HedgeGraph<E, V, N> {
    type Output = V;
    fn index(&self, index: &HedgeNode) -> &Self::Output {
        let id = self.id_from_hairs(index).unwrap();
        &self[id]
    }
}

impl<E, V, N: NodeStorage<NodeData = V>> IndexMut<&HedgeNode> for HedgeGraph<E, V, N> {
    fn index_mut(&mut self, index: &HedgeNode) -> &mut Self::Output {
        let id = self.id_from_hairs(index).unwrap();
        &mut self[id]
    }
}

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

impl<'a, E, V, I: Iterator<Item = Hedge>, N: NodeStorage<NodeData = V>> Iterator
    for NodeIterator<'a, E, V, N, I>
{
    type Item = (&'a HedgeNode, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.edges.next() {
            let node = self.graph.node_hairs(next);
            let node_pos = self.graph.id_from_hairs(node).unwrap().0;

            if self.seen[node_pos] {
                self.next()
            } else {
                self.seen.set(node_pos, true);
                Some((node, &self.graph[node]))
            }
        } else {
            None
        }
    }
}

#[cfg(feature = "symbolica")]
pub mod symbolica_interop;

use subgraph::{
    Cycle, HedgeNode, Inclusion, InternalSubGraph, OrientedCut, SubGraph, SubGraphHedgeIter,
    SubGraphOps,
};

use thiserror::Error;
use tree::{Parent, TraversalTree};

#[derive(Error, Debug)]
pub enum HedgeError {
    #[error("Invalid start node")]
    InvalidStart,
}

pub struct EdgeIter<'a, E, V, S, N: NodeStorage<NodeData = V>> {
    graph: &'a HedgeGraph<E, V, N>,
    included_iter: SubGraphHedgeIter<'a>,
    subgraph: &'a S,
}
impl<'a, E, V, S, N: NodeStorage<NodeData = V>> EdgeIter<'a, E, V, S, N>
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

impl<'a, E, V, S, N: NodeStorage<NodeData = V>> Iterator for EdgeIter<'a, E, V, S, N>
where
    S: SubGraph,
{
    type Item = (HedgePair, EdgeData<&'a E>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.included_iter.next()?;
        let orientation = self.graph.edge_store.involution.orientation(i);
        let data = &self.graph[self.graph[&i]];
        if let Some(e) = HedgePair::from_source_with_subgraph(
            i,
            &self.graph.edge_store.involution,
            self.subgraph,
        ) {
            Some((e, EdgeData::new(data, orientation)))
        } else {
            self.next()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Error)]
pub enum HedgeGraphError {
    #[error("Nodes do not partition")]
    NodesDoNotPartition,
    #[error("Invalid node")]
    NoNode,
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
}

pub mod hedgevec;
pub mod tree;

#[cfg(feature = "drawing")]
pub mod drawing;
#[cfg(feature = "layout")]
pub mod layout;
#[cfg(test)]
mod test_graphs;
#[cfg(test)]
mod tests;
