use core::panic;
use std::hash::Hash;
use std::num::TryFromIntError;
use std::ops::{Index, IndexMut, Neg};

use ahash::{AHashMap, AHashSet};
use bitvec::{slice::IterOnes, vec::BitVec};
use builder::HedgeGraphBuilder;
use hedgevec::{HedgeVec, SmartHedgeVec};
use indexmap::IndexSet;
use involution::{
    EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Involution, InvolutionError, InvolutiveMapping,
    Orientation,
};
use itertools::Itertools;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use bitvec::prelude::*;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
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
pub mod subgraph;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HedgeGraph<E, V> {
    nodes: Vec<HedgeNode>,             // Nodes
    involution: Involution<EdgeIndex>, // Involution of half-edges
    base_nodes: usize,                 // Number of nodes in the base graph
    node_data: Vec<V>,                 // data associated with nodes, same length as nodes
    edge_data: Vec<(E, HedgePair)>,    // data associated with edges (!= half-edges)
    hedge_data: Vec<NodeIndex>,        // data associated with half-edges
}

impl<E, V> HedgeGraph<E, V> {
    /// Gives the involved hedge.
    /// If the hedge is a source, it will return the sink, and vice versa.
    /// If the hedge is an identity, it will return Itself.
    pub fn inv(&self, hedge: Hedge) -> Hedge {
        self.involution.inv(hedge)
    }

    /// Splits the edge that hedge is a part of into two dangling hedges, adding the data to the side given by hedge.
    /// The underlying orientation of the new edges is the same as the original edge, i.e. the source will now have `Flow::Source` and the sink will have `Flow::Sink`.
    /// The superficial orientation has to be given knowing this.
    pub fn split_edge(&mut self, hedge: Hedge, data: EdgeData<E>) -> Result<(), HedgeGraphError> {
        let new_data = EdgeData::new(EdgeIndex(self.edge_data.len()), data.orientation);
        let invh = self.inv(hedge);
        let flow = self.flow(hedge);
        self.involution.split_edge(hedge, new_data)?;

        self.edge_data
            .push((data.data, HedgePair::Unpaired { hedge, flow }));

        let invh_edge_id = self[&invh];
        self[invh_edge_id].1 = HedgePair::Unpaired {
            hedge: invh,
            flow: -flow,
        };
        Ok(())
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
        let self_empty_filter = self.empty_subgraph::<BitVec>();
        let mut full_self = self.full_filter();
        let other_empty_filter = other.empty_subgraph::<BitVec>();
        let mut full_other = self_empty_filter.clone();
        full_self.extend(other_empty_filter.clone());
        full_other.extend(other.full_filter());

        let mut node_data = self.node_data;
        node_data.extend(other.node_data);

        let self_shift = self.nodes.len();

        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .map(|mut k| {
                k.hairs.extend(other_empty_filter.clone());
                k.internal_graph.filter.extend(other_empty_filter.clone());
                k
            })
            .chain(other.nodes.into_iter().map(|mut k| {
                let mut new_hairs = self_empty_filter.clone();
                new_hairs.extend(k.hairs.clone());
                k.hairs = new_hairs;

                let mut internal = self_empty_filter.clone();
                internal.extend(k.internal_graph.filter.clone());
                k.internal_graph.filter = internal;

                k
            }))
            .collect();

        let self_inv_shift = self.involution.len();
        let mut edge_data = self.edge_data;
        let edge_data_shift = edge_data.len();
        edge_data.extend(other.edge_data);

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
                .map(|(i, m)| m)
                .collect(),
        };

        // hedge_data: self
        //     .collect(),
        //     .involution
        //     .hedge_data
        //     .into_iter()
        //     .chain(other.involution.hedge_data.into_iter().map(|mut a| {
        //         a.0 += self_shift;
        //         a
        //     }))
        //     .collect(),

        let mut found_match = true;

        let mut hedge_data = self.hedge_data;
        hedge_data.extend(other.hedge_data);

        let mut g = HedgeGraph {
            base_nodes: self.base_nodes + other.base_nodes, // need to fix
            nodes,
            hedge_data,
            node_data,
            edge_data,
            involution,
        };

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
                                datas.as_ref().map(|a| &g.edge_data[a.0].0),
                                *underlying,
                                data.as_ref().map(|a| &g.edge_data[a.0].0),
                            ) {
                                matching_ids = Some((i, j));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((source, sink)) = matching_ids {
                let source_edge_id = g.involution[source];
                let sink_edge_id = g.involution[sink];

                let last = g.edge_data.len().checked_sub(1).unwrap();
                let second_last = g.edge_data.len().checked_sub(2).unwrap();

                g.edge_data.swap(source_edge_id.0, last);
                g.edge_data.swap(sink_edge_id.0, second_last);

                let mut remaps: [Option<(EdgeIndex, EdgeIndex)>; 2] = [None, None];

                // If sink_edge_id.0 is already the last, swap it to second-last first.
                if sink_edge_id.0 == last {
                    g.edge_data.swap(sink_edge_id.0, second_last); // swap last and second last

                    g.edge_data.swap(source_edge_id.0, last);

                    // now we need to remap any pointers to second_last, to source_edge_id
                    remaps[0] = Some((EdgeIndex(second_last), source_edge_id));
                } else {
                    g.edge_data.swap(source_edge_id.0, last);
                    g.edge_data.swap(sink_edge_id.0, second_last);

                    if source_edge_id.0 == second_last {
                        remaps[0] = Some((EdgeIndex(last), sink_edge_id));
                    } else {
                        remaps[0] = Some((EdgeIndex(last), source_edge_id));
                        remaps[1] = Some((EdgeIndex(second_last), sink_edge_id));
                    }
                }

                let source_data = EdgeData::new(
                    g.edge_data.pop().unwrap().0,
                    g.involution.orientation(source),
                );
                let sink_data =
                    EdgeData::new(g.edge_data.pop().unwrap().0, g.involution.orientation(sink));
                let (merge_flow, merge_data) = merge_fn(
                    g.involution.flow(source),
                    source_data,
                    g.involution.flow(sink),
                    sink_data,
                );

                let new_edge_data =
                    EdgeData::new(EdgeIndex(g.edge_data.len()), merge_data.orientation);

                g.edge_data
                    .push((merge_data.data, HedgePair::Paired { source, sink }));

                g.involution
                    .connect_identities(source, sink, |_, _, _, _| (merge_flow, new_edge_data));

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
            } else {
                found_match = false;
            }
        }

        g.check_and_set_nodes()?;

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
        let mut involution = self.involution;
        let o = orientation.into();
        if self.nodes.len() <= source.0 {
            return Err(HedgeGraphError::NoNode);
        }

        let mut edge_data = self.edge_data;
        let edge_index = EdgeIndex(edge_data.len());
        let hedge = involution.add_identity(edge_index, o, flow);
        edge_data.push((data, HedgePair::Unpaired { hedge, flow }));
        let mut hedge_data = self.hedge_data;
        hedge_data.push(source);

        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, mut k)| {
                k.internal_graph.filter.push(false);
                if NodeIndex(i) == source {
                    k.hairs.push(true);
                } else {
                    k.hairs.push(false);
                }
                k
            })
            .collect();

        let mut g = HedgeGraph {
            base_nodes: self.base_nodes,
            edge_data,
            hedge_data,
            node_data: self.node_data,
            nodes,
            involution,
        };

        g.check_and_set_nodes()?;

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

    /// Creates a random graph with the given number of nodes and edges.
    pub fn random(nodes: usize, edges: usize, seed: u64) -> HedgeGraph<(), ()> {
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

        let mut nodes = Vec::new();
        let mut node_data = Vec::new();
        let mut hedge_data = vec![NodeIndex(0); inv.len()];

        for (nid, n) in sources.into_iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(());
            for i in n.hairs.included_iter() {
                hedge_data[i.0] = NodeIndex(nid);
            }
        }

        let len = nodes.len();

        for (nid, n) in sinks.into_iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(());

            for i in n.hairs.included_iter() {
                hedge_data[i.0] = NodeIndex(nid + len);
            }
        }

        let mut edge_data = vec![];
        let involution = inv.map_full(|a, b| {
            let edge_id = EdgeIndex(edge_data.len());
            edge_data.push(((), a));
            b.map(|d| edge_id)
        });

        HedgeGraph {
            base_nodes: nodes.len(),
            nodes,
            node_data,
            hedge_data,
            edge_data,
            involution,
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
                    subgraph.internal_graph.filter.set(
                        self.involution.inv(Hedge(int.first_one().unwrap())).0,
                        false,
                    );
                    has_branch = true;
                }
            }
        }

        self.nesting_node_fix(subgraph);
    }

    fn fix_hedge_pairs(&mut self) {
        for (i, d) in self.involution.iter_edge_data() {
            let hedge_pair = self.involution.hedge_pair(i);
            self.edge_data[d.data.0].1 = hedge_pair;
        }
    }
    fn set_hedge_data(&mut self, hedge: Hedge, nodeid: NodeIndex) {
        self.hedge_data[hedge.0] = nodeid;
    }

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError> {
        let mut cover = self.empty_subgraph::<BitVec>();
        for i in 0..self.base_nodes {
            let node = self.nodes.get(i).unwrap();
            for h in node.hairs.included_iter() {
                if cover.includes(&h) {
                    return Err(HedgeGraphError::NodesDoNotPartition);
                } else {
                    cover.set(h.0, true);
                    self.hedge_data[h.0] = NodeIndex(i);
                }
            }
        }

        if cover.sym_diff(&self.full_filter()).count_ones() > 0 {
            return Err(HedgeGraphError::NodesDoNotPartition);
        }

        Ok(())
    }
}

// Subgraphs
impl<E, V> HedgeGraph<E, V> {
    pub fn paired_filter_from_pos(&self, pos: &[Hedge]) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.involution.len()];

        for &i in pos {
            filter.set(i.0, true);
            filter.set(self.involution.inv(i).0, true);
        }

        filter
    }

    /// Bitvec subgraph of all external (identity) hedges
    pub fn external_filter(&self) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.involution.len()];

        for (i, edge) in self.involution.inv.iter().enumerate() {
            if edge.is_identity() {
                filter.set(i, true);
            }
        }

        filter
    }

    /// Bitvec subgraph of all hedges
    pub fn full_filter(&self) -> BitVec {
        bitvec![usize, Lsb0; 1; self.involution.len()]
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
        let mut hairs = bitvec![usize, Lsb0; 0; self.involution.len()];

        if !internal_graph.valid::<E, V>(self) {
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
        let mut externalhedges = bitvec![usize, Lsb0; 0; self.involution.len()];

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
impl<E, V> HedgeGraph<E, V> {
    pub fn count_internal_edges(&self, subgraph: &InternalSubGraph) -> usize {
        let mut internal_edge_count = 0;
        // Iterate over all half-edges in the subgraph
        for hedge_index in subgraph.included_iter() {
            let inv_hedge_index = self.involution.inv(hedge_index);

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
        self.involution.len()
    }

    pub fn n_nodes(&self) -> usize {
        self.base_nodes
    }

    pub fn n_externals(&self) -> usize {
        self.involution
            .inv
            .iter()
            .filter(|e| e.is_identity())
            .count()
    }

    pub fn n_internals(&self) -> usize {
        self.involution
            .inv
            .iter()
            .filter(|e| e.is_internal())
            .count()
            / 2
    }

    pub fn n_base_nodes(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_node()).count()
    }

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
impl<E, V> HedgeGraph<E, V> {
    /// including pos
    pub fn neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> BitVec {
        subgraph.hairs(self.node_hairs(pos))
    }

    pub fn connected_neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> Option<BitVec> {
        Some(subgraph.hairs(self.involved_node_hairs(pos)?))
    }
    pub fn get_edge_data(&self, edge: Hedge) -> &E {
        &self[self[&edge]].0
    }

    pub fn hedge_pair(&self, hedge: Hedge) -> HedgePair {
        self.involution.hedge_pair(hedge)
    }

    pub fn get_edge_data_full(&self, hedge: Hedge) -> EdgeData<&E> {
        let orientation = self.involution.orientation(hedge);
        EdgeData::new(&self[self[&hedge]].0, orientation)
    }

    pub fn orientation(&self, hedge_pair: HedgePair) -> Orientation {
        self.involution.orientation(hedge_pair.any_hedge())
    }

    /// Gives the underlying orientation of this half-edge.
    pub fn flow(&self, hedge: Hedge) -> Flow {
        self.involution.flow(hedge)
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

    pub fn node_hairs(&self, hedge: Hedge) -> &HedgeNode {
        self.hairs_from_id(self[hedge])
    }

    pub fn hairs_from_id(&self, id: NodeIndex) -> &HedgeNode {
        &self[&id]
    }

    pub fn id_from_hairs(&self, id: &HedgeNode) -> Option<NodeIndex> {
        let e = id.hairs.included_iter().next()?;
        Some(self[e])
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
        Some(self[invh])
    }

    pub fn node_id(&self, hedge: Hedge) -> NodeIndex {
        self[hedge]
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
impl<E, V> HedgeGraph<E, V> {
    pub fn map_data_ref<'a, E2, V2>(
        &'a self,
        node_map: &impl Fn(&'a Self, &'a V, &'a HedgeNode) -> V2,
        edge_map: &impl Fn(&'a Self, HedgePair, EdgeData<&'a E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2> {
        let mut involution = self.involution.clone();
        HedgeGraph {
            nodes: self.nodes.clone(),
            base_nodes: self.base_nodes,
            node_data: self
                .node_data
                .iter()
                .zip(self.nodes.iter())
                .map(|(v, h)| node_map(self, v, h))
                .collect(),
            edge_data: self
                .edge_data
                .iter()
                .map(|(e, h)| {
                    let new_edgedata = edge_map(self, *h, EdgeData::new(e, self.orientation(*h)));

                    involution.edge_data_mut(h.any_hedge()).orientation = new_edgedata.orientation;
                    (new_edgedata.data, *h)
                })
                .collect(),
            hedge_data: self.hedge_data.clone(),
            involution,
        }
    }

    pub fn map_nodes_ref<V2>(
        &self,
        f: &impl Fn(&Self, &V, &HedgeNode) -> V2,
    ) -> HedgeGraph<&E, V2> {
        self.map_data_ref(f, &|_, _, e| e)
    }
    pub fn map<E2, V2>(
        mut self,
        mut f: impl FnMut(&Involution<EdgeIndex>, &Vec<HedgeNode>, &Vec<NodeIndex>, NodeIndex, V) -> V2,
        mut g: impl FnMut(
            &Involution<EdgeIndex>,
            &Vec<HedgeNode>,
            &Vec<NodeIndex>,
            HedgePair,
            EdgeData<E>,
        ) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2> {
        HedgeGraph {
            base_nodes: self.base_nodes,
            node_data: self
                .node_data
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    f(
                        &self.involution,
                        &self.nodes,
                        &self.hedge_data,
                        NodeIndex(i),
                        v,
                    )
                })
                .collect(),
            edge_data: self
                .edge_data
                .into_iter()
                .map(|(e, h)| {
                    let new_data = g(
                        &self.involution,
                        &self.nodes,
                        &self.hedge_data,
                        h,
                        EdgeData::new(e, self.involution.orientation(h.any_hedge())),
                    );

                    self.involution.edge_data_mut(h.any_hedge()).orientation = new_data.orientation;
                    (new_data.data, h)
                })
                .collect(),
            involution: self.involution,
            nodes: self.nodes,
            hedge_data: self.hedge_data,
        }
    }
    pub fn new_smart_hedgevec<T>(
        &self,
        f: &impl Fn(HedgePair, EdgeData<&E>) -> EdgeData<T>,
    ) -> SmartHedgeVec<T> {
        let mut data = Vec::new();

        let involution = self.involution.clone().map_full(|a, _| {
            let d = self.get_edge_data_full(a.any_hedge());
            let new_data = f(a, d);
            let edgeid = EdgeIndex(data.len());
            data.push((a, new_data.data));
            EdgeData::new(edgeid, new_data.orientation)
        });
        SmartHedgeVec { data, involution }
    }

    pub fn new_hedgevec<T>(&self, f: &impl Fn(&E) -> T) -> HedgeVec<T> {
        let data = self.edge_data.iter().map(|(e, _)| f(e)).collect();

        HedgeVec(data)
    }
}

// Cuts
impl<E, V> HedgeGraph<E, V> {
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

        for i in (start.0..self.involution.len()).map(Hedge) {
            let j = self.involution.inv(i);
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
            let invh = self.involution.inv(h);

            if h > invh && s.hairs.includes(&self.involution.inv(h)) {
                new_internals.push(h);
            }
        }

        let mut new_node = s.clone();

        for h in new_internals {
            new_node.hairs.set(h.0, false);
            new_node.hairs.set(self.involution.inv(h).0, false);
            new_node.internal_graph.filter.set(h.0, true);
            new_node
                .internal_graph
                .filter
                .set(self.involution.inv(h).0, true);
        }

        let hairy = new_node.internal_graph.filter.union(&new_node.hairs);
        let complement = hairy.complement(self);
        let count = self.count_connected_components(&complement);

        if count == 1 && !regions.insert(new_node.internal_graph.clone()) {
            return;
        }

        for h in new_node.included_iter() {
            let invh = self.involution.inv(h);

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
impl<E, V> HedgeGraph<E, V> {
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
        Ok(self.involution.n_internals(&cuts))
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
            if c > self.involution.inv(c) {
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
impl<E, V> HedgeGraph<E, V> {
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
        for i in self.involution.iter_idx() {
            let orientation = self.involution.edge_data(i).orientation;
            if let Orientation::Reversed = orientation {
                self.involution.flip_underlying(i);
            }
        }
    }

    pub fn align_to_tree_underlying(&mut self, tree: &TraversalTree) {
        for (i, p) in tree.parent_iter() {
            match p {
                Parent::Root => {
                    if tree.tree.includes(&i) {
                        self.involution.set_as_sink(i)
                    } else {
                        self.involution.set_as_source(i)
                    }
                }
                Parent::Hedge {
                    hedge_to_root,
                    traversal_order,
                } => {
                    if tree.tree.includes(&i) {
                        if *hedge_to_root == i {
                            self.involution.set_as_source(i)
                        } else {
                            self.involution.set_as_sink(i)
                        }
                    } else {
                        let tord = traversal_order;
                        if let Parent::Hedge {
                            traversal_order, ..
                        } = tree.connected_parent(i)
                        {
                            if tord > traversal_order {
                                self.involution.set_as_sink(i);
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
            match self.involution.hedge_data_mut(i) {
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
impl<E, V> HedgeGraph<E, V> {
    pub fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &V)> {
        self.nodes.iter().zip(self.node_data.iter())
    }

    pub fn base_nodes_iter(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.base_nodes).map(NodeIndex)
    }

    pub fn iter_edge_id<'a, S: SubGraph>(&'a self, subgraph: &'a S) -> EdgeIter<'a, E, V, S> {
        EdgeIter::new(self, subgraph)
    }

    pub fn iter_edges<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&'a E>)> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.involution.smart_data(i, subgraph).map(|d| {
                (
                    HedgePair::from_half_edge_with_subgraph(i, &self.involution, subgraph).unwrap(),
                    d.data,
                    d.as_ref().map(|&a| &self[a].0),
                )
            })
        })
    }

    pub fn iter_all_edges(&self) -> impl Iterator<Item = (HedgePair, EdgeIndex, EdgeData<&E>)> {
        self.involution
            .iter_edge_data()
            .map(move |(i, d)| (self.hedge_pair(i), d.data, d.as_ref().map(|&a| &self[a].0)))
    }

    pub fn iter_internal_edge_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a E>> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.involution
                .smart_data(i, subgraph)
                .map(|d| d.as_ref().map(|&a| &self[a].0))
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
            seen: bitvec![usize, Lsb0; 0; self.base_nodes],
        }
    }
}

// Display
impl<E, V> HedgeGraph<E, V> {
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

impl<E, V> Index<Hedge> for HedgeGraph<E, V> {
    type Output = NodeIndex;
    fn index(&self, index: Hedge) -> &Self::Output {
        &self.hedge_data[index.0]
    }
}
impl<E, V> Index<&Hedge> for HedgeGraph<E, V> {
    type Output = EdgeIndex;
    fn index(&self, index: &Hedge) -> &Self::Output {
        &self.involution[*index]
    }
}
impl<E, V> Index<NodeIndex> for HedgeGraph<E, V> {
    type Output = V;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.node_data[index.0]
    }
}
impl<E, V> IndexMut<NodeIndex> for HedgeGraph<E, V> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        &mut self.node_data[index.0]
    }
}
impl<E, V> Index<&NodeIndex> for HedgeGraph<E, V> {
    type Output = HedgeNode;
    fn index(&self, index: &NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}
impl<E, V> Index<&HedgeNode> for HedgeGraph<E, V> {
    type Output = V;
    fn index(&self, index: &HedgeNode) -> &Self::Output {
        let id = self.id_from_hairs(index).unwrap();
        &self[id]
    }
}

impl<E, V> IndexMut<&HedgeNode> for HedgeGraph<E, V> {
    fn index_mut(&mut self, index: &HedgeNode) -> &mut Self::Output {
        let id = self.id_from_hairs(index).unwrap();
        &mut self[id]
    }
}

impl<E, V> Index<EdgeIndex> for HedgeGraph<E, V> {
    type Output = (E, HedgePair);
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edge_data[index.0]
    }
}

impl<E, V> IndexMut<EdgeIndex> for HedgeGraph<E, V> {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        &mut self.edge_data[index.0]
    }
}

pub struct NodeIterator<'a, E, V, I = IterOnes<'a, usize, Lsb0>> {
    graph: &'a HedgeGraph<E, V>,
    edges: I,
    seen: BitVec,
}

impl<'a, E, V, I: Iterator<Item = Hedge>> Iterator for NodeIterator<'a, E, V, I> {
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

pub struct EdgeIter<'a, E, V, S> {
    graph: &'a HedgeGraph<E, V>,
    included_iter: SubGraphHedgeIter<'a>,
    subgraph: &'a S,
}
impl<'a, E, V, S> EdgeIter<'a, E, V, S>
where
    S: SubGraph,
{
    pub fn new(graph: &'a HedgeGraph<E, V>, subgraph: &'a S) -> Self {
        EdgeIter {
            graph,
            subgraph,
            included_iter: subgraph.included_iter(),
        }
    }
}

impl<'a, E, V, S> Iterator for EdgeIter<'a, E, V, S>
where
    S: SubGraph,
{
    type Item = (HedgePair, EdgeData<&'a E>);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.included_iter.next()?;
        let orientation = self.graph.involution.orientation(i);
        let (data, _) = &self.graph[self.graph[&i]];
        if let Some(e) =
            HedgePair::from_source_with_subgraph(i, &self.graph.involution, self.subgraph)
        {
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
