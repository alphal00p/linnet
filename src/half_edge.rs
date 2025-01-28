use core::panic;
use std::num::TryFromIntError;
use std::ops::{Index, IndexMut, Mul, Neg};
use std::{collections::VecDeque, fmt::Display, hash::Hash};

use ahash::{AHashMap, AHashSet};
use bitvec::{slice::IterOnes, vec::BitVec};
use indexmap::{IndexMap, IndexSet};
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
pub mod subgraph;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HedgeGraph<E, V> {
    node_data: Vec<V>,
    nodes: Vec<HedgeNode>,
    base_nodes: usize,
    pub involution: Involution<NodeIndex, E>, // Involution of half-edges
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

#[derive(Clone, Debug)]
pub struct HedgeNodeBuilder<V> {
    data: V,
    hedges: Vec<Hedge>,
}

#[derive(Clone, Debug)]
pub struct HedgeGraphBuilder<E, V> {
    nodes: Vec<HedgeNodeBuilder<V>>,
    involution: Involution<NodeIndex, E>,
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

// impl HedgeGraph<usize, usize> {
//     pub fn close_externals(value: &BareGraph) -> Self {
//         let mut builder = HedgeGraphBuilder::new();
//         let mut map = AHashMap::new();

//         let mut ext_map = AHashMap::new();

//         for (i, v) in value.vertices.iter().enumerate() {
//             if matches!(v.vertex_info, VertexInfo::InteractonVertexInfo(_)) {
//                 map.insert(i, builder.add_node(i));
//             }
//         }

//         for (i, edge) in value.edges.iter().enumerate() {
//             match edge.edge_type {
//                 EdgeType::Incoming => {
//                     let key = edge.particle.pdg_code;
//                     let sink = map[&edge.vertices[1]];
//                     ext_map
//                         .entry(key)
//                         .or_insert_with(|| (i, [None, Some(sink)]))
//                         .1[1] = Some(sink);
//                 }
//                 EdgeType::Outgoing => {
//                     let key = if edge.particle.is_self_antiparticle() {
//                         edge.particle.pdg_code
//                     } else {
//                         -edge.particle.pdg_code
//                     };
//                     let source = map[&edge.vertices[0]];
//                     ext_map
//                         .entry(key)
//                         .or_insert_with(|| (i, [Some(source), None]))
//                         .1[0] = Some(source);
//                 }
//                 EdgeType::Virtual => {
//                     let sink = map[&edge.vertices[1]];
//                     let source = map[&edge.vertices[0]];
//                     builder.add_edge(source, sink, i, true);
//                 }
//             }
//         }

//         for (_, (i, [source, sink])) in ext_map {
//             println!("{source:?},{sink:?}");
//             if let (Some(source), Some(sink)) = (source, sink) {
//                 builder.add_edge(source, sink, i, true);
//             }
//         }

//         builder.into()
//     }
// }

impl<E, V> HedgeGraphBuilder<E, V> {
    pub fn new() -> Self {
        HedgeGraphBuilder {
            nodes: Vec::new(),
            involution: Involution::new(),
        }
    }

    pub fn build(self) -> HedgeGraph<E, V> {
        self.into()
    }

    pub fn add_node(&mut self, data: V) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(HedgeNodeBuilder {
            data,
            hedges: Vec::new(),
        });
        NodeIndex(index)
    }

    pub fn add_edge(
        &mut self,
        source: NodeIndex,
        sink: NodeIndex,
        data: E,
        directed: impl Into<Orientation>,
    ) {
        let id = self.involution.add_pair(data, source, sink, directed);
        self.nodes[source.0].hedges.push(id.0);

        self.nodes[sink.0].hedges.push(id.1);
    }

    pub fn add_external_edge(
        &mut self,
        source: NodeIndex,
        data: E,
        orientation: impl Into<Orientation>,
        underlying: Flow,
    ) {
        let id = self
            .involution
            .add_identity(data, source, orientation, underlying);
        self.nodes[source.0].hedges.push(id);
    }
}

impl<E, V> Default for HedgeGraphBuilder<E, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E, V> From<HedgeGraphBuilder<E, V>> for HedgeGraph<E, V> {
    fn from(builder: HedgeGraphBuilder<E, V>) -> Self {
        let len = builder.involution.len();
        let nodes: Vec<HedgeNode> = builder
            .nodes
            .iter()
            .map(|x| (HedgeNode::from_builder(x, len)))
            .collect();

        let node_data: Vec<V> = builder.nodes.into_iter().map(|x| x.data).collect();
        let involution = Involution {
            inv: builder.involution.inv,
            hedge_data: builder.involution.hedge_data,
        };
        HedgeGraph {
            base_nodes: nodes.len(),
            nodes,
            node_data,
            involution,
        }
    }
}

#[cfg(feature = "symbolica")]
impl<N: Clone, E: Clone> From<symbolica::graph::Graph<N, E>> for HedgeGraph<E, N> {
    fn from(graph: symbolica::graph::Graph<N, E>) -> Self {
        let mut builder = HedgeGraphBuilder::new();
        let mut map = AHashMap::new();

        for (i, node) in graph.nodes().iter().enumerate() {
            map.insert(i, builder.add_node(node.data.clone()));
        }

        for edge in graph.edges() {
            let vertices = edge.vertices;
            let source = map[&vertices.0];
            let sink = map[&vertices.1];
            builder.add_edge(source, sink, edge.data.clone(), edge.directed);
        }

        builder.into()
    }
}

#[cfg(feature = "symbolica")]
impl<N: Clone, E: Clone> TryFrom<HedgeGraph<E, N>> for symbolica::graph::Graph<N, E> {
    type Error = HedgeGraphError;

    fn try_from(value: HedgeGraph<E, N>) -> Result<Self, Self::Error> {
        let mut graph = symbolica::graph::Graph::new();
        let mut map = AHashMap::new();

        for (i, node) in value.node_data.iter().enumerate() {
            map.insert(NodeIndex(i), graph.add_node(node.clone()));
        }

        for (i, node) in value.iter_egdes(&value.full_filter()) {
            if let HedgePair::Paired { source, sink } = i {
                let source = map[&value.node_id(source)];
                let sink = map[&value.node_id(sink)];

                let data = node.data.unwrap().clone();
                let orientation = node.orientation;

                match orientation {
                    Orientation::Default => {
                        graph
                            .add_edge(source, sink, true, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                    Orientation::Reversed => {
                        graph
                            .add_edge(sink, source, true, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                    Orientation::Undirected => {
                        graph
                            .add_edge(source, sink, false, data)
                            .map_err(HedgeGraphError::SymbolicaError)?;
                    }
                }
            } else {
                return Err(HedgeGraphError::HasIdentityHedge);
            }
        }

        Ok(graph)
    }
}

// impl From<&BareGraph> for HedgeGraph<usize, usize> {
//     fn from(value: &BareGraph) -> Self {
//         let mut builder = HedgeGraphBuilder::new();
//         let mut map = AHashMap::new();

//         for (i, _) in value.vertices.iter().enumerate() {
//             map.insert(i, builder.add_node(i));
//         }

//         for (i, edge) in value.edges.iter().enumerate() {
//             let source = map[&edge.vertices[0]];
//             let sink = map[&edge.vertices[1]];
//             builder.add_edge(source, sink, i, true);
//         }

//         builder.into()
//     }
// }

use subgraph::{
    Cycle, HedgeNode, Inclusion, InternalSubGraph, OrientedCut, SubGraph, SubGraphHedgeIter,
    SubGraphOps,
};

use thiserror::Error;

use crate::num_traits::RefZero;

// use crate::momentum::SignOrZero;

// use super::{BareGraph, EdgeType, VertexInfo};

#[derive(Error, Debug)]
pub enum HedgeError {
    #[error("Invalid start node")]
    InvalidStart,
}

pub struct EdgeIdIter<'a, E, V, S> {
    graph: &'a HedgeGraph<E, V>,
    included_iter: SubGraphHedgeIter<'a>,
    subgraph: &'a S,
}
impl<'a, E, V, S> EdgeIdIter<'a, E, V, S>
where
    S: SubGraph,
{
    pub fn new(graph: &'a HedgeGraph<E, V>, subgraph: &'a S) -> Self {
        EdgeIdIter {
            graph,
            subgraph,
            included_iter: subgraph.included_iter(),
        }
    }
}

impl<E, V, S> Iterator for EdgeIdIter<'_, E, V, S>
where
    S: SubGraph,
{
    type Item = HedgePair;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.included_iter.next()?;
        if let Some(e) =
            HedgePair::from_source_with_subgraph(i, &self.graph.involution, self.subgraph)
        {
            Some(e)
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
}

impl<E, V> HedgeGraph<E, V> {
    pub fn nodes<S: SubGraph>(&self, subgraph: &S) -> Vec<NodeIndex> {
        let mut nodes = IndexSet::new();
        for i in subgraph.included_iter() {
            let node = self.node_id(i);
            nodes.insert(node);
        }

        nodes.into_iter().collect()
    }

    pub fn connected_hedge(&self, hedge: Hedge) -> Hedge {
        self.involution.inv(hedge)
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = (&HedgeNode, &V)> {
        self.nodes.iter().zip(self.node_data.iter())
    }

    pub fn base_nodes_iter(&self) -> impl Iterator<Item = NodeIndex> {
        (0..self.base_nodes).map(NodeIndex)
    }

    /// Splits the edge that hedge is a part of into two dangling hedges, adding the data to the side given by hedge. The underlying orientation of the new edges is the same as the original edge, i.e. the source will now have `Flow::Source` and the sink will have `Flow::Sink`. The superficial orientation has to be given knowing this.
    pub fn split_edge(&mut self, hedge: Hedge, data: EdgeData<E>) -> Result<(), HedgeGraphError> {
        let (data_hedge, other) = match &mut self.involution[hedge] {
            InvolutiveMapping::Source { data, sink_idx } => {
                let old_data = data.take();
                (Some(old_data), *sink_idx)
            }
            InvolutiveMapping::Sink { source_idx } => (None, *source_idx),
            _ => return Err(HedgeGraphError::NoEdge),
        };

        if let Some(d) = data_hedge {
            self.involution.inv[hedge.0] = InvolutiveMapping::Identity {
                data,
                underlying: Flow::Source,
            };
            self.involution.inv[other.0] = InvolutiveMapping::Identity {
                data: d,
                underlying: Flow::Sink,
            };
        } else {
            self.involution.inv[hedge.0] = InvolutiveMapping::Identity {
                data,
                underlying: Flow::Sink,
            };

            let data = self.involution.edge_data_mut(other).take();

            self.involution.inv[other.0] = InvolutiveMapping::Identity {
                data,
                underlying: Flow::Source,
            };
        }

        Ok(())
    }

    /// Joins two graphs together, matching edges with the given function and merging them with the given function.
    /// The function `matching_fn` should return true if the two dangling half edges should be matched.
    /// The function `merge_fn` should return the new data for the merged edge, given the data of the two edges being merged.
    ///
    pub fn join(
        self,
        other: Self,
        matching_fn: impl Fn(Flow, EdgeData<&E>, Flow, EdgeData<&E>) -> bool,
        merge_fn: impl Fn(Flow, EdgeData<E>, Flow, EdgeData<E>) -> (Flow, EdgeData<E>),
    ) -> Result<Self, HedgeGraphError> {
        let self_empty_filter = self.empty_filter();
        let mut full_self = self.full_filter();
        let other_empty_filter = other.empty_filter();
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

        let self_inv_shift = self.involution.inv.len();

        let involution = Involution {
            inv: self
                .involution
                .inv
                .into_iter()
                .chain(other.involution.inv.into_iter().map(|i| match i {
                    InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink {
                        source_idx: Hedge(source_idx.0 + self_inv_shift),
                    },
                    InvolutiveMapping::Source { data, sink_idx } => InvolutiveMapping::Source {
                        data,
                        sink_idx: Hedge(sink_idx.0 + self_inv_shift),
                    },
                    a => a,
                }))
                .collect(),
            hedge_data: self
                .involution
                .hedge_data
                .into_iter()
                .chain(other.involution.hedge_data.into_iter().map(|mut a| {
                    a.0 += self_shift;
                    a
                }))
                .collect(),
        };

        let mut found_match = true;

        let mut g = HedgeGraph {
            base_nodes: self.base_nodes + other.base_nodes, // need to fix
            nodes,
            node_data,
            involution,
        };

        while found_match {
            let mut matching_ids = None;

            for i in full_self.included_iter() {
                if let InvolutiveMapping::Identity {
                    data: datas,
                    underlying: underlyings,
                } = &g.involution.inv[i.0]
                {
                    for j in full_other.included_iter() {
                        if let InvolutiveMapping::Identity { data, underlying } =
                            &g.involution.inv[j.0]
                        {
                            if matching_fn(*underlyings, datas.as_ref(), *underlying, data.as_ref())
                            {
                                matching_ids = Some((i, j));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((source, sink)) = matching_ids {
                g.involution.connect_identities(source, sink, &merge_fn);
            } else {
                found_match = false;
            }
        }

        g.check_and_set_nodes()?;

        Ok(g)
    }

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError> {
        let mut cover = self.empty_filter();
        for i in 0..self.base_nodes {
            let node = self.nodes.get(i).unwrap();
            for h in node.hairs.included_iter() {
                if cover.includes(&h) {
                    return Err(HedgeGraphError::NodesDoNotPartition);
                } else {
                    cover.set(h.0, true);
                    self.involution.set_hedge_data(h, NodeIndex(i));
                }
            }
        }

        if cover.sym_diff(&self.full_filter()).count_ones() > 0 {
            return Err(HedgeGraphError::NodesDoNotPartition);
        }

        Ok(())
    }

    /// Adds a dangling edge to the specified node with the given data and superficial orientation. /// The underlying orientation is always `Flow::Source`.
    pub fn add_dangling_edge(
        self,
        source: NodeIndex,
        data: E,
        orientation: impl Into<Orientation>,
    ) -> Result<(Hedge, Self), HedgeGraphError> {
        let mut involution = self.involution;
        let o = orientation.into();
        let mut data = Some(data);
        let mut hedge = None;

        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, mut k)| {
                k.internal_graph.filter.push(false);
                if NodeIndex(i) == source {
                    if let Some(d) = data.take() {
                        k.hairs.push(true);
                        hedge = Some(involution.add_identity(d, source, o, Flow::Source));
                    } else {
                        k.hairs.push(false);
                    }
                } else {
                    k.hairs.push(false);
                }

                k
            })
            .collect();

        let mut g = HedgeGraph {
            base_nodes: self.base_nodes,
            node_data: self.node_data,
            nodes,
            involution,
        };

        g.check_and_set_nodes()?;

        if let Some(hedge) = hedge {
            Ok((hedge, g))
        } else {
            Err(HedgeGraphError::NoNode)
        }
    }

    pub fn iter_edge_id<'a, S: SubGraph>(&'a self, subgraph: &'a S) -> EdgeIdIter<'a, E, V, S> {
        EdgeIdIter::new(self, subgraph)
    }

    // pub fn map_edges_ref<E2>(&self, f: &impl Fn(&E) -> E2) -> HedgeGraph<E2, V>
    // where
    //     V: Clone,
    // {
    //     let involution = self.involution.map_edge_data_ref(f);
    //     let nodes = self.nodes.clone();
    //     HedgeGraph {
    //         base_nodes: self.base_nodes,
    //         nodes,
    //         involution,
    //     }
    // }

    pub fn map_nodes_ref<V2>(&self, f: &impl Fn(&Self, &V, &HedgeNode) -> V2) -> HedgeGraph<E, V2>
    where
        E: Clone,
    {
        let involution = self.involution.clone();
        let node_data = self
            .node_data
            .iter()
            .zip(self.nodes.iter())
            .map(|(v, h)| f(self, v, h))
            .collect();
        HedgeGraph {
            base_nodes: self.base_nodes,
            node_data,
            nodes: self.nodes.clone(),
            involution,
        }
    }

    pub fn map<E2, V2>(
        self,
        f: impl FnMut(V) -> V2,
        g: impl FnMut(HedgePair, EdgeData<E>) -> EdgeData<E2>,
    ) -> HedgeGraph<E2, V2> {
        let involution = self.involution.map_edge_data(g);
        let node_data = self.node_data.into_iter().map(f).collect();
        HedgeGraph {
            base_nodes: self.base_nodes,
            nodes: self.nodes,
            node_data,
            involution,
        }
    }

    // pub fn map<E2, V2>(
    //     self,
    //     f: impl FnMut(V) -> V2,
    //     g: impl FnMut(EdgeId, EdgeData<E>) -> EdgeData<E2>,
    // ) -> HedgeGraph<E2, V2> {
    //     let involution = self.involution.map_edge_data(g);
    //     let node_data = self.node_data.into_iter().map(f).collect();
    //     HedgeGraph {
    //         base_nodes: self.base_nodes,
    //         nodes: self.nodes,
    //         node_data,
    //         involution,
    //     }
    // }

    pub fn iter_egde_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a E>> + 'a {
        subgraph
            .included_iter()
            .map(|i| self.involution.edge_data(i).as_ref())
    }

    pub fn iter_egdes<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = (HedgePair, EdgeData<&'a E>)> + 'a {
        subgraph.included_iter().flat_map(|i| {
            self.involution.smart_data(i, subgraph).map(|d| {
                (
                    HedgePair::from_half_edge_with_subgraph(i, &self.involution, subgraph).unwrap(),
                    d.as_ref(),
                )
            })
        })
    }

    pub fn iter_internal_edge_data<'a, S: SubGraph>(
        &'a self,
        subgraph: &'a S,
    ) -> impl Iterator<Item = EdgeData<&'a E>> + 'a {
        subgraph
            .included_iter()
            .flat_map(|i| self.involution.smart_data(i, subgraph).map(|d| d.as_ref()))
    }

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

    pub fn count_connected_components<S: SubGraph>(&self, subgraph: &S) -> usize {
        self.connected_components(subgraph).len()
    }

    pub fn node_hairs(&self, hedge: Hedge) -> &HedgeNode {
        self.hairs_from_id(*self.involution.hedge_data(hedge))
    }

    pub fn hairs_from_id(&self, id: NodeIndex) -> &HedgeNode {
        self.nodes.get(id.0).unwrap()
    }

    pub fn id_from_hairs(&self, id: &HedgeNode) -> Option<NodeIndex> {
        let e = id.hairs.included_iter().next()?;
        let nodeid = self.involution.hedge_data(e);

        Some(*nodeid)
    }

    pub fn involved_node_hairs(&self, hedge: Hedge) -> Option<&HedgeNode> {
        self.involution
            .involved_hedge_data(hedge)
            .map(|i| self.hairs_from_id(*i))
    }

    pub fn involved_node_id(&self, hedge: Hedge) -> Option<NodeIndex> {
        self.involution.involved_hedge_data(hedge).copied()
    }

    pub fn node_id(&self, hedge: Hedge) -> NodeIndex {
        *self.involution.hedge_data(hedge)
    }

    pub fn connected_components<S: SubGraph>(&self, subgraph: &S) -> Vec<BitVec> {
        let mut visited_edges = self.empty_filter();

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

    pub fn cyclotomatic_number(&self, subgraph: &InternalSubGraph) -> usize {
        let n_hedges = self.count_internal_edges(subgraph);
        // println!("n_hedges: {}", n_hedges);
        let n_nodes = self.number_of_nodes_in_subgraph(subgraph);
        // println!("n_nodes: {}", n_nodes);
        let n_components = self.count_connected_components(subgraph);
        // println!("n_components: {}", n_components);

        n_hedges - n_nodes + n_components
    }

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

        let mut current = self.empty_filter();
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

    /// including pos
    pub fn neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> BitVec {
        subgraph.hairs(self.node_hairs(pos))
    }

    pub fn connected_neighbors<S: SubGraph>(&self, subgraph: &S, pos: Hedge) -> Option<BitVec> {
        Some(subgraph.hairs(self.involved_node_hairs(pos)?))
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

    pub fn superficial_hedge_orientation(&self, hedge: Hedge) -> Option<Flow> {
        match &self.involution[hedge] {
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
        match &self.involution[hedge] {
            InvolutiveMapping::Identity { underlying, .. } => *underlying,
            InvolutiveMapping::Sink { .. } => Flow::Sink,
            InvolutiveMapping::Source { .. } => Flow::Source,
        }
    }

    pub fn random(nodes: usize, edges: usize, seed: u64) -> HedgeGraph<(), ()> {
        let mut inv: Involution<Option<NodeIndex>, ()> =
            Involution::<NodeIndex, ()>::random(edges, seed);

        let mut rng = SmallRng::seed_from_u64(seed);

        let mut externals = Vec::new();
        let mut sources = Vec::new();
        let mut sinks = Vec::new();

        for (i, e) in inv.inv.iter().enumerate() {
            let nodeid = HedgeNode::node_from_pos(&[i], inv.inv.len());
            match e {
                InvolutiveMapping::Identity { .. } => externals.push(nodeid),
                InvolutiveMapping::Source { .. } => sources.push(nodeid),
                InvolutiveMapping::Sink { .. } => sinks.push(nodeid),
            }
        }

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

        for (nid, n) in sources.into_iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(());
            for i in n.hairs.included_iter() {
                inv.set_hedge_data(i, Some(NodeIndex(nid)));
            }
        }

        let len = nodes.len();

        for (nid, n) in sinks.into_iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(());

            for i in n.hairs.included_iter() {
                inv.set_hedge_data(i, Some(NodeIndex(nid + len)));
            }
        }

        let new_inv = Involution {
            inv: inv.inv,
            hedge_data: inv.hedge_data.into_iter().map(|n| n.unwrap()).collect(),
        };

        HedgeGraph {
            base_nodes: nodes.len(),
            nodes,
            node_data,
            involution: new_inv,
        }
    }

    pub fn base_dot(&self) -> String {
        let mut out = "digraph {\n ".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";\n ");
        for (i, (e, n)) in self.involution.iter() {
            out.push_str(
                &e.default_dot(
                    i,
                    Some(n.0),
                    self.involved_node_hairs(i)
                        .map(|x| self.id_from_hairs(x).unwrap().0),
                    None,
                ),
            );
        }
        out += "}";
        out
    }

    pub fn base_dot_underlying(&self) -> String {
        let mut out = "digraph {\n ".to_string();
        out.push_str("  node [shape=circle,height=0.1,label=\"\"];  overlap=\"scale\";\n ");
        for (i, (e, n)) in self.involution.iter() {
            let m = match e {
                InvolutiveMapping::Identity { data, underlying } => {
                    let mut d = data.as_ref();
                    d.orientation = (*underlying).into();
                    InvolutiveMapping::Identity {
                        data: d,
                        underlying: *underlying,
                    }
                }
                InvolutiveMapping::Source { sink_idx, data } => {
                    let mut d = data.as_ref();
                    d.orientation = Orientation::Default;
                    InvolutiveMapping::Source {
                        sink_idx: *sink_idx,
                        data: d,
                    }
                }
                InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink {
                    source_idx: *source_idx,
                },
            };
            out.push_str(
                &m.default_dot(
                    i,
                    Some(n.0),
                    self.involved_node_hairs(i)
                        .map(|x| self.id_from_hairs(x).unwrap().0),
                    None,
                ),
            );
        }
        out += "}";
        out
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

    pub fn paired_filter_from_pos(&self, pos: &[Hedge]) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.involution.len()];

        for &i in pos {
            filter.set(i.0, true);
            filter.set(self.involution.inv(i).0, true);
        }

        filter
    }

    // pub fn filter_from_pos(&self, pos: &[usize]) -> BitVec {
    //     Nested<T>::filter_from_pos(pos, self.involution.len())
    // }

    // pub fn nesting_node_from_pos(&self, pos: &[usize]) -> Nested<T> {
    //     self.nesting_node_from_subgraph(SubGraph::from(self.filter_from_pos(pos)))
    // }

    fn remove_externals(&self, subgraph: &mut HedgeNode) {
        let externals = self.external_filter();

        subgraph.internal_graph.filter &= !externals;
    }

    pub fn external_filter(&self) -> BitVec {
        let mut filter = bitvec![usize, Lsb0; 0; self.involution.len()];

        for (i, edge) in self.involution.inv.iter().enumerate() {
            if edge.is_identity() {
                filter.set(i, true);
            }
        }

        filter
    }

    pub fn full_filter(&self) -> BitVec {
        bitvec![usize, Lsb0; 1; self.involution.len()]
    }

    pub fn empty_filter(&self) -> BitVec {
        bitvec![usize, Lsb0; 0; self.involution.len()]
    }

    pub fn clean_subgraph(&self, filter: BitVec) -> InternalSubGraph {
        InternalSubGraph::cleaned_filter_optimist(filter, self)
    }

    pub fn full_node(&self) -> HedgeNode {
        self.nesting_node_from_subgraph(self.full_graph())
    }

    pub fn cycle_basis(&self) -> (Vec<Cycle>, TraversalTree) {
        self.paton_cycle_basis(&self.full_graph(), self.node_hairs(Hedge(0)), None)
            .unwrap()
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
        for (i, (_, p)) in tree.inv.iter() {
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
                        let tord = *traversal_order;
                        if let Parent::Hedge {
                            traversal_order, ..
                        } = tree.inv.hedge_data(tree.inv.inv(i))
                        {
                            if tord > *traversal_order {
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
        for (i, (_, p)) in tree.inv.iter() {
            match &mut self.involution[i] {
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

    pub fn full_graph(&self) -> InternalSubGraph {
        InternalSubGraph::cleaned_filter_optimist(self.full_filter(), self)
    }

    pub fn empty_subgraph<S: SubGraph>(&self) -> S {
        S::empty(self.n_hedges())
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

    pub fn hairy_from_filter(&self, filter: BitVec) -> HedgeNode {
        self.nesting_node_from_subgraph(InternalSubGraph::cleaned_filter_pessimist(filter, self))
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

    pub fn get_edge_data(&self, edge: Hedge) -> &E {
        self.involution.edge_data(edge).data.as_ref().unwrap()
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

    pub fn new_derived_edge_data<T>(&self, f: &impl Fn(&E) -> T) -> HedgeVec<T> {
        HedgeVec::mapped_from_involution(&self.involution, f)
    }

    pub fn new_derived_edge_data_empty<T>(&self) -> HedgeVec<T> {
        HedgeVec::empty_from_involution(&self.involution)
    }
}

// Data stored once per edge (pair of half-edges or external edge)
pub struct HedgeVec<T> {
    // involution:Vec<(Hedge,Option<usize>)
    data: Vec<InvolutiveMapping<T>>,
}

impl<T> HedgeVec<T> {
    fn inv(&self, hedge: Hedge) -> Hedge {
        match self.data[hedge.0] {
            InvolutiveMapping::Sink { source_idx } => source_idx,
            InvolutiveMapping::Source { sink_idx, .. } => sink_idx,
            InvolutiveMapping::Identity { .. } => hedge,
        }
    }

    fn mapped_from_involution<N, E>(involution: &Involution<N, E>, f: &impl Fn(&E) -> T) -> Self {
        let data = involution
            .iter()
            .map(|(_, (e, d))| e.map_data_ref(f))
            .collect();
        HedgeVec { data }
    }

    fn empty_from_involution<N, E>(involution: &Involution<N, E>) -> Self {
        let data = involution.iter().map(|(_, (e, _))| e.map_empty()).collect();
        HedgeVec { data }
    }

    fn data_inv(&self, hedge: Hedge) -> Hedge {
        match self.data[hedge.0] {
            InvolutiveMapping::Sink { source_idx } => source_idx,
            _ => hedge,
        }
    }

    pub fn is_set(&self, hedge: Hedge) -> bool {
        self.get(hedge).is_set()
    }

    fn get(&self, hedge: Hedge) -> EdgeData<&T> {
        let invh = self.data_inv(hedge);

        match &self.data[invh.0] {
            InvolutiveMapping::Identity { data, .. } => data.as_ref(),
            InvolutiveMapping::Source { data, .. } => data.as_ref(),
            _ => panic!("should have gotten data inv"),
        }
    }
}

impl<T> IndexMut<Hedge> for HedgeVec<T> {
    fn index_mut(&mut self, index: Hedge) -> &mut Self::Output {
        let invh = self.data_inv(index);

        match &mut self.data[invh.0] {
            InvolutiveMapping::Identity { data, .. } => &mut data.data,
            InvolutiveMapping::Source { data, .. } => &mut data.data,
            _ => panic!("should have gotten data inv"),
        }
    }
}
impl<T> Index<Hedge> for HedgeVec<T> {
    type Output = Option<T>;
    fn index(&self, index: Hedge) -> &Self::Output {
        let invh = self.data_inv(index);

        match &self.data[invh.0] {
            InvolutiveMapping::Identity { data, .. } => &data.data,
            InvolutiveMapping::Source { data, .. } => &data.data,
            _ => panic!("should have gotten data inv"),
        }
    }
}

pub struct TraversalTree {
    pub traversal: Vec<Hedge>,
    inv: Involution<Parent, ()>, // essentially just a vec of Parent that is the same length as the vec of hedges
    pub tree: InternalSubGraph,
    pub covers: BitVec,
}

pub enum Parent {
    Unset,
    Root,
    Hedge {
        hedge_to_root: Hedge,
        traversal_order: usize,
    },
}

impl TraversalTree {
    pub fn children(&self, hedge: Hedge) -> BitVec {
        let mut children = <BitVec as SubGraph>::empty(self.inv.inv.len());

        for (i, m) in self.inv.hedge_data.iter().enumerate() {
            match m {
                Parent::Hedge { hedge_to_root, .. } => {
                    if *hedge_to_root == hedge {
                        children.set(i, true);
                    }
                }
                _ => {}
            }
        }

        children.set(hedge.0, false);

        children
    }

    pub fn covers(&self) -> BitVec {
        let mut covers = <BitVec as SubGraph>::empty(self.inv.inv.len());
        for (i, m) in self.inv.hedge_data.iter().enumerate() {
            match m {
                Parent::Unset => {}
                _ => {
                    covers.set(i, true);
                }
            }
        }
        covers
    }

    pub fn leaf_edges(&self) -> BitVec {
        let mut leaves = <BitVec as SubGraph>::empty(self.inv.inv.len());
        for hedge in self.covers().included_iter() {
            let is_not_parent = !self.inv.iter().any(|(_, (_, p))| {
                if let Parent::Hedge { hedge_to_root, .. } = p {
                    *hedge_to_root == hedge
                } else {
                    false
                }
            });
            if is_not_parent {
                leaves.set(hedge.0, true);
            }
        }
        leaves
    }

    pub fn leaf_nodes<N, E>(&self, graph: &HedgeGraph<N, E>) -> Vec<NodeIndex> {
        let mut leaves = IndexSet::new();

        for hedge in self.covers().included_iter() {
            if let Parent::Hedge { hedge_to_root, .. } = self.inv.hedge_data(hedge) {
                if *hedge_to_root == hedge {
                    let mut sect = self
                        .tree
                        .filter
                        .intersection(&graph.node_hairs(hedge).hairs);

                    sect.set(hedge.0, false);

                    if sect.count_ones() == 0 {
                        leaves.insert(graph.node_id(hedge));
                    }
                }
            }
        }

        leaves.into_iter().collect()
    }

    /// Parent of a hedge, if is a `Parent::Hedge` then the hedge to root is in the same node,
    /// but points towards the root
    pub fn parent(&self, hedge: Hedge) -> &Parent {
        &self.inv.hedge_data(hedge)
    }

    pub fn parent_node<N, E>(
        &self,
        child: NodeIndex,
        graph: &HedgeGraph<N, E>,
    ) -> Option<NodeIndex> {
        let any_hedge = graph
            .hairs_from_id(child)
            .hairs
            .included_iter()
            .next()
            .unwrap();

        if let Parent::Hedge { hedge_to_root, .. } = self.parent(any_hedge) {
            return Some(graph.node_id(self.inv.inv(*hedge_to_root)));
        };

        None
    }

    pub fn child_nodes<N, E>(&self, parent: NodeIndex, graph: &HedgeGraph<N, E>) -> Vec<NodeIndex> {
        let mut children = IndexSet::new();

        for h in graph.hairs_from_id(parent).hairs.included_iter() {
            if let Parent::Hedge { hedge_to_root, .. } = self.parent(h) {
                if *hedge_to_root != h {
                    if let Some(c) = graph.involved_node_id(h) {
                        children.insert(c);
                    }
                }
            }
        }

        children.into_iter().collect()
    }

    fn path_to_root(&self, start: Hedge) -> BitVec {
        let mut path = <BitVec as SubGraph>::empty(self.inv.inv.len());
        let mut current = start;
        path.set(current.0, true);

        while let Parent::Hedge { hedge_to_root, .. } = self.inv.hedge_data(current) {
            path.set(hedge_to_root.0, true);
            current = self.inv.inv(*hedge_to_root);
            path.set(current.0, true);
        }
        path
    }

    pub fn cycle(&self, cut: Hedge) -> Option<Cycle> {
        match self.inv.hedge_data(cut) {
            Parent::Hedge { hedge_to_root, .. } => {
                if *hedge_to_root == cut {
                    //if cut is in the tree, no cycle can be formed
                    return None;
                }
            }
            Parent::Root => {}
            _ => return None,
        }

        let cut_pair = self.inv.inv(cut);
        match self.inv.hedge_data(cut_pair) {
            Parent::Hedge { hedge_to_root, .. } => {
                if *hedge_to_root == cut {
                    //if cut is in the tree,no cycle can be formed
                    return None;
                }
            }
            Parent::Root => {}
            _ => return None,
        }

        let mut cycle = self.path_to_root(cut);
        cycle.sym_diff_with(&self.path_to_root(cut_pair));
        let mut cycle = Cycle::new_unchecked(cycle);
        cycle.loop_count = Some(1);

        Some(cycle)
    }

    pub fn bfs<E, V, S: SubGraph>(
        graph: &HedgeGraph<E, V>,
        subgraph: &S,
        root_node: &HedgeNode,
        // target: Option<&HedgeNode>,
    ) -> Self {
        let mut queue = VecDeque::new();
        let mut seen = subgraph.hairs(root_node);

        let mut traversal: Vec<Hedge> = Vec::new();
        let mut involution: Involution<Parent, ()> = graph
            .involution
            .forgetful_map_node_data_ref(|_| Parent::Unset);

        // add all hedges from root node that are not self loops
        // to the queue
        // They are all potential branches
        for i in seen.included_iter() {
            involution.set_hedge_data(i, Parent::Root);
            if !seen.includes(&graph.involution.inv(i)) {
                // if not self loop
                queue.push_back(i)
            }
        }
        while let Some(hedge) = queue.pop_front() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = involution.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    //  a new node in the traversal
                    traversal.push(connected);
                } else {
                    continue;
                }
                // mark the new node as seen
                seen.union_with(&cn);

                // for all hedges in this new node, they have a parent, the initial hedge
                for i in cn.included_iter() {
                    if let Parent::Unset = involution.hedge_data(i) {
                        involution.set_hedge_data(
                            i,
                            Parent::Hedge {
                                hedge_to_root: connected,
                                traversal_order: traversal.len(),
                            },
                        );
                    }
                    // if they lead to a new node, they are potential branches, add them to the queue
                    if !seen.includes(&involution.inv(i)) && subgraph.includes(&i) {
                        queue.push_back(i);
                    }
                }
            }
        }

        TraversalTree::new(graph, traversal, seen, involution)
    }

    pub fn new<E, V>(
        graph: &HedgeGraph<E, V>,
        traversal: Vec<Hedge>,
        covers: BitVec,
        inv: Involution<Parent, ()>,
    ) -> Self {
        let mut tree = graph.empty_filter();

        for (i, j) in traversal.iter().map(|x| (*x, inv.inv(*x))) {
            tree.set(i.0, true);
            tree.set(j.0, true);
        }

        TraversalTree {
            traversal,
            covers,
            inv,
            tree: InternalSubGraph::cleaned_filter_optimist(tree, graph),
        }
    }

    pub fn dfs<E, V, S: SubGraph>(
        graph: &HedgeGraph<E, V>,
        subgraph: &S,
        root_node: &HedgeNode,
        include_hegde: Option<Hedge>,
        // target: Option<&HedgeNode>,
    ) -> Self {
        let mut stack = Vec::new();
        let mut seen = subgraph.hairs(root_node);

        let mut traversal: Vec<Hedge> = Vec::new();
        let mut involution: Involution<Parent, ()> = graph
            .involution
            .forgetful_map_node_data_ref(|_| Parent::Unset);

        let mut included_hedge_is_possible = false;

        // add all hedges from root node that are not self loops
        // to the stack
        // They are all potential branches

        for i in seen.included_iter() {
            involution.set_hedge_data(i, Parent::Root);
            if !seen.includes(&graph.involution.inv(i)) {
                // if not self loop
                if let Some(hedge) = include_hegde {
                    if hedge != i {
                        stack.push(i);
                    } else {
                        println!("skipping{i}");
                        included_hedge_is_possible = true;
                    }
                } else {
                    stack.push(i);
                }
            }
        }

        if included_hedge_is_possible {
            stack.push(include_hegde.unwrap());
        }
        while let Some(hedge) = stack.pop() {
            // println!("looking at {hedge}");
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = involution.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    // is a new node in the traversal
                    traversal.push(connected);
                } else {
                    continue;
                }

                // mark the new node as seen
                seen.union_with(&cn);

                for i in cn.included_iter() {
                    if let Parent::Unset = involution.hedge_data(i) {
                        involution.set_hedge_data(
                            i,
                            Parent::Hedge {
                                hedge_to_root: connected,
                                traversal_order: traversal.len(),
                            },
                        );
                    }

                    if !seen.includes(&involution.inv(i)) && subgraph.includes(&i) {
                        stack.push(i);
                    }
                }
            }
        }

        TraversalTree::new(graph, traversal, seen, involution)
    }
}

#[cfg(feature = "drawing")]
pub mod drawing;
#[cfg(feature = "layout")]
pub mod layout;
#[cfg(test)]
mod tests;
