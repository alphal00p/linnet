use bitvec::{order::Lsb0, slice::IterOnes, vec::BitVec};

use crate::{
    half_edge::{
        builder::HedgeNodeBuilder,
        involution::{EdgeIndex, Hedge, Involution},
        subgraph::{
            BaseSubgraph, HedgeNode, Inclusion, InternalSubGraph, ModifySubgraph, SubGraph,
            SubGraphOps,
        },
        HedgeGraph, HedgeGraphError, NodeIndex,
    },
    tree::{
        parent_pointer::{PPNode, ParentPointerStore},
        Forest, RootData, RootId,
    },
};

use super::{NodeStorage, NodeStorageOps};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
/// An implementation of [`NodeStorage`] and [`NodeStorageOps`] that uses `Vec`s
/// and `BitVec`s to store node information and their incident half-edges.
///
/// This strategy is often straightforward but can have performance implications
/// for certain operations like node deletion or identification if it involves
/// frequent re-indexing or large data movements.
///
/// # Type Parameters
///
/// - `N`: The type of custom data associated with each node.
///
/// # Fields
///
/// - `node_data`: A `Vec<N>` storing the custom data for each node. The index
///   in this vector corresponds to a `NodeIndex`.
/// - `hedge_data`: A `Vec<NodeIndex>` where the index of the vector is a `Hedge`'s
///   underlying `usize` value, and the element at that index is the `NodeIndex`
///   to which the hedge is incident.
/// - `nodes`: A `Vec<BitVec>`. Each index in the outer `Vec` corresponds to a
///   `NodeIndex`. The `BitVec` at that index is a bitmask representing the set
///   of half-edges incident to that node.
pub struct NodeStorageVec<N> {
    /// Stores the custom data for each node. Indexed by `NodeIndex.0`.
    pub(crate) node_data: Vec<N>,
    /// Maps each half-edge index (`Hedge.0`) to the `NodeIndex` it belongs to.
    pub(crate) hedge_data: Vec<NodeIndex>,
    /// For each node (indexed by `NodeIndex.0`), stores a `BitVec` representing
    /// the set of half-edges incident to it.
    #[cfg_attr(feature = "bincode", bincode(with_serde))]
    pub(crate) nodes: Vec<BitVec>, // Nodes
}

#[derive(Clone, Debug)]
/// An iterator that yields the [`Hedge`] identifiers incident to a node,
/// based on iterating over the set bits in a `BitVec`.
///
/// This is typically used by [`NodeStorageVec`] to provide an iterator
/// for its `NeighborsIter` associated type.
pub struct BitVecNeighborIter<'a> {
    /// The underlying iterator over set bits in the `BitVec`.
    iter_ones: IterOnes<'a, usize, Lsb0>,
    /// The total number of possible hedges (size of the `BitVec`), used for `ExactSizeIterator`.
    len: usize,
}

impl<'a> From<&'a BitVec> for BitVecNeighborIter<'a> {
    fn from(value: &'a BitVec) -> Self {
        Self {
            iter_ones: value.iter_ones(),
            len: value.len(),
        }
    }
}

impl<'a> From<BitVecNeighborIter<'a>> for BitVec {
    fn from(value: BitVecNeighborIter<'a>) -> Self {
        let len = value.len;
        BitVec::from_hedge_iter(value, len)
    }
}

impl<'a> From<BitVecNeighborIter<'a>> for HedgeNode {
    fn from(value: BitVecNeighborIter<'a>) -> Self {
        HedgeNode {
            internal_graph: InternalSubGraph::empty(value.len),
            hairs: value.into(),
        }
    }
}

impl Iterator for BitVecNeighborIter<'_> {
    type Item = Hedge;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter_ones.next().map(Hedge)
    }
}

impl ExactSizeIterator for BitVecNeighborIter<'_> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<N> NodeStorage for NodeStorageVec<N> {
    type NodeData = N;
    type Neighbors = BitVec;
    type NeighborsIter<'a>
        = BitVecNeighborIter<'a>
    where
        Self: 'a;
    type Storage<M> = NodeStorageVec<M>;
}

impl<N> NodeStorageVec<N> {
    fn swap_nodes(&mut self, a: NodeIndex, b: NodeIndex) {
        if a != b {
            for i in self.nodes[a.0].included_iter() {
                self.hedge_data[i.0] = b;
            }
            for i in self.nodes[b.0].included_iter() {
                self.hedge_data[i.0] = a;
            }
            self.node_data.swap(a.0, b.0);
            self.nodes.swap(a.0, b.0);
        }
    }

    fn from_hairs_and_data(node_data: Vec<N>, nodes: Vec<BitVec>) -> Option<Self> {
        let n_hedges = nodes[0].size();
        let mut hedge_data = vec![None; n_hedges];

        for (i, n) in nodes.iter().enumerate() {
            // println!("{:?}", n);
            for h in n.included_iter() {
                hedge_data[h.0] = Some(NodeIndex(i));
            }
        }
        Some(Self {
            node_data,
            hedge_data: hedge_data.into_iter().collect::<Option<Vec<_>>>()?,
            nodes,
        })
    }
}

impl<N> NodeStorageOps for NodeStorageVec<N> {
    type OpStorage<A> = Self::Storage<A>;
    type Base = BitVec;

    fn node_len(&self) -> usize {
        self.nodes.len()
    }

    fn swap(&mut self, a: Hedge, b: Hedge) {
        if a != b {
            let node_a = self.hedge_data[a.0];
            let node_b = self.hedge_data[b.0];

            self.hedge_data.swap(a.0, b.0);
            self.hedge_data[a.0] = node_b;
            self.hedge_data[b.0] = node_a;

            self.nodes[node_a.0].swap(a.0, b.0);
            self.nodes[node_b.0].swap(a.0, b.0);
        }
    }

    fn delete<S: SubGraph<Base = Self::Base>>(&mut self, subgraph: &S) {
        let mut left = Hedge(0);
        let mut extracted = Hedge(self.hedge_data.len());
        while left < extracted {
            if !subgraph.includes(&left) {
                //left is in the right place
                left.0 += 1;
            } else {
                //left needs to be swapped
                extracted.0 -= 1;
                if !subgraph.includes(&extracted) {
                    //only with an extracted that is in the wrong spot
                    self.swap(left, extracted);
                    left.0 += 1;
                }
            }
        }

        // println!("left{}", left);

        let mut left_nodes = NodeIndex(0);
        let mut extracted_nodes = NodeIndex(self.node_data.len());
        while left_nodes < extracted_nodes {
            if !self.nodes[left_nodes.0].has_greater(left) {
                //left is in the right place
                left_nodes.0 += 1;
            } else {
                //left needs to be swapped
                extracted_nodes.0 -= 1;
                if !self.nodes[extracted_nodes.0].has_greater(left) {
                    //only with an extracted that is in the wrong spot
                    self.swap_nodes(left_nodes, extracted_nodes);
                    left_nodes.0 += 1;
                }
            }
        }

        let mut overlapping_nodes = left_nodes;
        let mut non_overlapping_extracted = NodeIndex(self.node_len());

        while overlapping_nodes < non_overlapping_extracted {
            if self.nodes[overlapping_nodes.0].intersects(&(..left)) {
                //overlapping is in the right place, as it intersects (is after left_nodes) but isn't fully included
                overlapping_nodes.0 += 1;
            } else {
                //overlapping needs to be swapped
                non_overlapping_extracted.0 -= 1;
                if self.nodes[non_overlapping_extracted.0].intersects(&(..left)) {
                    //only with an extracted that is in the wrong spot
                    self.swap_nodes(overlapping_nodes, non_overlapping_extracted);
                    overlapping_nodes.0 += 1;
                }
            }
        }

        let _ = self.nodes.split_off(overlapping_nodes.0);
        let _ = self.node_data.split_off(overlapping_nodes.0);
        let _ = self.hedge_data.split_off(left.0);

        for i in 0..(left_nodes.0) {
            let _ = self.nodes[i].split_off(left.0);
            // self.nodes[i].internal_graph.filter.split_off(left.0);

            // split == 0;
        }
        for i in (left_nodes.0)..(overlapping_nodes.0) {
            let _ = self.nodes[i].split_off(left.0);
        }
    }

    fn extract<S: SubGraph<Base = BitVec>, V2>(
        &mut self,
        subgraph: &S,
        mut split_node: impl FnMut(&Self::NodeData) -> V2,
        owned_node: impl FnMut(Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2> {
        let mut left = Hedge(0);
        let mut extracted = Hedge(self.hedge_data.len());
        while left < extracted {
            if !subgraph.includes(&left) {
                //left is in the right place
                left.0 += 1;
            } else {
                //left needs to be swapped
                extracted.0 -= 1;
                if !subgraph.includes(&extracted) {
                    //only with an extracted that is in the wrong spot
                    self.swap(left, extracted);
                    left.0 += 1;
                }
            }
        }

        // println!("left{}", left);

        let mut left_nodes = NodeIndex(0);
        let mut extracted_nodes = NodeIndex(self.node_data.len());
        while left_nodes < extracted_nodes {
            if !self.nodes[left_nodes.0].has_greater(left) {
                //left is in the right place
                left_nodes.0 += 1;
            } else {
                //left needs to be swapped
                extracted_nodes.0 -= 1;
                if !self.nodes[extracted_nodes.0].has_greater(left) {
                    //only with an extracted that is in the wrong spot
                    self.swap_nodes(left_nodes, extracted_nodes);
                    left_nodes.0 += 1;
                }
            }
        }

        let mut overlapping_nodes = left_nodes;
        let mut non_overlapping_extracted = NodeIndex(self.node_len());

        while overlapping_nodes < non_overlapping_extracted {
            if self.nodes[overlapping_nodes.0].intersects(&(..left)) {
                //overlapping is in the right place, as it intersects (is after left_nodes) but isn't fully included
                overlapping_nodes.0 += 1;
            } else {
                //overlapping needs to be swapped
                non_overlapping_extracted.0 -= 1;
                if self.nodes[non_overlapping_extracted.0].intersects(&(..left)) {
                    //only with an extracted that is in the wrong spot
                    self.swap_nodes(overlapping_nodes, non_overlapping_extracted);
                    overlapping_nodes.0 += 1;
                }
            }
        }

        let mut extracted_nodes = self.nodes.split_off(overlapping_nodes.0);
        let mut extracted_data: Vec<_> = self
            .node_data
            .split_off(overlapping_nodes.0)
            .into_iter()
            .map(owned_node)
            .collect();

        let _ = self.hedge_data.split_off(left.0);

        let mut overlapping_node_hairs = vec![];
        let mut overlapping_data = vec![];

        for i in 0..(left_nodes.0) {
            let _ = self.nodes[i].split_off(left.0);
            // self.nodes[i].internal_graph.filter.split_off(left.0);

            // split == 0;
        }
        for i in (left_nodes.0)..(overlapping_nodes.0) {
            overlapping_data.push(split_node(&self.node_data[i]));
            // println!("og {}", self.nodes[i].nhedges());

            let overlapped = self.nodes[i].split_off(left.0);
            // println!("overlapped {}", overlapped.nhedges());
            overlapping_node_hairs.push(overlapped);
        }

        for h in &mut extracted_nodes {
            // println!("Init nhedges {}", h.nhedges());
            *h = h.split_off(left.0);

            // println!("After nhedges {}", h.nhedges());
        }

        extracted_nodes.extend(overlapping_node_hairs);
        extracted_data.extend(overlapping_data);

        NodeStorageVec::from_hairs_and_data(extracted_data, extracted_nodes)
            .expect("Extracted nodes should cover extracted hedges")
    }

    // fn add_node(&mut self, node_data: Self::NodeData) -> NodeIndex {
    //     let empty = HedgeNode::empty(self.hedge_len());
    //     self.nodes.push(empty);
    //     self.node_data.push(node_data);
    //     NodeIndex(self.nodes.len() - 1)
    // }

    fn identify_nodes(
        &mut self,
        nodes: &[NodeIndex],
        node_data_merge: Self::NodeData,
    ) -> NodeIndex {
        let mut removed = BitVec::empty(self.nodes.len());
        let mut full_node = BitVec::empty(self.hedge_len());

        for n in nodes {
            removed.set(n.0, true);
            full_node.union_with(&self.nodes[n.0]);
        }

        let replacement = NodeIndex(removed.iter_ones().next().unwrap());

        for r in removed.iter_ones().skip(1).rev() {
            // let last_index = self.nodes.len() - 1;

            // Before doing anything, update any hedge pointers that point to the node being removed.
            for hedge in self.hedge_data.iter_mut() {
                if *hedge == NodeIndex(r) {
                    *hedge = replacement;
                }
            }

            // if r != last_index {
            //     // Swap the target with the last element in both vectors.
            //     self.nodes.swap(r, last_index);
            //     self.node_data.swap(r, last_index);

            //     // After swapping, update any hedge pointer that pointed to the moved element.
            //     // It used to be at last_index, now it is at r.
            //     for hedge in self.hedge_data.iter_mut() {
            //         if *hedge == NodeIndex(last_index) {
            //             *hedge = NodeIndex(r);
            //         }
            //     }
            // }
            // // Remove the (now last) element.

            // self.nodes.pop();
            // self.node_data.pop();
        }

        self.nodes[replacement.0] = full_node;
        self.node_data[replacement.0] = node_data_merge;

        replacement
    }

    fn forget_identification_history(&mut self) -> Vec<Self::NodeData> {
        let mut to_keep = BitVec::empty(self.nodes.len());

        for h in &self.hedge_data {
            to_keep.add(Hedge(h.0));
        }

        for n in to_keep.iter_zeros() {
            self.nodes[n] = BitVec::empty(self.hedge_len());
        }

        let mut left_nodes = NodeIndex(0);
        let mut extracted_nodes = NodeIndex(self.node_data.len());
        while left_nodes < extracted_nodes {
            if to_keep[left_nodes.0] {
                //left is in the right place
                left_nodes.0 += 1;
            } else {
                //left needs to be swapped
                extracted_nodes.0 -= 1;
                if to_keep[extracted_nodes.0] {
                    //only with an extracted that is in the wrong spot
                    self.swap_nodes(left_nodes, extracted_nodes);
                    // self.nodes.swap(left_nodes.0, extracted_nodes.0);
                    left_nodes.0 += 1;
                }
            }
        }

        let _ = self.nodes.split_off(left_nodes.0);
        let a = self.node_data.split_off(left_nodes.0);
        a
    }

    fn to_forest<U, H>(
        &self,
        map_data: impl Fn(&Self::NodeData) -> U,
    ) -> Forest<U, ParentPointerStore<H>> {
        let mut nodes: Vec<_> = std::iter::repeat_with(|| None)
            .take(self.hedge_len())
            .collect();

        let mut roots = vec![];

        for (set, d) in self.nodes.iter().zip(&self.node_data) {
            let mut first = None;
            for i in set.included_iter() {
                if let Some(root) = first {
                    nodes[i.0] = Some(PPNode::dataless_child(root))
                } else {
                    first = Some(i.into());
                    nodes[i.0] = Some(PPNode::dataless_root(RootId(roots.len())));
                }
            }
            roots.push(RootData {
                root_id: first.unwrap(),
                data: map_data(d),
            });
        }
        Forest {
            nodes: nodes
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .unwrap()
                .into_iter()
                .collect(),
            roots,
        }
    }

    fn iter(&self) -> impl Iterator<Item = (NodeIndex, &Self::NodeData)> {
        self.node_data
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeIndex(i), v))
    }

    fn drain(self) -> impl Iterator<Item = (NodeIndex, Self::NodeData)> {
        self.node_data
            .into_iter()
            .enumerate()
            .map(|(i, v)| (NodeIndex(i), v))
    }
    fn build<I: IntoIterator<Item = HedgeNodeBuilder<N>>>(node_iter: I, n_hedges: usize) -> Self {
        let mut nodes: Vec<BitVec> = vec![];
        let mut node_data = vec![];
        let mut hedgedata = vec![None; n_hedges];

        for (i, n) in node_iter.into_iter().enumerate() {
            for h in &n.hedges {
                hedgedata[h.0] = Some(NodeIndex(i));
            }
            nodes.push(n.to_base(n_hedges));
            node_data.push(n.data);
        }

        let hedge_data = hedgedata.into_iter().map(|x| x.unwrap()).collect();

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn iter_nodes(
        &self,
    ) -> impl Iterator<Item = (NodeIndex, Self::NeighborsIter<'_>, &Self::NodeData)> {
        self.nodes
            .iter()
            .map(Into::into)
            .zip(self.node_data.iter())
            .zip(self.iter_node_id())
            .map(|((node, data), id)| (id, node, data))
    }

    fn iter_nodes_mut(
        &mut self,
    ) -> impl Iterator<Item = (NodeIndex, Self::NeighborsIter<'_>, &mut Self::NodeData)> {
        self.nodes
            .iter()
            .map(Into::into)
            .enumerate()
            .zip(self.node_data.iter_mut())
            .map(|((id, node), data)| (NodeIndex(id), node, data))
    }

    fn node_id_ref(&self, hedge: Hedge) -> NodeIndex {
        self.hedge_data[hedge.0]
    }

    // fn get_node(&self, node_id: NodeIndex) -> Self:: {
    //     &self.nodes[node_id.0]
    // }

    fn get_neighbor_iterator(&self, node_id: NodeIndex) -> Self::NeighborsIter<'_> {
        BitVecNeighborIter {
            iter_ones: self.nodes[node_id.0].iter_ones(),
            len: self.hedge_len(),
        }
    }

    fn get_node_data(&self, node_id: NodeIndex) -> &N {
        &self.node_data[node_id.0]
    }

    fn get_node_data_mut(&mut self, node_id: NodeIndex) -> &mut Self::NodeData {
        &mut self.node_data[node_id.0]
    }

    fn hedge_len(&self) -> usize {
        self.hedge_data.len()
    }

    fn extend(self, other: Self) -> Self {
        let self_empty_filter = BitVec::empty(self.hedge_data.len());
        let other_empty_filter = BitVec::empty(other.hedge_data.len());
        let mut node_data = self.node_data;
        node_data.extend(other.node_data);

        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .map(|mut k| {
                k.extend(other_empty_filter.clone());
                k
            })
            .chain(other.nodes.into_iter().map(|mut k| {
                let mut new_hairs = self_empty_filter.clone();
                new_hairs.extend(k.clone());
                k = new_hairs;

                k
            }))
            .collect();

        let mut hedge_data = self.hedge_data;
        hedge_data.extend(other.hedge_data);

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn extend_mut(&mut self, other: Self) {
        let self_empty_filter = BitVec::empty(self.hedge_data.len());
        let other_empty_filter = BitVec::empty(other.hedge_data.len());
        let node_data = &mut self.node_data;
        node_data.extend(other.node_data);

        for n in self.nodes.iter_mut() {
            n.extend(other_empty_filter.clone());
        }

        let nodes: Vec<_> = other
            .nodes
            .into_iter()
            .map(|mut k| {
                let mut new_hairs = self_empty_filter.clone();
                new_hairs.extend(k.clone());
                k = new_hairs;

                k
            })
            .collect();

        self.nodes.extend(nodes);

        self.hedge_data.extend(other.hedge_data);
    }
    fn add_dangling_edge(self, source: NodeIndex) -> Result<Self, HedgeGraphError> {
        if self.nodes.len() <= source.0 {
            return Err(HedgeGraphError::NoNode);
        }
        let nodes: Vec<_> = self
            .nodes
            .into_iter()
            .enumerate()
            .map(|(i, mut k)| {
                if NodeIndex(i) == source {
                    k.push(true);
                } else {
                    k.push(false);
                }
                k
            })
            .collect();
        let mut hedge_data = self.hedge_data;
        hedge_data.push(source);

        Ok(NodeStorageVec {
            node_data: self.node_data,
            hedge_data,
            nodes,
        })
    }

    fn random(sources: &[Self::Neighbors], sinks: &[Self::Neighbors]) -> Self
    where
        N: Default,
    {
        let mut nodes = Vec::new();
        let mut node_data = Vec::new();

        let mut hedge_data = vec![NodeIndex(0); sources[0].nhedges()];

        for (nid, n) in sources.iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(N::default());
            for i in n.included_iter() {
                hedge_data[i.0] = NodeIndex(nid);
            }
        }

        let len = nodes.len();

        for (nid, n) in sinks.iter().enumerate() {
            nodes.push(n.clone());
            node_data.push(N::default());

            for i in n.included_iter() {
                hedge_data[i.0] = NodeIndex(nid + len);
            }
        }

        NodeStorageVec {
            node_data,
            hedge_data,
            nodes,
        }
    }

    fn check_and_set_nodes(&mut self) -> Result<(), HedgeGraphError> {
        let mut cover = BitVec::empty(self.hedge_len());
        for (i, node) in self.nodes.iter().enumerate() {
            for h in node.included_iter() {
                if cover.includes(&h) {
                    return Err(HedgeGraphError::NodesDoNotPartition(format!(
                        "They overlap. Cover:{cover:?}, crown: {h:?}"
                    )));
                } else {
                    cover.set(h.0, true);
                    self.hedge_data[h.0] = NodeIndex(i);
                }
            }
        }

        let full = !BitVec::empty(self.hedge_len());

        if cover.sym_diff(&full).count_ones() > 0 {
            return Err(HedgeGraphError::NodesDoNotPartition(format!(
                "They do not cover the whole graph: cover {cover:?}"
            )));
        }

        Ok(())
    }

    fn map_data_ref_graph<'a, E, V2>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        mut node_map: impl FnMut(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            Self::NeighborsIter<'a>,
            &'a Self::NodeData,
        ) -> V2,
    ) -> Self::OpStorage<V2> {
        let node_data = self
            .node_data
            .iter()
            .zip(self.nodes.iter())
            .map(|(v, h)| node_map(graph, h.into(), v))
            .collect();

        NodeStorageVec {
            node_data,
            hedge_data: self.hedge_data.clone(),
            nodes: self.nodes.clone(),
        }
    }

    // fn map_data_ref_graph<'a, E, V2>(
    //     &'a self,
    //     graph: &'a HedgeGraph<E, Self::NodeData, Self>,
    //     mut node_map: impl FnMut(
    //         &'a HedgeGraph<E, Self::NodeData, Self>,
    //         &'a HedgeNode,
    //         &'a Self::NodeData,
    //     ) -> V2,
    // ) -> Self::Storage<V2> {
    //     let node_data = self
    //         .node_data
    //         .iter()
    //         .zip(self.nodes.iter())
    //         .map(|(v, h)| node_map(graph, h, v))
    //         .collect();

    //     NodeStorageVec {
    //         node_data,
    //         hedge_data: self.hedge_data.clone(),
    //         nodes: self.nodes.clone(),
    //     }
    // }

    // fn map_data_ref_mut_graph<'a, V2>(
    //     &'a mut self,
    //     mut node_map: impl FnMut(&'a HedgeNode, &'a mut Self::NodeData) -> V2,
    // ) -> Self::Storage<V2> {
    //     let node_data = self
    //         .node_data
    //         .iter_mut()
    //         .zip(self.nodes.iter())
    //         .map(|(v, h)| node_map(h, v))
    //         .collect();

    //     NodeStorageVec {
    //         node_data,
    //         hedge_data: self.hedge_data.clone(),
    //         nodes: self.nodes.clone(),
    //     }
    // }
    fn map_data_ref_mut_graph<'a, V2>(
        &'a mut self,
        mut node_map: impl FnMut(Self::NeighborsIter<'a>, &'a mut Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2> {
        let node_data = self
            .node_data
            .iter_mut()
            .zip(self.nodes.iter())
            .map(|(v, h)| node_map(h.into(), v))
            .collect();

        NodeStorageVec {
            node_data,
            hedge_data: self.hedge_data.clone(),
            nodes: self.nodes.clone(),
        }
    }

    fn map_data_ref_graph_result<'a, E, V2, Er>(
        &'a self,
        graph: &'a HedgeGraph<E, Self::NodeData, Self>,
        mut node_map: impl FnMut(
            &'a HedgeGraph<E, Self::NodeData, Self>,
            Self::NeighborsIter<'a>,
            &'a Self::NodeData,
        ) -> Result<V2, Er>,
    ) -> Result<Self::OpStorage<V2>, Er> {
        let node_data: Result<Vec<_>, Er> = self
            .node_data
            .iter()
            .zip(self.nodes.iter())
            .map(|(v, h)| node_map(graph, h.into(), v))
            .collect();

        Ok(NodeStorageVec {
            node_data: node_data?,
            hedge_data: self.hedge_data.clone(),
            nodes: self.nodes.clone(),
        })
    }

    fn map_data_graph<'a, V2>(
        self,
        involution: &'a Involution<EdgeIndex>,
        mut f: impl FnMut(&'a Involution<EdgeIndex>, NodeIndex, Self::NodeData) -> V2,
    ) -> Self::OpStorage<V2> {
        let node_data = self
            .node_data
            .into_iter()
            .enumerate()
            .map(|(i, v)| f(involution, NodeIndex(i), v))
            .collect();

        NodeStorageVec {
            node_data,
            hedge_data: self.hedge_data,
            nodes: self.nodes,
        }
    }
}
