use std::{cmp::Ordering, collections::VecDeque};

use bitvec::vec::BitVec;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::tree::{parent_pointer::ParentPointerStore, Forest, ForestNodeStore, RootId};

use super::{
    involution::{Hedge, Involution},
    nodestorage::NodeStorage,
    subgraph::{Cycle, Inclusion, InternalSubGraph, SubGraph, SubGraphOps},
    HedgeGraph, HedgeGraphError, NodeIndex, NodeStorageOps,
};

impl<V, P: ForestNodeStore<NodeData = ()>> NodeStorage for Forest<V, P> {
    type Storage<N> = Forest<N, P>;
    type NodeData = V;
}

// pub struct HedgeTree<V, P: ForestNodeStore<NodeData = ()>, E> {
//     graph: HedgeGraph<E, V, Forest<V, P>>,
//     tree_subgraph: InternalSubGraph,
//     covers: BitVec,
// }

// impl<V, P: ForestNodeStore<NodeData = ()>, E> HedgeTree<V, P, E> {
//     pub fn ancestor_nodes(&self,) {
//         self.
//     }
// }

// #[derive(Debug, Clone)]
// pub struct TraversalTreeRef<
//     'a,
//     E,
//     V,
//     N: NodeStorage<NodeData = V>,
//     P: ForestNodeStore<NodeData = ()>,
// > {
//     graph: &'a HedgeGraph<E, V, N>,
//     simple: SimpleTraversalTree<P>,
//     tree_subgraph: InternalSubGraph,
// }

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TTRoot {
    Child,
    Root,
    None,
}

impl TTRoot {
    pub fn includes(&self) -> bool {
        match self {
            TTRoot::Child => true,
            TTRoot::Root => true,
            TTRoot::None => false,
        }
    }

    pub fn is_root(&self) -> bool {
        match self {
            TTRoot::Child => false,
            TTRoot::Root => true,
            TTRoot::None => false,
        }
    }

    pub fn is_child(&self) -> bool {
        match self {
            TTRoot::Child => true,
            TTRoot::Root => false,
            TTRoot::None => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTraversalTree<P: ForestNodeStore<NodeData = ()> = ParentPointerStore<()>> {
    forest: Forest<TTRoot, P>,
    pub tree_subgraph: BitVec,
}

impl<P: ForestNodeStore<NodeData = ()>> SimpleTraversalTree<P> {
    pub fn covers<S: SubGraph>(&self, subgraph: &S) -> BitVec {
        // println!("calculating covers..");
        let mut covers = BitVec::empty(self.tree_subgraph.len());

        // self.tree_subgraph.covers(graph)

        for i in 0..self.tree_subgraph.len() {
            if self.node_data(self.node_id(Hedge(i))).includes() && subgraph.includes(&Hedge(i)) {
                covers.set(i, true);
            }
        }

        covers
    }

    pub fn node_data(&self, node: NodeIndex) -> &TTRoot {
        &self.forest[RootId(node.0)]
    }

    pub fn internal<I: AsRef<Involution>>(&self, hedge: Hedge, _inv: I) -> bool {
        self.tree_subgraph.includes(&hedge)
    }

    pub fn tree_subgraph<I: AsRef<Involution>>(&self, inv: I) -> InternalSubGraph {
        let mut tree = BitVec::empty(self.tree_subgraph.len());
        for i in self.tree_subgraph.included_iter() {
            if self.internal(i, inv.as_ref()) {
                tree.set(i.0, true);
            }
        }
        unsafe { InternalSubGraph::new_unchecked(tree) }
    }

    pub fn get_cycle<I: AsRef<Involution>>(&self, cut: Hedge, inv: &I) -> Option<Cycle> {
        if self.internal(cut, inv) {
            return None;
        }
        let mut cycle = self.path_to_root(cut, inv);
        cycle.sym_diff_with(&self.path_to_root(inv.as_ref().inv(cut), inv));
        let mut cycle = Cycle::new_unchecked(cycle);
        cycle.loop_count = Some(1);

        Some(cycle)
    }

    pub fn node_id(&self, hedge: Hedge) -> NodeIndex {
        NodeIndex::from(self.forest.root(hedge.into()))
    }

    /// get the parent node of the current node
    pub fn node_parent<I: AsRef<Involution>>(&self, from: NodeIndex, inv: I) -> Option<NodeIndex> {
        let root = RootId::from(from);

        if self.forest[root].is_child() {
            // if the current node is in the forest
            // get the involved hedge connected to the root pointing hedge of the current node
            let involved = inv.as_ref().inv(self.forest[&root].into());
            Some(self.node_id(involved))
        } else {
            None
        }
    }

    fn path_to_root<I: AsRef<Involution>>(&self, start: Hedge, inv: I) -> BitVec {
        let mut path = BitVec::empty(self.tree_subgraph.len());

        self.ancestor_iter_hedge(start, inv.as_ref())
            .for_each(|a| path.set(a.0, true));
        path
    }

    /// among the hedges of the same node get the one pointing to the root node.
    /// if the from hedge is that pointing to the root node, go to the involved hedge.
    pub fn hedge_parent<I: AsRef<Involution>>(&self, from: Hedge, inv: I) -> Option<Hedge> {
        let root = self.forest.root(from.into()); //Get "NodeId/RootId"

        match self.forest[root] {
            TTRoot::Child => {
                let roothedge = self.forest[&root].into(); //Get "chosen" root among node hairs

                if from == roothedge {
                    //if it is the same as the input, go to the involved hedge
                    Some(inv.as_ref().inv(from))
                } else {
                    // else go to the "chosen" hedge
                    Some(roothedge)
                }
            }
            TTRoot::None => None,
            TTRoot::Root => None, // if it is attached to the root node, it has no parent
        }
    }
}

pub struct TraversalTreeAncestorHedgeIterator<'a, P: ForestNodeStore<NodeData = ()>> {
    tt: &'a SimpleTraversalTree<P>,
    inv: &'a Involution,
    current: Option<Hedge>,
}

impl<P: ForestNodeStore<NodeData = ()>> Iterator for TraversalTreeAncestorHedgeIterator<'_, P> {
    type Item = Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.current {
            self.current = self.tt.hedge_parent(c, self.inv);
            Some(c)
        } else {
            None
        }
    }
}

impl<P: ForestNodeStore<NodeData = ()>> SimpleTraversalTree<P> {
    pub fn ancestor_iter_hedge<'a>(
        &'a self,
        start: Hedge,
        inv: &'a Involution,
    ) -> impl Iterator<Item = Hedge> + 'a {
        TraversalTreeAncestorHedgeIterator {
            tt: self,
            inv,
            current: Some(start),
        }
    }

    pub fn dot<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> String {
        let mut structure = graph.just_structure();
        structure.align_superficial_to_tree(self);

        structure.base_dot()
    }

    pub fn tree_order<I: AsRef<Involution>>(
        &self,
        left: NodeIndex,
        right: NodeIndex,
        inv: I,
    ) -> Option<Ordering> {
        if left == right {
            return Some(Ordering::Equal);
        }

        let left_to_root_passes_through_right = self
            .ancestor_iter_node(left, inv.as_ref())
            .any(|a| a == right);
        if left_to_root_passes_through_right {
            return Some(Ordering::Less);
        }

        let right_to_root_passes_through_left = self
            .ancestor_iter_node(right, inv.as_ref())
            .any(|a| a == left);
        if right_to_root_passes_through_left {
            return Some(Ordering::Greater);
        }

        None
    }

    /// Iterate over all half-edges in the tree.
    ///
    /// Each iteration yields a three-tuple:
    ///
    /// - the half-edge
    /// - whether it is part of a child, or root node, or outside of the tree
    /// - the root-pointing half-edge if it is actually part of the tree
    ///
    pub fn iter_hedges(&self) -> impl Iterator<Item = (Hedge, TTRoot, Option<Hedge>)> + '_ {
        self.forest.iter_node_ids().map(|a| {
            let root = self.forest.root(a); //Get "NodeId/RootId"
            let hedge: Hedge = a.into();
            let ttype = self.forest[root];
            let root_among_hedges = if self.tree_subgraph.includes(&hedge) {
                Some(self.forest[&root].into())
            } else {
                None
            };

            (hedge, ttype, root_among_hedges)
        })
    }
}

pub struct TraversalTreeAncestorNodeIterator<'a, P: ForestNodeStore<NodeData = ()>> {
    tt: &'a SimpleTraversalTree<P>,
    inv: &'a Involution,
    current: Option<NodeIndex>,
}

impl<P: ForestNodeStore<NodeData = ()>> Iterator for TraversalTreeAncestorNodeIterator<'_, P> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.current {
            self.current = self.tt.node_parent(c, self.inv);
            Some(c)
        } else {
            None
        }
    }
}

impl<P: ForestNodeStore<NodeData = ()>> SimpleTraversalTree<P> {
    pub fn ancestor_iter_node<'a>(
        &'a self,
        from: NodeIndex,
        inv: &'a Involution,
    ) -> impl Iterator<Item = NodeIndex> + 'a {
        TraversalTreeAncestorNodeIterator {
            tt: self,
            inv,
            current: Some(from),
        }
    }
}

impl SimpleTraversalTree {
    pub fn empty<E, V, N: NodeStorageOps<NodeData = V>>(graph: &HedgeGraph<E, V, N>) -> Self {
        let forest = graph.node_store.to_forest(|_| TTRoot::None);
        SimpleTraversalTree {
            forest,
            tree_subgraph: graph.empty_subgraph(),
        }
    }

    // pub fn root(&self,Hedge) -> Hedge {
    //     self.forest.root()
    // }

    pub fn depth_first_traverse<S: SubGraph, E, V, N: NodeStorageOps<NodeData = V>>(
        graph: &HedgeGraph<E, V, N>,
        subgraph: &S,
        root_node: &NodeIndex,
        include_hedge: Option<Hedge>,
    ) -> Result<Self, HedgeGraphError> {
        let mut seen = subgraph.hairs(&graph[root_node]);

        if seen.count_ones() == 0 {
            // if the root node is not in the subgraph
            return Err(HedgeGraphError::RootNodeNotInSubgraph(*root_node));
        }

        let mut stack = seen.included_iter().collect::<Vec<_>>();
        if let Some(r) = include_hedge {
            if graph.inv(r) == r {
                return Err(HedgeGraphError::ExternalHedgeIncluded(r)); //cannot include an external hedge in the traversal
            }

            let pos = stack.iter().find_position(|a| **a == r).map(|a| a.0);

            let last = stack.len() - 1;
            if let Some(pos) = pos {
                stack.swap(pos, last);
            } else {
                return Err(HedgeGraphError::NotInNode(r, *root_node));
            }
        }

        let mut init = Self::empty(graph);
        init.forest[RootId(root_node.0)] = TTRoot::Root;

        while let Some(hedge) = stack.pop() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.involved_node_hairs(hedge) {
                let connected = graph.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    // is a new node in the traversal
                    init.tree_subgraph.set(connected.0, true);
                    init.tree_subgraph.set(hedge.0, true);
                    let node_id = init.forest.change_to_root(connected.into());
                    init.forest[node_id] = TTRoot::Child;
                } else {
                    continue;
                }

                // mark the new node as seen
                // seen.union_with(&cn.hairs);

                for i in cn.hairs.included_iter() {
                    if subgraph.includes(&i) {
                        seen.set(i.0, true);
                        if !seen.includes(&graph.inv(i)) {
                            stack.push(i);
                        }
                    }
                }
            }
        }

        // init.tree_subgraph = tree_subgraph;
        Ok(init)
    }

    pub fn breadth_first_traverse<S: SubGraph, E, V, N: NodeStorageOps<NodeData = V>>(
        graph: &HedgeGraph<E, V, N>,
        subgraph: &S,
        root_node: &NodeIndex,
        include_hedge: Option<Hedge>,
    ) -> Result<Self, HedgeGraphError> {
        let mut seen = subgraph.hairs(&graph[root_node]);

        if seen.count_ones() == 0 {
            // if the root node is not in the subgraph
            return Err(HedgeGraphError::InvalidNode(*root_node));
        }

        let mut queue = seen.included_iter().collect::<VecDeque<_>>();
        if let Some(r) = include_hedge {
            if graph.inv(r) == r {
                return Err(HedgeGraphError::InvalidHedge(r)); //cannot include an external hedge in the traversal
            }
            let pos = queue.iter().find_position(|a| **a == r).map(|a| a.0);
            if let Some(pos) = pos {
                queue.swap(pos, 0);
            } else {
                return Err(HedgeGraphError::InvalidHedge(r));
            }
        }

        let mut init = Self::empty(graph);

        init.forest[RootId(root_node.0)] = TTRoot::Root;
        while let Some(hedge) = queue.pop_front() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = graph.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    //  a new node in the traversal
                    //
                    init.tree_subgraph.set(connected.0, true);
                    init.tree_subgraph.set(hedge.0, true);
                    let node_id = init.forest.change_to_root(connected.into());
                    init.forest[node_id] = TTRoot::Child;
                } else {
                    continue;
                }
                // mark the new node as seen
                seen.union_with(&cn);

                // for all hedges in this new node, they have a parent, the initial hedge
                for i in cn.included_iter() {
                    // if they lead to a new node, they are potential branches, add them to the queue
                    if !seen.includes(&graph.inv(i)) && subgraph.includes(&i) {
                        queue.push_back(i);
                    }
                }
            }
        }
        Ok(init)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnedTraversalTree<P: ForestNodeStore<NodeData = ()>> {
    graph: HedgeGraph<(), bool, Forest<bool, P>>,
    tree_subgraph: InternalSubGraph,
    covers: BitVec,
}

// impl TraversalTree {
//     pub fn children(&self, hedge: Hedge) -> BitVec {
//         let mut children = <BitVec as SubGraph>::empty(self.inv.inv.len());

//         for (i, m) in self.parents.iter().enumerate() {
//             if let Parent::Hedge { hedge_to_root, .. } = m {
//                 if *hedge_to_root == hedge {
//                     children.set(i, true);
//                 }
//             }
//         }

//         children.set(hedge.0, false);

//         children
//     }

//     pub fn leaf_edges(&self) -> BitVec {
//         let mut leaves = <BitVec as SubGraph>::empty(self.inv.inv.len());
//         for hedge in self.covers().included_iter() {
//             let is_not_parent = !self.parent_iter().any(|(_, p)| {
//                 if let Parent::Hedge { hedge_to_root, .. } = p {
//                     *hedge_to_root == hedge
//                 } else {
//                     false
//                 }
//             });
//             if is_not_parent {
//                 leaves.set(hedge.0, true);
//             }
//         }
//         leaves
//     }

//     pub fn leaf_nodes<V, N: NodeStorageOps<NodeData = V>, E>(
//         &self,
//         graph: &HedgeGraph<E, V, N>,
//     ) -> Vec<NodeIndex> {
//         let mut leaves = IndexSet::new();

//         for hedge in self.covers().included_iter() {
//             if let Parent::Hedge { hedge_to_root, .. } = self.parent(hedge) {
//                 if *hedge_to_root == hedge {
//                     let mut sect = self
//                         .tree
//                         .filter
//                         .intersection(&graph.node_hairs(hedge).hairs);

//                     sect.set(hedge.0, false);

//                     if sect.count_ones() == 0 {
//                         leaves.insert(graph.node_id(hedge));
//                     }
//                 }
//             }
//         }

//         leaves.into_iter().collect()
//     }

//     pub fn child_nodes<V, N: NodeStorageOps<NodeData = V>, E>(
//         &self,
//         parent: NodeIndex,
//         graph: &HedgeGraph<E, V, N>,
//     ) -> Vec<NodeIndex> {
//         let mut children = IndexSet::new();

//         for h in graph.hairs_from_id(parent).hairs.included_iter() {
//             if let Parent::Hedge { hedge_to_root, .. } = self.parent(h) {
//                 if *hedge_to_root != h {
//                     if let Some(c) = graph.involved_node_id(h) {
//                         children.insert(c);
//                     }
//                 }
//             }
//         }

//         children.into_iter().collect()
//     }
// }
#[cfg(test)]
pub mod tests {
    use crate::{dot, half_edge::involution::Orientation};

    use super::*;

    #[test]
    fn double_pentagon_tree() {
        let mut graph: HedgeGraph<
            crate::dot_parser::DotEdgeData,
            crate::dot_parser::DotVertexData,
        > = dot!(
            digraph {
        node [shape=circle,height=0.1,label=""];  overlap="scale"; layout="neato";
        00 -> 07[ dir=none,label="a"];
        00 -> 12[ dir=forward,label="d"];
        01 -> 00[ dir=forward,label="d"];
        05 -> 02[ dir=forward,label="d"];
        02 -> 01[ dir=forward,label="d"];
        09 -> 04[ dir=forward,label="d"];
        02 -> 06[ dir=none,label="a"];
        01 -> 03[ dir=none,label="a"];
        03 -> 13[ dir=forward,label="d"];
        04 -> 03[ dir=forward,label="d"];
        04 -> 05[ dir=none,label="g"];
        06 -> 07[ dir=forward,label="e-"];
        07 -> 11[ dir=forward,label="e-"];
        08 -> 06[ dir=forward,label="e-"];
        10 -> 05[ dir=forward,label="d"];
        }
        )
        .unwrap();

        // println!("{}", graph.dot_display(&graph.full_filter()));

        let tree = SimpleTraversalTree::depth_first_traverse(
            &graph,
            &graph.full_filter(),
            &NodeIndex(0),
            None,
        )
        .unwrap();
        // println!("{}", tree.dot(&graph));

        graph.align_underlying_to_tree(&tree);
        graph.align_superficial_to_tree(&tree);
        assert!(graph
            .iter_all_edges()
            .all(|(_, _, d)| !matches!(d.orientation, Orientation::Reversed)));
        // println!("{}", tree.dot(&graph))
        // Test implementation
    }
}
