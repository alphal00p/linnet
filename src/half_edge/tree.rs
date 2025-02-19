use std::{collections::VecDeque, ops::Index};

use bitvec::vec::BitVec;
use indexmap::IndexSet;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::tree::{
    child_pointer::ParentChildStore,
    child_vec::ChildVecStore,
    parent_pointer::{ParentId, ParentPointerStore},
    Forest, ForestNodeStore, RootId, TreeNodeId,
};

use super::{
    involution::{Hedge, Involution},
    nodestorage::NodeStorage,
    subgraph::{Cycle, HedgeNode, Inclusion, InternalSubGraph, SubGraph, SubGraphOps},
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

#[derive(Debug, Clone)]
pub struct TraversalTreeRef<
    'a,
    E,
    V,
    N: NodeStorage<NodeData = V>,
    P: ForestNodeStore<NodeData = ()>,
> {
    graph: &'a HedgeGraph<E, V, N>,
    simple: SimpleTraversalTree<P>,
    tree_subgraph: InternalSubGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleTraversalTree<P: ForestNodeStore<NodeData = ()> = ParentPointerStore<()>> {
    forest: Forest<bool, P>,
    pub covers: BitVec,
}

impl<P: ForestNodeStore<NodeData = ()>> SimpleTraversalTree<P> {
    pub fn internal<I: AsRef<Involution>>(&self, hedge: Hedge, inv: I) -> bool {
        let treeid = TreeNodeId::from(hedge);
        let invtreeid = TreeNodeId::from(inv.as_ref().inv(hedge));
        self.forest[self.forest.root(treeid)] && self.forest[self.forest.root(invtreeid)] && {
            self.forest[&treeid].is_root() || self.forest[&invtreeid].is_root()
        }
    }

    pub fn tree_subgraph<I: AsRef<Involution>>(&self, inv: I) -> InternalSubGraph {
        let mut tree = BitVec::empty(self.covers.len());
        for i in self.covers.included_iter() {
            if self.internal(i, inv.as_ref()) {
                tree.set(i.0, true);
            }
        }
        unsafe { InternalSubGraph::new_unchecked(tree) }
    }

    pub fn cycle<I: AsRef<Involution>>(&self, cut: Hedge, inv: &I) -> Option<Cycle> {
        if !self.internal(cut, inv) {
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

    pub fn node_parent<I: AsRef<Involution>>(&self, from: NodeIndex, inv: I) -> Option<NodeIndex> {
        let root = RootId::from(from);

        if self.forest[root] {
            let involved = inv.as_ref().inv(self.forest[&root].into());
            Some(self.node_id(involved))
        } else {
            None
        }
    }

    fn path_to_root<I: AsRef<Involution>>(&self, start: Hedge, inv: I) -> BitVec {
        let mut path = BitVec::empty(self.covers.len());

        self.ancestor_iter_hedge(start, inv.as_ref())
            .for_each(|a| path.set(a.0, true));
        path
    }

    pub fn hedge_parent<I: AsRef<Involution>>(&self, from: Hedge, inv: I) -> Option<Hedge> {
        let root = self.forest.root(from.into());

        if self.forest[root] {
            let roothedge = self.forest[&root].into();

            if from == roothedge {
                Some(inv.as_ref().inv(from))
            } else {
                Some(roothedge)
            }
        } else {
            None
        }
    }
}

pub struct TraversalTreeAncestorHedgeIterator<'a, P: ForestNodeStore<NodeData = ()>> {
    tt: &'a SimpleTraversalTree<P>,
    inv: &'a Involution,
    current: Option<Hedge>,
}

impl<'a, P: ForestNodeStore<NodeData = ()>> Iterator for TraversalTreeAncestorHedgeIterator<'a, P> {
    type Item = Hedge;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.current {
            self.current = self.tt.hedge_parent(c, &self.inv);
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
}

pub struct TraversalTreeAncestorNodeIterator<'a, P: ForestNodeStore<NodeData = ()>> {
    tt: &'a SimpleTraversalTree<P>,
    inv: &'a Involution,
    current: Option<NodeIndex>,
}

impl<'a, P: ForestNodeStore<NodeData = ()>> Iterator for TraversalTreeAncestorNodeIterator<'a, P> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.current {
            self.current = self.tt.node_parent(c, &self.inv);
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
        let nodes: Vec<_> = graph
            .iter_nodes()
            .map(|(n, _)| (false, n.hairs.clone()))
            .collect();

        let forest = Forest::from_bitvec_partition(nodes).unwrap();
        SimpleTraversalTree {
            forest,
            covers: graph.empty_subgraph(),
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
            return Err(HedgeGraphError::InvalidNode(*root_node));
        }

        let mut stack = seen.included_iter().collect::<Vec<_>>();
        if let Some(r) = include_hedge {
            if graph.inv(r) == r {
                return Err(HedgeGraphError::InvalidHedge(r)); //cannot include an external hedge in the traversal
            }

            let pos = stack.iter().find_position(|a| **a == r).map(|a| a.0);

            let last = stack.len() - 1;
            if let Some(pos) = pos {
                stack.swap(pos, last);
            } else {
                return Err(HedgeGraphError::InvalidHedge(r));
            }
        }

        let mut init = Self::empty(graph);

        while let Some(hedge) = stack.pop() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = graph.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    // is a new node in the traversal
                    let node_id = init.forest.change_to_root(connected.into());
                    init.forest[node_id] = true;
                } else {
                    continue;
                }

                // mark the new node as seen
                seen.union_with(&cn);

                for i in cn.included_iter() {
                    if !seen.includes(&graph.inv(i)) && subgraph.includes(&i) {
                        stack.push(i);
                    }
                }
            }
        }

        init.covers = seen;
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

        while let Some(hedge) = queue.pop_front() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = graph.inv(hedge);

                if !seen.includes(&connected) && subgraph.includes(&connected) {
                    // if this new hedge hasn't been seen before, it means the node it belongs to
                    //  a new node in the traversal
                    let node_id = init.forest.change_to_root(connected.into());
                    init.forest[node_id] = true;
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
        init.covers = seen;
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
