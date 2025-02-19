use std::collections::VecDeque;

use bitvec::vec::BitVec;
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

use crate::tree::{
    child_pointer::ParentChildStore, child_vec::ChildVecStore, parent_pointer::ParentPointerStore,
    Forest, ForestNodeStore,
};

use super::{
    involution::{Hedge, Involution},
    nodestorage::NodeStorage,
    subgraph::{Cycle, HedgeNode, Inclusion, InternalSubGraph, SubGraph, SubGraphOps},
    HedgeGraph, NodeIndex, NodeStorageOps,
};

impl<V, P: ForestNodeStore<NodeData = ()>> NodeStorage for Forest<V, P> {
    type Storage<N> = Forest<N, P>;
    type NodeData = V;
}

pub struct HedgeTree<V, P: ForestNodeStore<NodeData = ()>, E> {
    graph: HedgeGraph<E, V, Forest<V, P>>,
    tree_subgraph: InternalSubGraph,
    covers: BitVec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalTree {
    //a parent pointer structure
    pub traversal: Vec<Hedge>,
    pub parents: Vec<Parent>,
    pub(crate) inv: Involution<()>, // essentially just a vec of Parent that is the same length as the vec of hedges
    pub tree: InternalSubGraph,
    pub covers: BitVec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Parent {
    Unset,
    Root,
    Hedge {
        hedge_to_root: Hedge,
        traversal_order: usize,
    },
}

pub enum Child {
    None,
    Hedge(Hedge),
}

impl TraversalTree {
    pub fn parent_iter(&self) -> impl Iterator<Item = (Hedge, &Parent)> {
        self.parents.iter().enumerate().map(|(h, p)| (Hedge(h), p))
    }

    pub fn children(&self, hedge: Hedge) -> BitVec {
        let mut children = <BitVec as SubGraph>::empty(self.inv.inv.len());

        for (i, m) in self.parents.iter().enumerate() {
            if let Parent::Hedge { hedge_to_root, .. } = m {
                if *hedge_to_root == hedge {
                    children.set(i, true);
                }
            }
        }

        children.set(hedge.0, false);

        children
    }

    pub fn covers(&self) -> BitVec {
        let mut covers = <BitVec as SubGraph>::empty(self.inv.inv.len());
        for (i, m) in self.parents.iter().enumerate() {
            match m {
                Parent::Unset => {}
                _ => {
                    covers.set(i, true);
                }
            }
        }
        covers
    }

    pub fn connected_parent(&self, hedge: Hedge) -> &Parent {
        &self.parents[self.inv.inv(hedge).0]
    }

    pub fn leaf_edges(&self) -> BitVec {
        let mut leaves = <BitVec as SubGraph>::empty(self.inv.inv.len());
        for hedge in self.covers().included_iter() {
            let is_not_parent = !self.parent_iter().any(|(_, p)| {
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

    /// Parent of a hedge, if is a `Parent::Hedge` then the hedge to root is in the same node,
    /// but points towards the root
    pub fn parent(&self, hedge: Hedge) -> &Parent {
        &self.parents[hedge.0]
    }

    pub fn leaf_nodes<V, N: NodeStorageOps<NodeData = V>, E>(
        &self,
        graph: &HedgeGraph<E, V, N>,
    ) -> Vec<NodeIndex> {
        let mut leaves = IndexSet::new();

        for hedge in self.covers().included_iter() {
            if let Parent::Hedge { hedge_to_root, .. } = self.parent(hedge) {
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

    pub fn parent_node<V, N: NodeStorageOps<NodeData = V>, E>(
        &self,
        child: NodeIndex,
        graph: &HedgeGraph<E, V, N>,
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

    pub fn child_nodes<V, N: NodeStorageOps<NodeData = V>, E>(
        &self,
        parent: NodeIndex,
        graph: &HedgeGraph<E, V, N>,
    ) -> Vec<NodeIndex> {
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

        while let Parent::Hedge { hedge_to_root, .. } = self.parent(current) {
            path.set(hedge_to_root.0, true);
            current = self.inv.inv(*hedge_to_root);
            path.set(current.0, true);
        }
        path
    }

    pub fn cycle(&self, cut: Hedge) -> Option<Cycle> {
        match self.parent(cut) {
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
        match self.parent(cut_pair) {
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

    pub fn bfs<V, N: NodeStorageOps<NodeData = V>, E, S: SubGraph>(
        graph: &HedgeGraph<E, V, N>,
        subgraph: &S,
        root_node: &HedgeNode,
        // target: Option<&HedgeNode>,
    ) -> Self {
        let mut queue = VecDeque::new();
        let mut seen = subgraph.hairs(root_node);
        let mut parents = vec![Parent::Unset; graph.n_hedges()];

        let mut traversal: Vec<Hedge> = Vec::new();

        // add all hedges from root node that are not self loops
        // to the queue
        // They are all potential branches
        for i in seen.included_iter() {
            parents[i.0] = Parent::Root;
            if !seen.includes(&graph.inv(i)) {
                // if not self loop
                queue.push_back(i)
            }
        }
        while let Some(hedge) = queue.pop_front() {
            // if the hedge is not external get the neighbors of the paired hedge
            if let Some(cn) = graph.connected_neighbors(subgraph, hedge) {
                let connected = graph.inv(hedge);

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
                    if let Parent::Unset = parents[i.0] {
                        parents[i.0] = Parent::Hedge {
                            hedge_to_root: connected,
                            traversal_order: traversal.len(),
                        };
                    }
                    // if they lead to a new node, they are potential branches, add them to the queue
                    if !seen.includes(&graph.inv(i)) && subgraph.includes(&i) {
                        queue.push_back(i);
                    }
                }
            }
        }

        TraversalTree::new(graph, traversal, seen, parents)
    }

    pub fn new<V, N: NodeStorageOps<NodeData = V>, E>(
        graph: &HedgeGraph<E, V, N>,
        traversal: Vec<Hedge>,
        covers: BitVec,
        parents: Vec<Parent>,
    ) -> Self {
        let mut tree = graph.empty_subgraph::<BitVec>();

        let involution = graph.edge_store.involution.map_data_ref(&|_| ());

        for (i, j) in traversal.iter().map(|x| (*x, involution.inv(*x))) {
            tree.set(i.0, true);
            tree.set(j.0, true);
        }

        TraversalTree {
            traversal,
            covers,
            inv: involution,
            parents,
            tree: InternalSubGraph::cleaned_filter_optimist(tree, graph),
        }
    }

    pub fn dfs<V, N: NodeStorageOps<NodeData = V>, E, S: SubGraph>(
        graph: &HedgeGraph<E, V, N>,
        subgraph: &S,
        root_node: &HedgeNode,
        include_hegde: Option<Hedge>,
        // target: Option<&HedgeNode>,
    ) -> Self {
        let mut stack = Vec::new();
        let mut seen = subgraph.hairs(root_node);

        let mut traversal: Vec<Hedge> = Vec::new();
        let mut parents = vec![Parent::Unset; graph.n_hedges()];

        let mut included_hedge_is_possible = false;

        // add all hedges from root node that are not self loops
        // to the stack
        // They are all potential branches

        for i in seen.included_iter() {
            parents[i.0] = Parent::Root;
            if !seen.includes(&graph.inv(i)) {
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
                let connected = graph.inv(hedge);

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
                    if let Parent::Unset = parents[i.0] {
                        parents[i.0] = Parent::Hedge {
                            hedge_to_root: connected,
                            traversal_order: traversal.len(),
                        };
                    }

                    if !seen.includes(&graph.inv(i)) && subgraph.includes(&i) {
                        stack.push(i);
                    }
                }
            }
        }

        TraversalTree::new(graph, traversal, seen, parents)
    }
}
