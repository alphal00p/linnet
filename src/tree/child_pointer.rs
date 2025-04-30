//! Implements a tree structure using a first-child, next-sibling representation.
//! Each node stores its parent, data, a pointer to its first child, and pointers
//! to its left and right siblings using a *cyclic* doubly linked list for siblings.

use std::{collections::VecDeque, fmt::Write};

use bitvec::vec::BitVec;

use crate::half_edge::{
    involution::Hedge,
    subgraph::{Inclusion, ModifySubgraph, SubGraph},
};

use super::{
    child_vec::ChildVecStore,
    parent_pointer::{PPNode, ParentId, ParentPointerStore},
    ForestError, ForestNodeStore, ForestNodeStoreAncestors, ForestNodeStoreBfs,
    ForestNodeStoreDown, ForestNodeStorePreorder, RootId, TreeNodeId,
};

/// Represents a node within a `ParentChildStore`.
///
/// Contains the parent pointer and data (`PPNode`), an optional pointer to the
/// *first* child, and pointers to the left and right siblings. The sibling
/// pointers form a cyclic doubly linked list among all children of the same parent.
/// If a node is the only child, its `neighbor_left` and `neighbor_right` point to itself.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct PCNode<V> {
    /// Parent pointer and node data.
    pub(crate) parent_pointer: PPNode<V>,
    /// Pointer to the first child of this node, if any.
    pub(crate) child: Option<TreeNodeId>, // To get the other children, follow the sibling links of this child
    /// Pointer to the previous (left) sibling in the cyclic list. Points to self if only child.
    pub(crate) neighbor_left: TreeNodeId, // Will form a cyclic linked list, if equal to right, then it is the only child
    /// Pointer to the next (right) sibling in the cyclic list. Points to self if only child.
    pub(crate) neighbor_right: TreeNodeId, // previous sibling, if any. Will form a cyclic linked list, if equal to left, then it is the only child
}

impl<V> PCNode<V> {
    pub fn debug_display(
        &self,
        writer: &mut impl std::fmt::Write,
        draw_data: &mut impl FnMut(Option<&V>) -> Option<String>,
    ) {
        write!(writer, "{}<- ->{}", self.neighbor_left, self.neighbor_right).unwrap();
        if let Some(child) = self.child {
            write!(writer, "child: {}", child).unwrap();
        }
        self.parent_pointer.debug_display(writer, draw_data);
    }

    pub fn shift(&mut self, root: RootId, nodes: TreeNodeId) {
        self.neighbor_left.0 += nodes.0;
        self.neighbor_right.0 += nodes.0;
        self.parent_pointer.shift(root, nodes);
        self.child.as_mut().map(|a| a.0 += nodes.0).unwrap();
    }

    pub fn forget<U>(&self) -> PCNode<U> {
        PCNode {
            parent_pointer: self.parent_pointer.forget(),
            child: self.child,
            neighbor_left: self.neighbor_left,
            neighbor_right: self.neighbor_right,
        }
    }

    pub fn map<F, U>(self, transform: F) -> PCNode<U>
    where
        F: FnMut(V) -> U,
    {
        PCNode {
            parent_pointer: self.parent_pointer.map(transform),
            child: self.child,
            neighbor_left: self.neighbor_left,
            neighbor_right: self.neighbor_right,
        }
    }

    pub fn map_ref<F, U>(&self, transform: F) -> PCNode<U>
    where
        F: FnMut(&V) -> U,
    {
        PCNode {
            parent_pointer: self.parent_pointer.map_ref(transform),
            child: self.child,
            neighbor_right: self.neighbor_right,
            neighbor_left: self.neighbor_left,
        }
    }
}

/// A forest data structure using a first-child, next-sibling representation with cyclic sibling links.
///
/// Each node stores its parent, data, a pointer to its first child, and pointers to its left and
/// right siblings. Siblings under a parent form a cyclic doubly linked list.
///
/// This representation allows for efficient iteration over children and siblings.
/// It implements `ForestNodeStore` and `ForestNodeStoreDown`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
pub struct ParentChildStore<V> {
    /// The flat list of nodes. The index in the vector corresponds to the `TreeNodeId`.
    pub(crate) nodes: Vec<PCNode<V>>,
}

impl<V> std::ops::Index<&TreeNodeId> for ParentChildStore<V> {
    type Output = ParentId;
    fn index(&self, index: &TreeNodeId) -> &Self::Output {
        &self.nodes[index.0].parent_pointer.parent
    }
}

impl<V> std::ops::Index<TreeNodeId> for ParentChildStore<V> {
    type Output = V;
    fn index(&self, index: TreeNodeId) -> &Self::Output {
        self.nodes[index.0].parent_pointer.data.as_ref().unwrap()
    }
}

impl<V> std::ops::IndexMut<TreeNodeId> for ParentChildStore<V> {
    fn index_mut(&mut self, index: TreeNodeId) -> &mut Self::Output {
        self.nodes[index.0].parent_pointer.data.as_mut().unwrap()
    }
}

impl<V> FromIterator<PCNode<V>> for ParentChildStore<V> {
    fn from_iter<I: IntoIterator<Item = PCNode<V>>>(iter: I) -> Self {
        ParentChildStore {
            nodes: iter.into_iter().collect(),
        }
    }
}

/// An iterator over the siblings of a node (including the node itself).
/// Uses the cyclic `neighbor_right` pointers.
pub enum NeighborIter<'a, V> {
    /// Case where the node has no children/siblings to iterate over.
    None,
    Some {
        initial: TreeNodeId,
        current: Option<TreeNodeId>,
        first: bool,
        store: &'a ParentChildStore<V>,
    },
}

impl<V> Iterator for NeighborIter<'_, V> {
    type Item = TreeNodeId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NeighborIter::None => None,
            NeighborIter::Some {
                initial,
                current,
                first,
                store,
            } => {
                let next = *current;
                if let Some(c) = next {
                    if c == *initial && !*first {
                        return None;
                    }
                    *first = false;
                    *current = store.right_neighbor_cyclic(c);
                }
                next
            }
        }
    }
}

impl<V> ParentChildStore<V> {
    /// Returns the left sibling of `node_id`.
    /// Returns `None` if `node_id` is the only child of its parent (based on cyclic link).
    pub fn left_neighbor_cyclic(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        if self.nodes[node_id.0].neighbor_left == node_id {
            None
        } else {
            Some(self.nodes[node_id.0].neighbor_left)
        }
    }

    /// Returns the left sibling of `node_id`, stopping at the "first" child.
    ///
    /// This differs from `left_neighbor_cyclic` by returning `None` if moving left
    /// would wrap around from the *parent's first recorded child* back to the last child.
    /// Useful if a non-cyclic view of siblings is needed, although the internal
    /// representation *is* cyclic.
    /// Returns `None` if the node is a root or the only child.
    pub fn left_neighbor_ending(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        let left = self.nodes[node_id.0].neighbor_left;
        if left == node_id {
            None
        } else {
            let first_neighbor = match self[&node_id] {
                ParentId::Node(p) => p,
                ParentId::Root(_) => return None,
                ParentId::PointingRoot(p) => p,
            };
            if left == first_neighbor {
                None
            } else {
                Some(left)
            }
        }
    }

    /// Returns the first child of `node_id`, if one exists.
    pub fn first_child(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        self.nodes[node_id.0].child
    }

    /// Returns the right sibling of `node_id`.
    /// Returns `None` if `node_id` is the only child of its parent (based on cyclic link).
    pub fn right_neighbor_cyclic(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        if self.nodes[node_id.0].neighbor_right == node_id {
            None
        } else {
            Some(self.nodes[node_id.0].neighbor_right)
        }
    }

    /// Returns the right sibling of `node_id`, stopping before wrapping around the cycle.
    ///
    /// This differs from `right_neighbor_cyclic` by returning `None` if moving right
    /// would wrap around from the *last* child back to the *parent's first recorded child*.
    /// Useful if a non-cyclic view of siblings is needed.
    /// Returns `None` if the node is a root or the only child.
    pub fn right_neighbor_ending(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        let right = self.nodes[node_id.0].neighbor_right;
        if right == node_id {
            None
        } else {
            let first_neighbor = match self[&node_id] {
                ParentId::Node(p) => p,
                ParentId::Root(_) => return None,
                ParentId::PointingRoot(p) => p,
            };
            if right == first_neighbor {
                None
            } else {
                Some(right)
            }
        }
    }
}

impl<V> ForestNodeStoreDown for ParentChildStore<V> {
    fn iter_leaves(&self) -> impl Iterator<Item = TreeNodeId> {
        self.nodes.iter().enumerate().filter_map(|(i, a)| {
            if a.child.is_none() {
                Some(TreeNodeId(i))
            } else {
                None
            }
        })
    }

    fn iter_children(&self, node_id: TreeNodeId) -> impl Iterator<Item = TreeNodeId> {
        if let Some(c) = self.first_child(node_id) {
            NeighborIter::Some {
                initial: c,
                current: Some(c),
                store: self,
                first: true,
            }
        } else {
            NeighborIter::None
        }
    }
}

impl<V> ForestNodeStore for ParentChildStore<V> {
    type NodeData = V;
    type Store<T> = ParentChildStore<T>;

    fn validate(&self) -> Result<Vec<(RootId, TreeNodeId)>, super::ForestError> {
        let mut roots = vec![];
        let mut seen = BitVec::empty(self.n_nodes());

        while let Some(next) = seen.iter_zeros().next() {
            let current = TreeNodeId(next);
            seen.add(Hedge(next));
            let mut children: BitVec = BitVec::empty(self.n_nodes());
            for p in self.iter_ancestors(current) {
                let h = Hedge::from(p);
                seen.add(h);
                if children.includes(&h) {
                    return Err(ForestError::CyclicPP);
                }
                children.add(h);
            }
            for c in self.iter_children(current) {
                if let ParentId::Node(n) = self[&c] {
                    if n != current {
                        return Err(ForestError::WrongPP(current));
                    }
                }
            }
            let root_id = self.root_node(current);
            let root = self.root(root_id);

            roots.push((root, root_id));
        }

        Ok(roots)
    }

    fn debug_draw(
        &self,
        mut node_display: impl FnMut(Option<&Self::NodeData>) -> Option<String>,
    ) -> String {
        // fn debug_draw(&self, node_display: impl FnMut()) -> String {
        let mut output = String::new();

        // Recursive helper function - structure remains the same, handles sub-trees
        fn draw_subtree_recursive<W: Write, V>(
            f: &mut W,                                                  // Output writer
            nodes: &ParentChildStore<V>,                                // Node store access
            node_id: TreeNodeId,                                        // Current node to draw
            prefix: &str,        // Current line prefix (e.g., "  │   ")
            is_last_child: bool, // Is this the last child of its parent?
            format_node: &mut impl FnMut(Option<&V>) -> Option<String>, // Mutable reference to node formatter closure
            seen: &mut BitVec,
        ) -> Result<(), std::fmt::Error> {
            // Determine the connector shape based on whether it's the last child
            let connector = if is_last_child {
                "└── "
            } else {
                "├── "
            };

            write!(f, "{}{}{}:", prefix, connector, node_id)?;
            nodes.nodes[node_id.0].debug_display(f, format_node);

            seen.add(Hedge::from(node_id));
            // Prepare the prefix for the children of this node
            let child_prefix = format!("{}{}", prefix, if is_last_child { "    " } else { "│   " });

            let children: Vec<_> = nodes.iter_children(node_id).collect();
            let num_children = children.len();
            for (i, child_id) in children.into_iter().enumerate() {
                seen.add(Hedge::from(child_id));
                // Recurse: pass the new prefix and whether this child is the last one
                draw_subtree_recursive(
                    f,
                    nodes,
                    child_id,
                    &child_prefix,
                    i == num_children - 1,
                    format_node,
                    seen,
                )?;
            }
            Ok(())
        } // End of helper fn definition

        let mut seen = BitVec::empty(self.n_nodes());

        while let Some(next) = seen.iter_zeros().next() {
            let root_id = self.root_node(TreeNodeId(next));
            println!("{}", root_id);
            seen.add(Hedge(next));

            seen.add(Hedge::from(root_id));

            let node_line_prefix = "  ";

            write!(output, "{}{}:", node_line_prefix, root_id).unwrap();
            self.nodes[root_id.0].debug_display(&mut output, &mut node_display);

            let children: Vec<_> = self
                .iter_children(root_id)
                .inspect(|a| {
                    seen.add(Hedge::from(*a));
                })
                .collect();
            let num_children = children.len();

            for (i, child_id) in children.into_iter().enumerate() {
                // Pass the prefix that aligns the connectors (└──, ├──)
                // under the root node line.
                seen.add(Hedge::from(child_id));
                let _ = draw_subtree_recursive(
                    &mut output,
                    self,
                    child_id,
                    node_line_prefix, // Prefix aligns the connector itself
                    i == num_children - 1,
                    &mut node_display,
                    &mut seen,
                );
            }

            let _ = writeln!(output);
        }

        output
    }
    fn set_root(&mut self, a: TreeNodeId, root: RootId) {
        let rootx = self.root_node(a);
        self.nodes[rootx.0].parent_pointer.parent = ParentId::Root(root);
    }

    fn forgetful_map<U>(&self) -> Self::Store<U> {
        ParentChildStore {
            nodes: self.nodes.iter().map(PCNode::forget).collect(),
        }
    }

    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn split_off(&mut self, at: TreeNodeId) -> Self {
        for mut current in (0..self.nodes.len()).map(TreeNodeId) {
            while let ParentId::Node(p) = self[&current] {
                let mut newp = p;
                while (newp <= at) ^ (current > at) {
                    // the pointer crosses the splitting boundary, we need to find a new pointer.

                    match self[&newp] {
                        ParentId::Node(next) => {
                            newp = next;
                        }
                        ParentId::Root(_) => {
                            // split_root = true;
                            break;
                        }
                        ParentId::PointingRoot(next) => {
                            newp = next;
                        }
                    }
                }

                if let ParentId::Root(r) = self[&newp] {
                    // set current to now be the root, and original root to point here.

                    self.nodes[newp.0].parent_pointer.parent = ParentId::PointingRoot(current);
                    self.nodes[current.0].parent_pointer.parent = ParentId::Root(r);
                } else {
                    self.reparent(newp, current);
                }

                current = p
            }
        }

        for n in (0..self.nodes.len()).map(TreeNodeId) {
            if self.nodes[n.0].neighbor_left > at {
                self.nodes[n.0].neighbor_left.0 -= at.0;
            }

            if self.nodes[n.0].neighbor_right > at {
                self.nodes[n.0].neighbor_right.0 -= at.0;
            }

            if let Some(c) = &mut self.nodes[n.0].child {
                if *c > at {
                    c.0 -= at.0;
                }
            }

            if let ParentId::Node(n) = &mut self.nodes[n.0].parent_pointer.parent {
                if *n > at {
                    n.0 -= at.0;
                }
            }

            if let ParentId::PointingRoot(rp) = self[&n] {
                let root = self[&rp];
                self.nodes[n.0].parent_pointer.parent = root;
            }
        }
        Self {
            nodes: self.nodes.split_off(at.0),
        }
    }

    fn extend(&mut self, other: Self, shift_roots_by: RootId) {
        let nodeshift = TreeNodeId(self.nodes.len());
        self.nodes.extend(other.nodes.into_iter().map(|mut r| {
            r.shift(shift_roots_by, nodeshift);
            r
        }));
    }

    fn set_node_data(&mut self, data: Self::NodeData, node_id: TreeNodeId) -> Option<V> {
        let old = self.nodes[node_id.0].parent_pointer.data.take();
        self.nodes[node_id.0].parent_pointer.data = Some(data);
        old
    }

    fn reparent(&mut self, parent: TreeNodeId, child: TreeNodeId) {
        let parent_node = &mut self.nodes[parent.0];
        if let Some(first_child_id) = parent_node.child {
            // Parent already has children. Insert new child *after* the current last child.
            let last_child_id = self.nodes[first_child_id.0].neighbor_left; // Get current last child (left of first)

            self.nodes[last_child_id.0].neighbor_right = child;
            self.nodes[first_child_id.0].neighbor_left = child;
            self.nodes[child.0].neighbor_left = last_child_id;
            self.nodes[child.0].neighbor_right = first_child_id;
        } else {
            parent_node.child = Some(child);

            self.nodes[child.0].neighbor_left = child;
            self.nodes[child.0].neighbor_right = child;
        }

        self.nodes[child.0].parent_pointer.parent = ParentId::Node(parent);
    }

    fn swap(&mut self, a: TreeNodeId, b: TreeNodeId) {
        if a != b {
            if let ParentId::Node(pp) = self.nodes[a.0].parent_pointer.parent {
                let mut children = self.nodes[pp.0].child;
                let mut init = None;
                while let Some(child) = children {
                    if init.is_none() {
                        init = children
                    } else if init == children {
                        break;
                    }
                    let childnode = &mut self.nodes[child.0];

                    children = Some(childnode.neighbor_right);
                    if childnode.neighbor_right == a {
                        childnode.neighbor_right = b;
                    }

                    if childnode.neighbor_left == a {
                        childnode.neighbor_left = b;
                    }
                }
            }

            if let ParentId::Node(pp) = self.nodes[b.0].parent_pointer.parent {
                let mut children = self.nodes[pp.0].child;
                let mut init = None;
                while let Some(child) = children {
                    if init.is_none() {
                        init = children
                    } else if init == children {
                        break;
                    }
                    let childnode = &mut self.nodes[child.0];

                    children = Some(childnode.neighbor_right);
                    if childnode.neighbor_right == a {
                        childnode.neighbor_right = b;
                    }

                    if childnode.neighbor_left == a {
                        childnode.neighbor_left = b;
                    }
                }
            }

            let mut children = self.nodes[a.0].child;
            let mut init = None;
            while let Some(child) = children {
                if init.is_none() {
                    init = children
                } else if init == children {
                    break;
                }
                children = Some(self.nodes[child.0].neighbor_right);
                if let ParentId::Node(pp) = &mut self.nodes[child.0].parent_pointer.parent {
                    if *pp == a {
                        *pp = b;
                    }
                }
            }
            let mut children = self.nodes[b.0].child;
            let mut init = None;
            while let Some(child) = children {
                if init.is_none() {
                    init = children
                } else if init == children {
                    break;
                }
                children = Some(self.nodes[child.0].neighbor_right);
                if let ParentId::Node(pp) = &mut self.nodes[child.0].parent_pointer.parent {
                    if *pp == b {
                        *pp = a;
                    }
                }
            }

            self.nodes.swap(a.0, b.0);
        }
    }

    fn to_store(self) -> Self::Store<Self::NodeData> {
        self
    }

    fn from_store(store: Self::Store<Self::NodeData>) -> Self {
        store
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, Option<&Self::NodeData>)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (TreeNodeId(i), n.parent_pointer.data.as_ref()))
    }

    fn add_root(&mut self, data: Self::NodeData, root_id: super::RootId) -> TreeNodeId {
        let node_id = TreeNodeId(self.nodes.len());
        self.nodes.push(PCNode {
            parent_pointer: PPNode::root(data, root_id),
            child: None,
            neighbor_left: node_id,
            neighbor_right: node_id,
        });
        node_id
    }

    fn add_dataless_child(&mut self, parent: TreeNodeId) -> TreeNodeId {
        let child = TreeNodeId(self.nodes.len());
        debug_assert!(parent.0 < self.nodes.len(), "Parent ID out of bounds");

        self.nodes.push(PCNode {
            parent_pointer: PPNode::dataless_child(parent),
            child: None,
            neighbor_left: child,
            neighbor_right: child,
        });

        self.reparent(parent, child);

        child
    }

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId {
        let child = TreeNodeId(self.nodes.len());
        debug_assert!(parent.0 < self.nodes.len(), "Parent ID out of bounds");

        self.nodes.push(PCNode {
            parent_pointer: PPNode::child(data, parent),
            child: None, // New node initially has no children
            neighbor_left: child,
            neighbor_right: child,
        });
        self.reparent(parent, child);

        child
    }

    fn map<F, U>(self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(Self::NodeData) -> U,
    {
        self.nodes
            .into_iter()
            .map(|node| node.map(&mut transform))
            .collect()
    }

    fn map_ref<F, U>(&self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(&Self::NodeData) -> U,
    {
        self.nodes
            .iter()
            .map(|node| node.map_ref(&mut transform))
            .collect()
    }
}

// Preorder Iterator specific to ParentChildStore internal structure
mod pc_preorder {
    use super::*;
    pub struct PreorderIter<'a, V> {
        store: &'a ParentChildStore<V>,
        current: Option<TreeNodeId>,
        // Could potentially add a stack or use the recursive logic helper
    }

    #[allow(clippy::non_canonical_clone_impl)]
    impl<V> Clone for PreorderIter<'_, V> {
        fn clone(&self) -> Self {
            Self {
                store: self.store,
                current: self.current,
            }
        }
    }

    impl<V> Copy for PreorderIter<'_, V> {}

    impl<'a, V> PreorderIter<'a, V> {
        pub fn new(store: &'a ParentChildStore<V>, start: TreeNodeId) -> Self {
            PreorderIter {
                store,
                current: Some(start),
            }
        }

        // Helper based on the original logic
        fn next_node_logic(&self, node: TreeNodeId) -> Option<TreeNodeId> {
            // 1. If the current node has a (first) child, return it.
            if let Some(child) = self.store.first_child(node) {
                return Some(child);
            }
            // 2. Otherwise, climb upward
            let mut current = node;
            loop {
                // Try right sibling. Crucially, use right_neighbor_cyclic.
                // Need to detect when we loop back to the first sibling without going up.
                let right_neighbor = self.store.right_neighbor_cyclic(current);

                // Find the parent's first child to detect cycle completion *at this level*.
                let parent_id = self.store[&current];
                let first_child_of_parent = match parent_id {
                    ParentId::Node(p) | ParentId::PointingRoot(p) => self.store.nodes[p.0].child,
                    ParentId::Root(_) => None,
                };

                // If there IS a right neighbor, AND it's NOT the first child (meaning we haven't wrapped)
                if let Some(sibling) = right_neighbor {
                    if Some(sibling) != first_child_of_parent {
                        return Some(sibling); // Go to the sibling
                    }
                    // If sibling == first_child_of_parent, we wrapped around. Need to go up. Fall through.
                }
                // Only child or wrapped around siblings: move upward.
                match parent_id {
                    ParentId::Node(p) | ParentId::PointingRoot(p) => current = p,
                    ParentId::Root(_) => return None, // Reached root while climbing; traversal ends
                }
            }
        }
    }

    impl<V> Iterator for PreorderIter<'_, V> {
        type Item = TreeNodeId;
        fn next(&mut self) -> Option<Self::Item> {
            let node_to_return = self.current?;
            self.current = self.next_node_logic(node_to_return);
            Some(node_to_return)
        }
    }
}

// Bfs Iterator specific to ParentChildStore
mod pc_bfs {
    use super::*;
    #[derive(Clone)]
    pub struct BfsTreeIter<'a, V> {
        store: &'a ParentChildStore<V>,
        queue: VecDeque<TreeNodeId>,
    }
    impl<'a, V> BfsTreeIter<'a, V> {
        pub fn new(store: &'a ParentChildStore<V>, start: TreeNodeId) -> Self {
            let mut queue = VecDeque::new();
            queue.push_back(start);
            BfsTreeIter { store, queue }
        }
    }
    impl<V> Iterator for BfsTreeIter<'_, V> {
        type Item = TreeNodeId;
        fn next(&mut self) -> Option<Self::Item> {
            let node = self.queue.pop_front()?;
            // Use the store's iter_children which correctly uses NeighborIter
            for child in self.store.iter_children(node) {
                self.queue.push_back(child);
            }
            Some(node)
        }
    }
}

impl<V> ForestNodeStorePreorder for ParentChildStore<V> {
    type Iterator<'a>
        = pc_preorder::PreorderIter<'a, V>
    where
        Self: 'a;
    fn iter_preorder(&self, start: TreeNodeId) -> Self::Iterator<'_> {
        pc_preorder::PreorderIter::new(self, start)
    }
}

impl<V> ForestNodeStoreBfs for ParentChildStore<V> {
    fn iter_bfs(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        pc_bfs::BfsTreeIter::new(self, start)
    }
}

impl<V> From<ParentChildStore<V>> for ParentPointerStore<V> {
    fn from(child_store: ParentChildStore<V>) -> Self {
        child_store
            .nodes
            .into_iter()
            .map(|pc_node| pc_node.parent_pointer)
            .collect()
    }
}

impl<V> From<ParentPointerStore<V>> for ParentChildStore<V> {
    fn from(value: ParentPointerStore<V>) -> Self {
        ChildVecStore::from(value).into()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn create() {}
}
