//! Implements a tree structure using a first-child, next-sibling representation.
//! Each node stores its parent, data, a pointer to its first child, and pointers
//! to its left and right siblings using a *cyclic* doubly linked list for siblings.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::{
    child_vec::ChildVecStore,
    parent_pointer::{PPNode, ParentId, ParentPointerStore},
    ForestNodeStore, ForestNodeStoreBfs, ForestNodeStoreDown, ForestNodeStorePreorder, TreeNodeId,
};

/// Represents a node within a `ParentChildStore`.
///
/// Contains the parent pointer and data (`PPNode`), an optional pointer to the
/// *first* child, and pointers to the left and right siblings. The sibling
/// pointers form a cyclic doubly linked list among all children of the same parent.
/// If a node is the only child, its `neighbor_left` and `neighbor_right` point to itself.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
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
        &self.nodes[index.0].parent_pointer.data
    }
}

impl<V> std::ops::IndexMut<TreeNodeId> for ParentChildStore<V> {
    fn index_mut(&mut self, index: TreeNodeId) -> &mut Self::Output {
        &mut self.nodes[index.0].parent_pointer.data
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

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (TreeNodeId(i), &n.parent_pointer.data))
    }

    fn iter_node_id(&self) -> impl Iterator<Item = TreeNodeId> {
        (0..self.nodes.len()).map(TreeNodeId)
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

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId {
        let new_child_id = TreeNodeId(self.nodes.len());
        debug_assert!(parent.0 < self.nodes.len(), "Parent ID out of bounds");

        let (neighbor_left, neighbor_right) = {
            let parent_node = &mut self.nodes[parent.0];
            if let Some(first_child_id) = parent_node.child {
                // Parent already has children. Insert new child *after* the current last child.
                let last_child_id = self.nodes[first_child_id.0].neighbor_left; // Get current last child (left of first)

                // Update links:
                // 1. New child's left neighbor is the old last child.
                // 2. New child's right neighbor is the first child (completing the cycle).
                // 3. Old last child's right neighbor becomes the new child.
                // 4. First child's left neighbor becomes the new child.

                self.nodes[last_child_id.0].neighbor_right = new_child_id; // 3
                self.nodes[first_child_id.0].neighbor_left = new_child_id; // 4

                (last_child_id, first_child_id) // 1, 2
            } else {
                // Parent had no children. New child is the first and only child.
                parent_node.child = Some(new_child_id); // Set parent's child pointer
                                                        // Neighbors point to self.
                (new_child_id, new_child_id)
            }
        };

        // Add the new node
        self.nodes.push(PCNode {
            parent_pointer: PPNode::child(data, parent),
            child: None, // New node initially has no children
            neighbor_left,
            neighbor_right,
        });

        new_child_id
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
    #[derive(Clone)]
    pub struct PreorderIter<'a, V> {
        store: &'a ParentChildStore<V>,
        current: Option<TreeNodeId>,
        // Could potentially add a stack or use the recursive logic helper
    }

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
                    ParentId::Node(p) => self.store.nodes[p.0].child,
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
                    ParentId::Node(p) => current = p,
                    ParentId::Root(_) => return None, // Reached root while climbing; traversal ends
                }
            }
        }
    }

    impl<'a, V> Iterator for PreorderIter<'a, V> {
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
    fn iter_preorder(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
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
