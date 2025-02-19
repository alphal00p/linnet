use std::collections::VecDeque;

use super::{
    child_vec::ChildVecStore,
    parent_pointer::{PPNode, ParentId, ParentPointerStore},
    ForestNodeStore, ForestNodeStoreDown, TreeNodeId,
};

pub struct PCNode<V> {
    pub(crate) parent_pointer: PPNode<V>,
    pub(crate) child: Option<TreeNodeId>, // first child, if any. To get the other children, follow the sibling links of this child
    pub(crate) neighbor_left: TreeNodeId, // next sibling, if any. Will form a cyclic linked list, if equal to right, then it is the only child
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

pub struct ParentChildStore<V> {
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

pub enum NeighborIter<'a, V> {
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
    pub fn left_neighbor_cyclic(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        if self.nodes[node_id.0].neighbor_left == node_id {
            None
        } else {
            Some(self.nodes[node_id.0].neighbor_left)
        }
    }

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

    pub fn first_child(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        self.nodes[node_id.0].child
    }

    pub fn right_neighbor_cyclic(&self, node_id: TreeNodeId) -> Option<TreeNodeId> {
        if self.nodes[node_id.0].neighbor_right == node_id {
            None
        } else {
            Some(self.nodes[node_id.0].neighbor_right)
        }
    }

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
        let node_id = TreeNodeId(self.nodes.len());
        let child = self.nodes[parent.0].child;
        let new_child = node_id;
        let neighbor_left;
        let neighbor_right;

        self.nodes[parent.0].child = Some(new_child);
        if let Some(child) = child {
            let childnode = &mut self.nodes[child.0];
            let neighbor_left_child = childnode.neighbor_left;
            childnode.neighbor_left = new_child;
            neighbor_right = child;

            let neighbor_left_node = &mut self.nodes[neighbor_left_child.0];
            neighbor_left_node.neighbor_right = new_child;
            neighbor_left = neighbor_left_child;
        } else {
            neighbor_left = new_child;
            neighbor_right = new_child;
        }

        self.nodes.push(PCNode {
            parent_pointer: PPNode::child(data, parent),
            child,
            neighbor_left,
            neighbor_right,
        });
        node_id
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

/// A pre–order DFS-iterator over a tree (using the parent/child store).
pub struct PreorderIter<'a, V> {
    store: &'a ParentChildStore<V>,
    current: Option<TreeNodeId>,
}

impl<V> ParentChildStore<V> {
    /// Returns a pre–order DFS iterator starting from the given node.
    pub fn iter_preorder(&self, start: TreeNodeId) -> PreorderIter<V> {
        PreorderIter::new(self, start)
    }
}

impl<'a, V> PreorderIter<'a, V> {
    /// Create a new iterator starting at `start`. Typically `start` is a root node.
    pub fn new(store: &'a ParentChildStore<V>, start: TreeNodeId) -> Self {
        PreorderIter {
            store,
            current: Some(start),
        }
    }
}

impl<V> Iterator for PreorderIter<'_, V> {
    type Item = TreeNodeId;
    fn next(&mut self) -> Option<Self::Item> {
        let node = self.current?;
        // Compute the next node in pre-order:
        self.current = self.next_preorder(node);
        Some(node)
    }
}

impl<V> PreorderIter<'_, V> {
    /// Returns the next node (if any) in pre–order after `node`.
    ///
    /// The algorithm is as follows:
    /// 1. If `node` has a first child, that is the next node.
    /// 2. Otherwise, climb the parent chain until a node with a right sibling is found.
    /// 3. If none is found, the traversal is finished.
    pub fn next_preorder(&self, node: TreeNodeId) -> Option<TreeNodeId> {
        // 1. If the current node has a (first) child, return it.
        if let Some(child) = self.store.first_child(node) {
            return Some(child);
        }
        // 2. Otherwise, climb upward until we hit a node with an available right sibling.
        let mut current = node;
        loop {
            // Using our store’s method to ask: “Is there a right sibling?”
            if let Some(sibling) = self.store.right_neighbor_ending(current) {
                return Some(sibling);
            }
            // No sibling here; move upward if possible.
            let parent = self.store[&current];
            match parent {
                ParentId::Node(p) => current = p,
                ParentId::Root(_) => return None, // we reached a root; no more nodes in pre–order
            }
        }
    }
}

pub struct BfsTreeIter<'a, V> {
    store: &'a ParentChildStore<V>,
    queue: VecDeque<TreeNodeId>,
}

impl<V> ParentChildStore<V> {
    /// Returns a BFS iterator starting at the given node.
    pub fn iter_bfs(&self, start: TreeNodeId) -> BfsTreeIter<V> {
        BfsTreeIter::new(self, start)
    }
}

impl<'a, V> BfsTreeIter<'a, V> {
    /// Create a new BFS iterator starting at `start`.
    pub fn new(store: &'a ParentChildStore<V>, start: TreeNodeId) -> Self {
        let mut queue = VecDeque::new();
        // Start with the root node.
        queue.push_back(start);
        BfsTreeIter { store, queue }
    }
}

impl<V> Iterator for BfsTreeIter<'_, V> {
    type Item = TreeNodeId;
    fn next(&mut self) -> Option<Self::Item> {
        // Remove the node at the front (FIFO behavior)
        let node = self.queue.pop_front()?;
        // Enqueue all children of the current node.
        // We use iter_children(), which (via your neighbor functions)
        // returns all of the children in order.
        for child in self.store.iter_children(node) {
            self.queue.push_back(child);
        }
        Some(node)
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
