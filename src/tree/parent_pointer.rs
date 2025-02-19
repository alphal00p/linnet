use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use super::{ForestNodeStore, RootId, TreeNodeId};
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PPNode<V> {
    pub(crate) parent: ParentId,
    pub(crate) data: V,
}

impl<V> PPNode<V> {
    pub fn child(data: V, parent: TreeNodeId) -> Self {
        PPNode {
            parent: ParentId::Node(parent),
            data,
        }
    }

    pub fn root(data: V, root_id: RootId) -> Self {
        PPNode {
            parent: ParentId::Root(root_id),
            data,
        }
    }

    pub fn map<F, U>(self, mut transform: F) -> PPNode<U>
    where
        F: FnMut(V) -> U,
    {
        PPNode {
            parent: self.parent,
            data: transform(self.data),
        }
    }

    pub fn map_ref<F, U>(&self, mut transform: F) -> PPNode<U>
    where
        F: FnMut(&V) -> U,
    {
        PPNode {
            parent: self.parent,
            data: transform(&self.data),
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ParentId {
    Root(RootId),
    Node(TreeNodeId),
}

impl ParentId {
    pub fn is_root(&self) -> bool {
        match self {
            ParentId::Root(_) => true,
            ParentId::Node(_) => false,
        }
    }

    pub fn is_node(&self) -> bool {
        !self.is_root()
    }
}

pub struct ParentPointerStore<V> {
    pub(crate) nodes: Vec<PPNode<V>>,
}

impl<V> ParentPointerStore<V> {
    /// Re–roots the tree at the given node (making it a root).
    /// Along the chain from new_root to the old root the parent pointers are reversed.
    pub fn change_root(&mut self, new_root: TreeNodeId) -> RootId {
        let mut current = new_root;
        let mut prev = None;
        loop {
            // Save the original parent.
            let orig_parent = self.nodes[current.0].parent;
            // Update current’s parent pointer: if there is no previous node we want a new Root,
            // otherwise we point to the previous node.
            self.nodes[current.0].parent = match prev {
                None => ParentId::Root(crate::tree::RootId(new_root.0)),
                Some(p) => ParentId::Node(p),
            };
            // If the original parent was a node, move upward.
            match orig_parent {
                ParentId::Node(p) => {
                    prev = Some(current);
                    current = p;
                }
                ParentId::Root(r) => return r,
            }
        }
    }
}

impl<V> FromIterator<PPNode<V>> for ParentPointerStore<V> {
    fn from_iter<I: IntoIterator<Item = PPNode<V>>>(iter: I) -> Self {
        ParentPointerStore {
            nodes: iter.into_iter().collect(),
        }
    }
}

impl<V> Index<&TreeNodeId> for ParentPointerStore<V> {
    type Output = ParentId;
    fn index(&self, index: &TreeNodeId) -> &Self::Output {
        &self.nodes[index.0].parent
    }
}

impl<V> Index<TreeNodeId> for ParentPointerStore<V> {
    type Output = V;
    fn index(&self, index: TreeNodeId) -> &Self::Output {
        &self.nodes[index.0].data
    }
}

impl<V> IndexMut<TreeNodeId> for ParentPointerStore<V> {
    fn index_mut(&mut self, index: TreeNodeId) -> &mut Self::Output {
        &mut self.nodes[index.0].data
    }
}

impl<V> ForestNodeStore for ParentPointerStore<V> {
    type NodeData = V;
    type Store<T> = ParentPointerStore<T>;

    fn add_root(&mut self, data: Self::NodeData, root_id: RootId) -> TreeNodeId {
        let node_id = TreeNodeId(self.nodes.len());
        self.nodes.push(PPNode::root(data, root_id));
        node_id
    }

    fn iter_node_id(&self) -> impl Iterator<Item = TreeNodeId> {
        (0..self.nodes.len()).map(TreeNodeId)
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (TreeNodeId(i), &n.data))
    }

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId {
        let node_id = TreeNodeId(self.nodes.len());
        self.nodes.push(PPNode::child(data, parent));
        node_id
    }

    fn map<F, U>(self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(Self::NodeData) -> U,
    {
        self.nodes
            .into_iter()
            .map(|n| n.map(&mut transform))
            .collect()
    }

    fn map_ref<F, U>(&self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(&Self::NodeData) -> U,
    {
        self.nodes
            .iter()
            .map(|n| n.map_ref(&mut transform))
            .collect()
    }
}
