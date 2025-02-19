use std::ops::{Index, IndexMut};

use parent_pointer::ParentId;

pub mod child_pointer;
pub mod parent_pointer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TreeNodeId(usize);
pub struct RootData<R> {
    data: R,
    root_id: TreeNodeId,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RootId(usize);

pub struct Forest<R, N> {
    nodes: N,
    roots: Vec<RootData<R>>,
}

impl<R, N> Index<&RootId> for Forest<R, N> {
    type Output = TreeNodeId;
    fn index(&self, index: &RootId) -> &Self::Output {
        &self.roots[index.0].root_id
    }
}

impl<R, N> Index<RootId> for Forest<R, N> {
    type Output = R;
    fn index(&self, index: RootId) -> &Self::Output {
        &self.roots[index.0].data
    }
}

impl<R, N> IndexMut<RootId> for Forest<R, N> {
    fn index_mut(&mut self, index: RootId) -> &mut Self::Output {
        &mut self.roots[index.0].data
    }
}

impl<R, N: Default> Default for Forest<R, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R, N: Default> Forest<R, N> {
    pub fn new() -> Self {
        Forest {
            nodes: N::default(),
            roots: Vec::new(),
        }
    }

    pub fn add_root<T>(&mut self, node_data: T, tree_data: R) -> TreeNodeId
    where
        N: ForestNodeStore<NodeData = T>,
    {
        let root_id = self.nodes.add_root(node_data, RootId(self.roots.len()));
        self.roots.push(RootData {
            data: tree_data,
            root_id,
        });
        root_id
    }
}
pub trait ForestNodeStoreDown: ForestNodeStore {
    fn iter_leaves(&self) -> impl Iterator<Item = TreeNodeId>;

    fn iter_children(&self, node_id: TreeNodeId) -> impl Iterator<Item = TreeNodeId>;
}

pub trait ForestNodeStore:
    for<'a> Index<&'a TreeNodeId, Output = ParentId> + Index<TreeNodeId, Output = Self::NodeData>
{
    type NodeData;
    type Store<T>: ForestNodeStore<NodeData = T>;

    fn root(&self, nodeid: TreeNodeId) -> RootId {
        match self[&nodeid] {
            ParentId::Root(root) => root,
            ParentId::Node(node) => self.root(node),
        }
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)>;

    fn add_root(&mut self, data: Self::NodeData, root_id: RootId) -> TreeNodeId;

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId;

    fn map<F, U>(self, transform: F) -> Self::Store<U>
    where
        F: FnMut(Self::NodeData) -> U;

    fn map_ref<F, U>(&self, transform: F) -> Self::Store<U>
    where
        F: FnMut(&Self::NodeData) -> U;
}
