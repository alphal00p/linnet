use std::ops::{Index, IndexMut};

use bitvec::vec::BitVec;
use child_pointer::ParentChildStore;
use child_vec::ChildVecStore;
use parent_pointer::{PPNode, ParentId, ParentPointerStore};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::half_edge::{
    involution::Hedge,
    subgraph::{Inclusion, SubGraph, SubGraphOps},
    NodeIndex,
};

pub mod child_pointer;
pub mod child_vec;
pub mod parent_pointer;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TreeNodeId(usize);

impl From<usize> for TreeNodeId {
    fn from(i: usize) -> Self {
        TreeNodeId(i)
    }
}

impl From<Hedge> for TreeNodeId {
    fn from(h: Hedge) -> Self {
        h.0.into()
    }
}

impl From<TreeNodeId> for Hedge {
    fn from(h: TreeNodeId) -> Self {
        Hedge(h.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootData<R> {
    pub(crate) data: R,
    pub(crate) root_id: TreeNodeId,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RootId(pub(crate) usize);

impl From<NodeIndex> for RootId {
    fn from(n: NodeIndex) -> Self {
        n.0.into()
    }
}

impl From<RootId> for NodeIndex {
    fn from(r: RootId) -> Self {
        r.0.into()
    }
}

impl From<usize> for RootId {
    fn from(i: usize) -> Self {
        RootId(i)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forest<R, N> {
    pub(crate) nodes: N,
    pub(crate) roots: Vec<RootData<R>>,
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

impl<R, N: ForestNodeStore> Index<&TreeNodeId> for Forest<R, N> {
    type Output = ParentId;
    fn index(&self, index: &TreeNodeId) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<R, N: ForestNodeStore> Index<TreeNodeId> for Forest<R, N> {
    type Output = N::NodeData;
    fn index(&self, index: TreeNodeId) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<R, N: Default> Default for Forest<R, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R, N: ForestNodeStore> Forest<R, N> {
    pub fn iter_roots(&self) -> impl Iterator<Item = (&R, &TreeNodeId)> {
        self.roots.iter().map(|r| (&r.data, &r.root_id))
    }
}

impl<R, N: Default> Forest<R, N> {
    pub fn new() -> Self {
        Forest {
            nodes: N::default(),
            roots: Vec::new(),
        }
    }
}

impl<R, N> Forest<R, N> {
    pub fn add_root<T>(&mut self, node_data: T, tree_data: R) -> (TreeNodeId, RootId)
    where
        N: ForestNodeStore<NodeData = T>,
    {
        let root_id = RootId(self.roots.len());
        let root_node_id = self.nodes.add_root(node_data, root_id);
        self.roots.push(RootData {
            data: tree_data,
            root_id: root_node_id,
        });
        (root_node_id, root_id)
    }

    pub fn add_child<T>(&mut self, parent_id: TreeNodeId, node_data: T) -> TreeNodeId
    where
        N: ForestNodeStore<NodeData = T>,
    {
        self.nodes.add_child(node_data, parent_id)
    }
}

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ForestError {
    #[error("Length mismatch")]
    LengthMismatch,
    #[error("Does not partition")]
    DoesNotPartion,
}

impl<U> Forest<U, ParentPointerStore<()>> {
    pub fn from_bitvec_partition(bitvec_part: Vec<(U, BitVec)>) -> Result<Self, ForestError> {
        let mut nodes = vec![];
        let mut roots = vec![];
        let mut cover: Option<BitVec> = None;

        for (d, set) in bitvec_part {
            let len = set.len();
            if let Some(c) = &mut cover {
                if c.len() != len {
                    return Err(ForestError::LengthMismatch);
                }
                if c.intersects(&set) {
                    return Err(ForestError::DoesNotPartion);
                }
                c.union_with(&set);
            } else {
                cover = Some(BitVec::empty(len));
                nodes = vec![None; len];
            }
            let mut first = None;
            for i in set.included_iter() {
                if let Some(root) = first {
                    nodes[i.0] = Some(PPNode::child((), root))
                } else {
                    first = Some(i.into());
                    nodes[i.0] = Some(PPNode::root((), RootId(roots.len())));
                }
            }
            roots.push(RootData {
                root_id: first.unwrap(),
                data: d,
            });
        }
        Ok(Forest {
            nodes: nodes
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .unwrap()
                .into_iter()
                .collect(),
            roots,
        })
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
        // println!("root");
        match self[&nodeid] {
            ParentId::Root(root) => root,
            ParentId::Node(node) => self.root(node),
        }
    }

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)>;

    fn iter_node_id(&self) -> impl Iterator<Item = TreeNodeId>;

    fn add_root(&mut self, data: Self::NodeData, root_id: RootId) -> TreeNodeId;

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId;

    fn map<F, U>(self, transform: F) -> Self::Store<U>
    where
        F: FnMut(Self::NodeData) -> U;

    fn map_ref<F, U>(&self, transform: F) -> Self::Store<U>
    where
        F: FnMut(&Self::NodeData) -> U;
}

impl<R, N: ForestNodeStore> Forest<R, N> {
    pub fn root(&self, nodeid: TreeNodeId) -> RootId {
        self.nodes.root(nodeid)
    }
}

impl<R, V> From<Forest<R, ParentPointerStore<V>>> for Forest<R, ParentChildStore<V>> {
    fn from(forest: Forest<R, ParentPointerStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}

impl<R, V> From<Forest<R, ParentChildStore<V>>> for Forest<R, ParentPointerStore<V>> {
    fn from(forest: Forest<R, ParentChildStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}

impl<R, V> From<Forest<R, ChildVecStore<V>>> for Forest<R, ParentChildStore<V>> {
    fn from(forest: Forest<R, ChildVecStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}

impl<R, V> From<Forest<R, ParentChildStore<V>>> for Forest<R, ChildVecStore<V>> {
    fn from(forest: Forest<R, ParentChildStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}

impl<R, V> From<Forest<R, ChildVecStore<V>>> for Forest<R, ParentPointerStore<V>> {
    fn from(forest: Forest<R, ChildVecStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}

impl<R, V> From<Forest<R, ParentPointerStore<V>>> for Forest<R, ChildVecStore<V>> {
    fn from(forest: Forest<R, ParentPointerStore<V>>) -> Self {
        Forest {
            nodes: forest.nodes.into(),
            roots: forest.roots,
        }
    }
}
