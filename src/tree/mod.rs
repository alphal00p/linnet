//! Defines core data structures and traits for representing forests (collections of trees).
//!
//! This module provides multiple ways to store tree node relationships, each with different
//! trade-offs in terms of memory usage and traversal performance:
//!
//! *   [`ParentPointerStore`]: Each node stores only its data and a pointer to its parent.
//!     Memory efficient, fast upward traversal, slow downward traversal (finding children).
//! *   [`ParentChildStore`]: Uses a first-child, next-sibling representation with cyclic
//!     sibling links. Balances memory and allows efficient child/sibling iteration.
//! *   [`ChildVecStore`]: Each node stores its parent, data, and an explicit `Vec` of its children.
//!     Simple child iteration, potentially higher memory usage for nodes with many children.
//!
//! The core components are:
//! *   [`Forest<R, N>`]: The main struct representing a collection of trees. It stores root data (`R`)
//!     and uses a specific node storage strategy (`N`) that implements [`ForestNodeStore`].
//! *   [`ForestNodeStore`] trait: Defines the basic interface for node storage (adding nodes,
//!     accessing data/parent, mapping data).
//! *   [`ForestNodeStoreDown`] trait: Extends `ForestNodeStore` with methods for downward traversal
//!     (iterating children and leaves).
//! *   [`TreeNodeId`], [`RootId`], [`ParentId`]: Typed identifiers for nodes, roots, and parent links.
//!
//! Conversions between the different store types are provided via `From` implementations.

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
pub mod iterato;
pub mod parent_pointer;
/// A type-safe identifier for a node within a `Forest`.
/// Wraps a `usize` index into the underlying node storage vector.
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

/// Internal data associated with the root of a tree in the `Forest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootData<R> {
    pub(crate) data: R,
    pub(crate) root_id: TreeNodeId,
}

/// A type-safe identifier for a tree within a `Forest`.
/// Wraps a `usize` index into the `Forest`'s roots vector.
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

/// Represents a forest (a collection of disjoint trees).
///
/// `R` is the type of data associated with each root (tree).
/// `N` is the storage implementation for the nodes, which must implement [`ForestNodeStore`].
/// Typically `N` will be one of [`ParentPointerStore<V>`], [`ParentChildStore<V>`], or [`ChildVecStore<V>`],
/// where `V` is the type of data associated with each node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forest<R, N> {
    /// The underlying storage for all nodes in the forest.
    pub(crate) nodes: N,
    /// Metadata for each root/tree in the forest.
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

impl<R, N: ForestNodeStore> IndexMut<TreeNodeId> for Forest<R, N> {
    fn index_mut(&mut self, index: TreeNodeId) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

impl<R, N: Default> Default for Forest<R, N> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Forest Methods (Trait-Bounded) ---

/// Methods available when the node store supports downward traversal.
impl<R, N: ForestNodeStoreDown> Forest<R, N> {
    /// Returns an iterator over the direct children of the node `start`.
    pub fn iter_children(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_children(start)
    }
}

/// Methods available when the node store supports iterating leaves of a specific root.
/// (Note: This is automatically implemented if ForestNodeStoreDown is implemented).
impl<R, N: ForestNodeStoreRootLeaves> Forest<R, N> {
    /// Returns an iterator over all leaf nodes within the tree identified by `root_id`.
    pub fn iter_root_leaves(&self, root_id: RootId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_root_leaves(root_id)
    }
}

/// Methods available when the node store supports pre-order traversal.
impl<R, N: ForestNodeStorePreorder> Forest<R, N> {
    /// Returns a pre-order DFS iterator starting from the given node.
    pub fn iter_preorder(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_preorder(start)
    }
}

/// Methods available when the node store supports BFS traversal.
impl<R, N: ForestNodeStoreBfs> Forest<R, N> {
    /// Returns a BFS iterator starting at the given node.
    pub fn iter_bfs(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_bfs(start)
    }
}

/// Methods available when the node store supports ancestor traversal.
/// (Note: This is automatically implemented for any ForestNodeStore).
impl<R, N: ForestNodeStoreAncestors> Forest<R, N> {
    /// Returns an iterator that traverses upwards from `start_node` towards its root.
    pub fn iter_ancestors(&self, start_node: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_ancestors(start_node)
    }
}

/// General Forest methods available for any `ForestNodeStore`.
impl<R, N: ForestNodeStore> Forest<R, N> {
    pub fn iter_roots(&self) -> impl Iterator<Item = (&R, &TreeNodeId)> {
        self.roots.iter().map(|r| (&r.data, &r.root_id))
    }
    // pub fn iter_root_ids(&self) -> impl Iterator<Item = RootId> { /* ... */
    // }

    pub fn iter_root_ids(&self) -> impl Iterator<Item = RootId> {
        (0..self.roots.len()).map(RootId)
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &N::NodeData)> {
        self.nodes.iter_nodes()
    }
    pub fn iter_node_ids(&self) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.nodes.iter_node_id()
    }

    pub fn root(&self, nodeid: TreeNodeId) -> RootId {
        self.nodes.root(nodeid)
    }

    // Note: map_nodes and map_nodes_ref stay here, constraints adjusted slightly
    pub fn map_nodes<F, T, U>(self, transform: F) -> Forest<R, N::Store<U>>
    where
        N: ForestNodeStore<NodeData = T>,
        N::Store<U>: ForestNodeStore<NodeData = U>,
        F: FnMut(T) -> U,
    {
        Forest {
            nodes: self.nodes.map(transform),
            roots: self.roots,
        }
    }

    pub fn map_nodes_ref<F, T, U>(&self, transform: F) -> Forest<R, N::Store<U>>
    where
        N: ForestNodeStore<NodeData = T>,
        N::Store<U>: ForestNodeStore<NodeData = U>,
        F: FnMut(&T) -> U,
        R: Clone,
    {
        Forest {
            nodes: self.nodes.map_ref(transform),
            // Clone root data; mapping only affects node data type
            roots: self.roots.clone(),
        }
    }
}

/// Methods for constructing/modifying the forest.
impl<R, N> Forest<R, N> {
    /// Adds a new root node to the forest, creating a new tree.
    ///
    /// Takes the data for the new node (`node_data`) and the data for the new tree (`tree_data`).
    /// Returns the `TreeNodeId` of the new root node and the `RootId` of the new tree.
    /// Requires `N` to implement `ForestNodeStore<NodeData = T>`.
    pub fn add_root<T>(&mut self, node_data: T, tree_data: R) -> (TreeNodeId, RootId)
    where
        N: ForestNodeStore<NodeData = T>, // Constraint on the node store type
    {
        let root_id = RootId(self.roots.len());
        let root_node_id = self.nodes.add_root(node_data, root_id);
        self.roots.push(RootData {
            data: tree_data,
            root_id: root_node_id,
        });
        (root_node_id, root_id)
    }

    /// Adds a new child node to the specified parent node.
    ///
    /// Takes the `TreeNodeId` of the parent and the data for the new child node (`node_data`).
    /// Returns the `TreeNodeId` of the newly added child node.
    /// Requires `N` to implement `ForestNodeStore<NodeData = T>`.
    pub fn add_child<T>(&mut self, parent_id: TreeNodeId, node_data: T) -> TreeNodeId
    where
        N: ForestNodeStore<NodeData = T>, // Constraint on the node store type
    {
        self.nodes.add_child(node_data, parent_id)
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

/// Errors that can occur during forest operations.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ForestError {
    #[error("Length mismatch")]
    LengthMismatch,
    #[error("Does not partition: Sets overlap")]
    DoesNotPartion,
    #[error("Does not partition: Sets do not cover the domain")]
    IncompletePartition,
    #[error("Invalid TreeNodeId: {0:?}")]
    InvalidNodeId(TreeNodeId),
    #[error("Invalid RootId: {0:?}")]
    InvalidRootId(RootId),
}

/// Construction from BitVec partition (remains the same)
impl<U: Clone> Forest<U, ParentPointerStore<()>> {
    /// Creates a forest from a partition represented by a vector of (RootData, BitVec).
    /// Each BitVec represents the set of node indices belonging to that tree/root.
    /// Nodes within a set are linked arbitrarily (first encountered becomes root, others point to it).
    /// Node data is set to `()`.
    ///
    /// Returns `Err(ForestError)` if the BitVecs have different lengths or overlap.
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

// --- Core Traits ---

/// The core trait defining the interface for node storage within a `Forest`.
/// Requires Index and IndexMut support for accessing node data and ParentId.
pub trait ForestNodeStore:
    for<'a> Index<&'a TreeNodeId, Output = ParentId>      // `store[&node_id]` -> ParentId
    + Index<TreeNodeId, Output = Self::NodeData>          // `store[node_id]` -> NodeData
    + IndexMut<TreeNodeId, Output = Self::NodeData>   // `store[node_id] = ...`
    + Sized // Needed for some default implementations returning Self iterators
{
    type NodeData;
    type Store<T>: ForestNodeStore<NodeData = T>;

    /// Finds the `RootId` by traversing upwards. Default implementation provided.
    fn root(&self, nodeid: TreeNodeId) -> RootId { let mut current = nodeid;
            loop {
                match self[&current] { // Uses the Index<&TreeNodeId> implementation
                    ParentId::Root(root_id) => return root_id,
                    ParentId::Node(parent_node_id) => current = parent_node_id,
                }
            } }

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)>;
    fn iter_node_id(&self) -> impl Iterator<Item = TreeNodeId> + '_;

    /// Adds a new root node.
    fn add_root(&mut self, data: Self::NodeData, root_id: RootId) -> TreeNodeId;

    /// Adds a new child node as the *last* child of the parent.
    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId;

    fn map<F, U>(self, transform: F) -> Self::Store<U> where F: FnMut(Self::NodeData) -> U;
    fn map_ref<F, U>(&self, transform: F) -> Self::Store<U> where F: FnMut(&Self::NodeData) -> U;
}

/// Trait extension for `ForestNodeStore` providing downward traversal capabilities.
pub trait ForestNodeStoreDown: ForestNodeStore {
    /// Returns an iterator over all leaf nodes in the entire store.
    fn iter_leaves(&self) -> impl Iterator<Item = TreeNodeId> + '_;

    /// Returns an iterator over the direct children of the given `node_id`.
    fn iter_children(&self, node_id: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_;
}

// --- Traversal Traits ---

/// Trait for stores supporting ancestor iteration (upward traversal).
pub trait ForestNodeStoreAncestors: ForestNodeStore {
    /// Returns an iterator from `start_node` up to its root (inclusive).
    fn iter_ancestors(&self, start_node: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_;
}

/// Trait for stores supporting pre-order traversal.
pub trait ForestNodeStorePreorder: ForestNodeStore {
    /// Returns a pre-order DFS iterator starting from `start`.
    fn iter_preorder(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_;
}

/// Trait for stores supporting breadth-first traversal.
pub trait ForestNodeStoreBfs: ForestNodeStore {
    /// Returns a BFS iterator starting from `start`.
    fn iter_bfs(&self, start: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_;
}

/// Trait for stores supporting iteration over leaves belonging to a specific root.
pub trait ForestNodeStoreRootLeaves: ForestNodeStoreDown {
    /// Returns an iterator over the leaf nodes belonging only to the tree with `root_id`.
    fn iter_root_leaves(&self, root_id: RootId) -> impl Iterator<Item = TreeNodeId> + '_;
}

// --- Default Trait Implementations ---

/// Default implementation for ancestor iteration for any `ForestNodeStore`.
impl<N: ForestNodeStore> ForestNodeStoreAncestors for N {
    fn iter_ancestors(&self, start_node: TreeNodeId) -> impl Iterator<Item = TreeNodeId> + '_ {
        // Use the helper struct from the iterators module
        iterato::AncestorsIter::new(self, start_node)
    }
}

/// Default implementation for root leaves iteration for any `ForestNodeStoreDown`.
impl<N: ForestNodeStoreDown> ForestNodeStoreRootLeaves for N {
    fn iter_root_leaves(&self, root_id: RootId) -> impl Iterator<Item = TreeNodeId> + '_ {
        self.iter_leaves()
            .filter(move |&leaf_id| self.root(leaf_id) == root_id)
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
