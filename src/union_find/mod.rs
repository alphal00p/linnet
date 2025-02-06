// First, define the newtypes for node and data indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParentPointer(pub usize);

impl From<usize> for ParentPointer {
    fn from(x: usize) -> Self {
        ParentPointer(x)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SetIndex(pub usize);

impl From<usize> for SetIndex {
    fn from(x: usize) -> Self {
        SetIndex(x)
    }
}

use std::{
    cell::Cell,
    ops::{Index, IndexMut},
};

/// The enum representing a node in the union–find tree.
///
/// - `Root { set_data_idx, rank }` means this node is a root and stores its
///   union–by–rank value and an index (of type `DataIndex`) into the associated data.
/// - `Child(parent)` means this node is not a root; it points to its parent (a `NodeIndex`).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UFNode {
    Root { set_data_idx: SetIndex, rank: usize },
    Child(ParentPointer),
}

/// The basic union–find structure. It partitions a contiguous vector of base data (`T`)
/// and—for each active set—stores associated data (`U`) in a separate compact vector.
///
/// Internally:
/// - The parent–pointer tree is stored as a `Vec<Cell<Node>>` so that `find` (with path compression)
///   can update parent pointers even on an immutable reference.
/// - Each root’s associated data index is stored both in its `Node::Root` variant and in a separate
///   mapping (`node_data: Vec<DataIndex>`). The inverse mapping (`data_to_node: Vec<NodeIndex>`)
///   tells you which node owns a given slot in `set_data`.
pub struct UnionFind<T, U> {
    /// The base storage (kept contiguous so you can later filter or index it).
    pub elements: Vec<T>,
    /// The parent–pointer tree (each node is wrapped in a `Cell`).
    nodes: Vec<Cell<UFNode>>,
    /// Associated data for *roots*. Non-roots do not own data. We store an Option<U>
    /// so we can "take" it out when merging without needing U: Clone.
    set_data: Vec<Option<U>>,
    /// Inverse mapping: for each slot in `set_data`, the node (current root) that owns that slot.
    data_to_node: Vec<ParentPointer>,
}

impl<T, U> UnionFind<T, U> {
    /// Creates a new union–find.
    ///
    /// Both `elements` and `associated` must have the same length. Initially, each element is in its own set
    /// and its associated data is stored at the same index (wrapped as a `DataIndex`) in `set_data`.
    pub fn new(elements: Vec<T>, associated: Vec<U>) -> Self {
        let n = elements.len();
        assert_eq!(
            n,
            associated.len(),
            "elements and associated data must have the same length"
        );
        let nodes = (0..n)
            .map(|i| {
                Cell::new(UFNode::Root {
                    set_data_idx: SetIndex(i),
                    rank: 0,
                })
            })
            .collect();
        let data_to_node = (0..n).map(ParentPointer).collect();
        Self {
            elements,
            nodes,
            set_data: associated.into_iter().map(Some).collect(),
            data_to_node,
        }
    }

    /// Finds the representative (root) of the set containing `x`, performing path compression.
    ///
    /// This method takes only `&self` because each node is stored in a `Cell`, allowing interior mutation.
    pub fn find(&self, x: ParentPointer) -> ParentPointer {
        let node = self.nodes[x.0].get();
        match node {
            UFNode::Root { .. } => x,
            UFNode::Child(parent) => {
                let root = self.find(parent);
                // Path compression: update x's pointer to point directly to the root.
                self[&x].set(UFNode::Child(root));
                root
            }
        }
    }

    /// Returns a reference to the associated data for the set containing `x`.
    pub fn find_data(&self, x: ParentPointer) -> &U {
        &self[self.find_data_index(x)]
    }

    /// Returns a reference to the associated data for the set containing `x`.
    pub fn find_data_index(&self, x: ParentPointer) -> SetIndex {
        let root = self.find(x);
        match self[&root].get() {
            UFNode::Root { set_data_idx, .. } => set_data_idx,
            _ => unreachable!("find() should always return a root"),
        }
    }

    /// Merges the sets containing `x` and `y` using the provided merger function.
    ///
    /// The merger function receives two copies of the associated data (thus requiring `U: Clone`)
    /// and returns a new value. The winning set (determined by union–by–rank) retains its data slot,
    /// while the losing set's slot is removed via swap removal (with appropriate pointer updates).
    pub fn union<F>(&mut self, x: ParentPointer, y: ParentPointer, merge: F) -> ParentPointer
    where
        F: FnOnce(U, U) -> U,
        U: Clone,
    {
        let root_x = self.find(x);
        let root_y = self.find(y);
        if root_x == root_y {
            return root_x;
        }

        // Retrieve rank and associated data index from each root.
        let (rank_x, data_idx_x) = match self[&root_x].get() {
            UFNode::Root { rank, set_data_idx } => (rank, set_data_idx),
            _ => unreachable!(),
        };
        let (rank_y, data_idx_y) = match self[&root_y].get() {
            UFNode::Root { rank, set_data_idx } => (rank, set_data_idx),
            _ => unreachable!(),
        };

        // Decide the winner and loser by union–by–rank.
        let (winner, loser, winner_data_idx, loser_data_idx) = if rank_x < rank_y {
            (root_y, root_x, data_idx_y, data_idx_x)
        } else {
            (root_x, root_y, data_idx_x, data_idx_y)
        };

        // If the ranks are equal, increment the winner's rank.
        if rank_x == rank_y {
            if let UFNode::Root { set_data_idx, rank } = self.nodes[winner.0].get() {
                self.nodes[winner.0].set(UFNode::Root {
                    set_data_idx,
                    rank: rank + 1,
                });
            }
        }

        // Make the loser point to the winner.
        self.nodes[loser.0].set(UFNode::Child(winner));

        // Merge the associated data.
        let merged = merge(
            self.set_data[winner_data_idx.0].take().unwrap(),
            self.set_data[loser_data_idx.0].take().unwrap(),
        );
        self.set_data[winner_data_idx.0] = Some(merged);

        // Remove the losing set’s associated data from the compact storage via swap removal.
        let last_idx = self.set_data.len() - 1;
        if loser_data_idx.0 != last_idx {
            self.set_data.swap(loser_data_idx.0, last_idx);
            self.data_to_node.swap(loser_data_idx.0, last_idx);
            // Update the node that now owns the swapped-in data.
            let swapped_node = self.data_to_node[loser_data_idx.0];
            if let UFNode::Root { rank, .. } = self.nodes[swapped_node.0].get() {
                self.nodes[swapped_node.0].set(UFNode::Root {
                    set_data_idx: SetIndex(loser_data_idx.0),
                    rank,
                });
            }
            // If the winner’s data slot was at the end, update it.
            if winner_data_idx.0 == last_idx {
                if let UFNode::Root { rank, .. } = self.nodes[winner.0].get() {
                    self.nodes[winner.0].set(UFNode::Root {
                        set_data_idx: SetIndex(loser_data_idx.0),
                        rank,
                    });
                }
            }
        }
        self.set_data.pop();
        self.data_to_node.pop();

        winner
    }

    pub fn map_set_data_of<F>(&mut self, x: ParentPointer, f: F)
    where
        F: FnOnce(&mut U),
    {
        let set_id = self.find_data_index(x);
        f(&mut self[set_id]);
    }

    /// `map_into` consumes `self` and produces a brand-new `UnionFind<T2, U2>`
    /// that has the *same union structure*, but with `T` transformed into `T2`
    /// and `U` transformed into `U2`.
    ///
    /// This does **not** require `T: Clone` or `U: Clone` because we are
    /// taking ownership of the old vectors (`elements`, `set_data`, etc.).
    ///
    /// - `f_elem` will be applied to each element `T`.
    /// - `f_data` will be applied to each associated-data `U`.
    ///
    /// All rank/parent relationships remain exactly the same.
    pub fn map_into<T2, U2, F1, F2>(self, f_elem: F1, mut f_data: F2) -> UnionFind<T2, U2>
    where
        F1: FnMut(T) -> T2,
        F2: FnMut(U) -> U2,
    {
        // 1) Transform the elements
        let elements: Vec<T2> = self.elements.into_iter().map(f_elem).collect();

        // 2) Transform set_data: each slot is Option<U>.
        // We'll map the "Some(u)" to "Some(f_data(u))".
        let set_data: Vec<Option<U2>> = self
            .set_data
            .into_iter()
            .map(|maybe_u| maybe_u.map(&mut f_data))
            .collect();

        // 3) We can keep `data_to_node` as-is (just `Vec<ParentPointer>`).
        let data_to_node = self.data_to_node;

        // 4) Replicate each Node inside a new `Cell` (so the new UnionFind
        //    has distinct `Cell`s, but the same child/parent/rank data).
        let nodes: Vec<Cell<UFNode>> = self
            .nodes
            .into_iter()
            .map(|cell_old| {
                let node_old = cell_old.get();
                Cell::new(node_old)
            })
            .collect();

        UnionFind {
            elements,
            nodes,
            data_to_node,
            set_data,
        }
    }

    pub fn replace_set_data_of<F>(&mut self, x: ParentPointer, f: F)
    where
        F: FnOnce(U) -> U,
    {
        let set_id = self.find_data_index(x);

        // Take ownership of the old data
        let old_data = self.set_data[set_id.0].take().expect("data must exist");
        let new_data = f(old_data);
        self[set_id] = new_data;
    }
}

impl<T, U> Index<SetIndex> for UnionFind<T, U> {
    type Output = U;
    fn index(&self, index: SetIndex) -> &Self::Output {
        self.set_data[index.0].as_ref().expect("data must exist")
    }
}

impl<T, U> IndexMut<SetIndex> for UnionFind<T, U> {
    fn index_mut(&mut self, index: SetIndex) -> &mut Self::Output {
        self.set_data[index.0].as_mut().expect("data must exist")
    }
}

impl<T, U> Index<ParentPointer> for UnionFind<T, U> {
    type Output = T;
    fn index(&self, index: ParentPointer) -> &Self::Output {
        &self.elements[index.0]
    }
}

impl<T, U> Index<&ParentPointer> for UnionFind<T, U> {
    type Output = Cell<UFNode>;
    fn index(&self, index: &ParentPointer) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl<T, U> IndexMut<ParentPointer> for UnionFind<T, U> {
    fn index_mut(&mut self, index: ParentPointer) -> &mut Self::Output {
        &mut self.elements[index.0]
    }
}

pub mod bitvec_find;
#[cfg(test)]
pub mod test;
