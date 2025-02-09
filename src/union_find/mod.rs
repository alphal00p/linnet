use std::{
    cell::Cell,
    ops::{Index, IndexMut},
};

/// A newtype for a node (index into `self.nodes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParentPointer(pub usize);

impl From<usize> for ParentPointer {
    fn from(x: usize) -> Self {
        ParentPointer(x)
    }
}

/// A newtype for the set–data index (index into `UnionFind::set_data`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SetIndex(pub usize);

impl From<usize> for SetIndex {
    fn from(x: usize) -> Self {
        SetIndex(x)
    }
}

/// A node can be:
/// - `Root { set_data_idx, rank }`: this node is a root, with `rank` for union–by–rank,
///   and it owns the data at `set_data_idx`.
/// - `Child(parent)`: a non–root pointing to another node's index.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UFNode {
    Root { set_data_idx: SetIndex, rank: usize },
    Child(ParentPointer),
}

/// A UnionFind structure that:
/// - Stores a parallel `elements: Vec<T>` (one per node).
/// - Maintains a parent–pointer forest (`Vec<Cell<UFNode>>`).
/// - Stores associated data (`U`) for each root in `set_data: Vec<Option<U>>`,
///   using swap–removal when merging.
/// - `data_to_node` is the inverse of `set_data_idx`, telling us which node currently owns
///   each slot in `set_data`.
pub struct UnionFind<T, U> {
    /// The base storage of node elements (parallel to `nodes`).
    pub elements: Vec<T>,

    /// Each node is a `Cell<UFNode>` for in-place mutation during path compression.
    pub nodes: Vec<Cell<UFNode>>,

    /// For each root, there's exactly one `Some(U)` slot here.
    /// Non–roots may have been swapped out to maintain compactness.
    set_data: Vec<SetData<U>>,
}

pub struct SetData<U> {
    root_pointer: ParentPointer,
    data: Option<U>,
}

pub fn left<E>(l: E, _: E) -> E {
    l
}

pub fn right<E>(_: E, r: E) -> E {
    r
}

impl<T, U> UnionFind<T, U> {
    /// Builds a union-find where each node is its own set, with `rank=0` and `SetIndex(i)` owning
    /// the `i`th slot in `set_data`.
    pub fn new(elements: Vec<T>, associated: Vec<U>) -> Self {
        let n = elements.len();
        assert_eq!(n, associated.len());
        let nodes = (0..n)
            .map(|i| {
                Cell::new(UFNode::Root {
                    set_data_idx: SetIndex(i),
                    rank: 0,
                })
            })
            .collect();

        let set_data = associated
            .into_iter()
            .enumerate()
            .map(|(i, d)| SetData {
                root_pointer: ParentPointer(i),
                data: Some(d),
            })
            .collect();

        Self {
            elements,
            nodes,
            set_data,
        }
    }

    /// **Find** the representative (root) of the set containing `x`, path compressing along the way.
    pub fn find(&self, x: ParentPointer) -> ParentPointer {
        match self[&x].get() {
            UFNode::Root { .. } => x,
            UFNode::Child(parent) => {
                let root = self.find(parent);
                // path compression
                self[&x].set(UFNode::Child(root));
                root
            }
        }
    }

    /// Returns the `SetIndex` for the set containing `x`.
    pub fn find_data_index(&self, x: ParentPointer) -> SetIndex {
        let root = self.find(x);
        match self[&root].get() {
            UFNode::Root { set_data_idx, .. } => set_data_idx,
            UFNode::Child(_) => unreachable!("find always returns a root"),
        }
    }

    /// Returns a shared reference to the data for `x`'s set, unwrapping the `Option`.
    /// Panics if no data is present (which shouldn't happen for a valid root).
    pub fn find_data(&self, x: ParentPointer) -> &U {
        &self[self.find_data_index(x)]
    }

    /// **Union** the sets containing `x` and `y`, merging their data with `merge(U, U) -> U`.
    ///
    /// - Union–by–rank
    /// - Merged data is placed in the winner’s slot.
    /// - Loser’s slot is swap–removed from `set_data`.
    /// - Returns the new root.
    pub fn union<F>(&mut self, x: ParentPointer, y: ParentPointer, merge: F) -> ParentPointer
    where
        F: FnOnce(U, U) -> U,
    {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return rx;
        }

        // Extract rank + data index from each root
        let (rank_x, data_x) = match self[&rx].get() {
            UFNode::Root { rank, set_data_idx } => (rank, set_data_idx),
            _ => unreachable!(),
        };
        let (rank_y, data_y) = match self[&ry].get() {
            UFNode::Root { rank, set_data_idx } => (rank, set_data_idx),
            _ => unreachable!(),
        };

        let (winner, loser, winner_data_idx, loser_data_idx, same_rank) = match rank_x.cmp(&rank_y)
        {
            std::cmp::Ordering::Less => (ry, rx, data_y, data_x, false),
            std::cmp::Ordering::Greater => (rx, ry, data_x, data_y, false),
            std::cmp::Ordering::Equal => (rx, ry, data_x, data_y, true),
        };

        if same_rank {
            if let UFNode::Root { set_data_idx, rank } = self[&winner].get() {
                self[&winner].set(UFNode::Root {
                    set_data_idx,
                    rank: rank + 1,
                });
            }
        }

        // Make loser point to winner
        self[&loser].set(UFNode::Child(winner));

        // Merge their data
        // We can now fetch the `Option<U>` for each side using `uf[&SetIndex]`.
        let winner_opt = self[&winner_data_idx].data.take();
        let loser_opt = self[&loser_data_idx].data.take();

        // Take out the two `U`s, merge them, store back in winner's slot
        let merged = merge(
            winner_opt.expect("winner has no data?"),
            loser_opt.expect("loser has no data?"),
        );
        self[&winner_data_idx].data = Some(merged);

        // Swap–remove from the losing slot
        let last_idx = self.set_data.len() - 1;
        if loser_data_idx.0 != last_idx {
            // Swap data
            self.set_data.swap(loser_data_idx.0, last_idx);

            // Fix the node that got swapped in
            let swapped_node = self.set_data[loser_data_idx.0].root_pointer;
            if let UFNode::Root { set_data_idx, rank } = self[&swapped_node].get() {
                // set_data_idx might have been the old `last_idx`.
                // Now it must be `loser_data_idx`.
                if set_data_idx.0 == last_idx {
                    self[&swapped_node].set(UFNode::Root {
                        set_data_idx: loser_data_idx,
                        rank,
                    });
                }
            }

            // If the winner's data was at last_idx, fix it
            if winner_data_idx.0 == last_idx {
                if let UFNode::Root { set_data_idx, rank } = self[&winner].get() {
                    if set_data_idx.0 == last_idx {
                        // point it to the new location
                        self[&winner].set(UFNode::Root {
                            set_data_idx: loser_data_idx,
                            rank,
                        });
                    }
                }
            }
        }
        self.set_data.pop();

        winner
    }

    /// Allows mutating the set–data of `x` in place, unwrapping the `Option`.
    pub fn map_set_data_of<F>(&mut self, x: ParentPointer, f: F)
    where
        F: FnOnce(&mut U),
    {
        let idx = self.find_data_index(x);
        let data_ref = &mut self[idx]; // &mut U
        f(data_ref);
    }

    /// Takes ownership of the old data for `x`'s set, applies a function, and replaces it.
    pub fn replace_set_data_of<F>(&mut self, x: ParentPointer, f: F)
    where
        F: FnOnce(U) -> U,
    {
        let idx = self.find_data_index(x);
        let old_data = self[&idx].data.take().expect("no data to replace");
        self[&idx].data.replace(f(old_data));
    }
}

// -------------------------------------------------------------------
// Index impls
// -------------------------------------------------------------------

/// 1) `impl Index<SetIndex>` => returns `&U` (unwrapped from `Option<U>`).
impl<T, U> Index<SetIndex> for UnionFind<T, U> {
    type Output = U;
    fn index(&self, idx: SetIndex) -> &Self::Output {
        self.set_data[idx.0]
            .data
            .as_ref()
            .expect("no data in that slot!")
    }
}

/// 1b) `impl IndexMut<SetIndex>` => returns `&mut U`.
impl<T, U> IndexMut<SetIndex> for UnionFind<T, U> {
    fn index_mut(&mut self, idx: SetIndex) -> &mut Self::Output {
        self.set_data[idx.0]
            .data
            .as_mut()
            .expect("no data in that slot!")
    }
}

/// 2) `impl Index<&SetIndex>` => returns `&Option<U>`.
///    This lets you see if it’s Some/None, or call methods like `.take()`.
impl<T, U> Index<&SetIndex> for UnionFind<T, U> {
    type Output = SetData<U>;
    fn index(&self, idx: &SetIndex) -> &Self::Output {
        &self.set_data[idx.0]
    }
}

/// 2b) `impl IndexMut<&SetIndex>` => returns `&mut Option<U>`.
impl<T, U> IndexMut<&SetIndex> for UnionFind<T, U> {
    fn index_mut(&mut self, idx: &SetIndex) -> &mut Self::Output {
        &mut self.set_data[idx.0]
    }
}

/// `impl Index<ParentPointer>` => returns `&T`.
/// This is the direct element in `elements`.
impl<T, U> Index<ParentPointer> for UnionFind<T, U> {
    type Output = T;
    fn index(&self, idx: ParentPointer) -> &Self::Output {
        &self.elements[idx.0]
    }
}

/// `impl IndexMut<ParentPointer>` => returns `&mut T`.
impl<T, U> IndexMut<ParentPointer> for UnionFind<T, U> {
    fn index_mut(&mut self, idx: ParentPointer) -> &mut Self::Output {
        &mut self.elements[idx.0]
    }
}

/// `impl Index<&ParentPointer>` => returns `&Cell<UFNode>`,
/// allowing `self[&x].get()` or `self[&x].set(...)`.
impl<T, U> Index<&ParentPointer> for UnionFind<T, U> {
    type Output = Cell<UFNode>;
    fn index(&self, idx: &ParentPointer) -> &Self::Output {
        &self.nodes[idx.0]
    }
}

pub mod bitvec_find;
#[cfg(test)]
pub mod test;
