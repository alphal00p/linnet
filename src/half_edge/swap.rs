use std::ops::{AddAssign, SubAssign};

use crate::permutation::Permutation;

use super::{
    hedgevec::SmartEdgeVec,
    involution::{EdgeIndex, Hedge},
    nodestore::NodeStorageOps,
    HedgeGraph, NodeIndex,
};

pub trait Swap<Index> {
    fn swap(&mut self, i: Index, j: Index);

    fn len(&self) -> Index;

    fn is_empty(&self) -> bool;

    fn permute(&mut self, perm: &Permutation)
    where
        Index: From<usize>,
    {
        let trans = perm.transpositions();

        // println!("trans:{trans:?}");

        for (i, j) in trans.into_iter().rev() {
            self.swap(Index::from(i), Index::from(j));
        }
    }

    fn partition(&mut self, filter: impl Fn(&Index) -> bool) -> Index
    where
        Index: From<usize> + AddAssign + PartialOrd + SubAssign + Copy,
    {
        let mut left = Index::from(0);
        let one = Index::from(1);
        let mut extracted = self.len();

        while left < extracted {
            if filter(&left) {
                //left is in the right place
                left += one;
            } else {
                //left needs to be swapped
                extracted -= one;
                if filter(&extracted) {
                    // println!("{extracted}<=>{left}");
                    //only with an extracted that is in the wrong spot
                    self.swap(left, extracted);
                    left += one;
                }
            }
        }
        left
    }
}
impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<Hedge> for HedgeGraph<E, V, H, N> {
    fn len(&self) -> Hedge {
        self.hedge_data.len()
    }

    fn is_empty(&self) -> bool {
        self.hedge_data.is_empty()
    }

    fn swap(&mut self, i: Hedge, j: Hedge) {
        // println!("Swapping {i:?} with {j:?}");
        self.hedge_data.swap(i, j);
        // println!("nodeswap");
        self.node_store.swap(i, j);
        // println!("edgeswap");
        self.edge_store.swap(i, j);
    }
}

impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<EdgeIndex> for HedgeGraph<E, V, H, N> {
    fn swap(&mut self, i: EdgeIndex, j: EdgeIndex) {
        self.edge_store.swap(i, j);
    }

    fn is_empty(&self) -> bool {
        <SmartEdgeVec<E> as Swap<EdgeIndex>>::is_empty(&self.edge_store)
    }

    fn len(&self) -> EdgeIndex {
        self.edge_store.len()
    }
}

impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<NodeIndex> for HedgeGraph<E, V, H, N> {
    fn swap(&mut self, i: NodeIndex, j: NodeIndex) {
        self.node_store.swap(i, j);
    }

    fn is_empty(&self) -> bool {
        <N as Swap<NodeIndex>>::is_empty(&self.node_store)
    }

    fn len(&self) -> NodeIndex {
        self.node_store.len()
    }
}

#[cfg(test)]
mod test {
    use crate::{dot, parser::DotGraph};

    #[test]
    fn swap() {
        let graph: DotGraph = dot!(digraph {
            edge [label = "test"]
            node [label = "test"]
            in [style=invis]
            A
            B
            in -> A:3
            in -> A
            in -> A
            2->3 [dir=back]
            3->2
            in -> A [label = "override"]
            A -> in
        })
        .unwrap();

        println!("{}", graph.dot_of(&graph.full_filter()));
        println!("{}", graph.debug_dot())
    }
}
