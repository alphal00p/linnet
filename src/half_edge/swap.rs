use crate::permutation::Permutation;

use super::{
    involution::{EdgeIndex, Hedge},
    nodestore::NodeStorageOps,
    HedgeGraph, NodeIndex,
};

pub trait Swap<Index> {
    fn swap(&mut self, i: Index, j: Index);

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
}
impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<Hedge> for HedgeGraph<E, V, H, N> {
    fn swap(&mut self, i: Hedge, j: Hedge) {
        // println!("Swapping {i:?} with {j:?}");
        self.hedge_data.swap(i.0, j.0);
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
}

impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<NodeIndex> for HedgeGraph<E, V, H, N> {
    fn swap(&mut self, i: NodeIndex, j: NodeIndex) {
        self.node_store.swap(i, j);
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
