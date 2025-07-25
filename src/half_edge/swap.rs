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

        for (i, j) in trans {
            self.swap(Index::from(i), Index::from(j));
        }
    }
}
impl<E, V, H, N: NodeStorageOps<NodeData = V>> Swap<Hedge> for HedgeGraph<E, V, H, N> {
    fn swap(&mut self, i: Hedge, j: Hedge) {
        self.hedge_data.swap(i.0, j.0);
        self.node_store.swap(i, j);
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
            edge [
                label = "test"
            ]
            node [
                label = "test"
            ]
            in [flow=source]
            out [flow=sink]
            A
            B
            in -> A
            in -> A
            in -> A
            in -> A [label = "override"]
            out -> A
        })
        .unwrap();

        println!("{}", graph.debug_dot())
    }
}
