use super::{
    child_pointer::{PCNode, ParentChildStore},
    parent_pointer::{PPNode, ParentId, ParentPointerStore},
    ForestNodeStore, ForestNodeStoreDown, RootId, TreeNodeId,
};

/// A node in the ChildVecStore. It contains a PPNode plus an ordered vector of children.
pub struct CVNode<V> {
    pub parent_pointer: PPNode<V>,
    pub children: Vec<TreeNodeId>,
}

/// The ChildVecStore itself.
pub struct ChildVecStore<V> {
    pub nodes: Vec<CVNode<V>>,
}

//
// Indexing implementations, similar to ParentChildStore/ParentPointerStore.
//
impl<V> std::ops::Index<&TreeNodeId> for ChildVecStore<V> {
    type Output = ParentId;
    fn index(&self, index: &TreeNodeId) -> &Self::Output {
        &self.nodes[index.0].parent_pointer.parent
    }
}

impl<V> std::ops::Index<TreeNodeId> for ChildVecStore<V> {
    type Output = V;
    fn index(&self, index: TreeNodeId) -> &Self::Output {
        &self.nodes[index.0].parent_pointer.data
    }
}

impl<V> std::ops::IndexMut<TreeNodeId> for ChildVecStore<V> {
    fn index_mut(&mut self, index: TreeNodeId) -> &mut Self::Output {
        &mut self.nodes[index.0].parent_pointer.data
    }
}

impl<V> FromIterator<CVNode<V>> for ChildVecStore<V> {
    fn from_iter<I: IntoIterator<Item = CVNode<V>>>(iter: I) -> Self {
        ChildVecStore {
            nodes: iter.into_iter().collect(),
        }
    }
}

//
// Implement the ForestNodeStore trait for ChildVecStore:
//
impl<V> ForestNodeStore for ChildVecStore<V> {
    type NodeData = V;
    type Store<T> = ChildVecStore<T>;

    fn iter_nodes(&self) -> impl Iterator<Item = (TreeNodeId, &Self::NodeData)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (TreeNodeId(i), &node.parent_pointer.data))
    }

    fn iter_node_id(&self) -> impl Iterator<Item = TreeNodeId> {
        (0..self.nodes.len()).map(TreeNodeId)
    }

    fn add_root(&mut self, data: Self::NodeData, root_id: RootId) -> TreeNodeId {
        let node_id = TreeNodeId(self.nodes.len());
        self.nodes.push(CVNode {
            parent_pointer: PPNode::root(data, root_id),
            children: Vec::new(),
        });
        node_id
    }

    fn add_child(&mut self, data: Self::NodeData, parent: TreeNodeId) -> TreeNodeId {
        let node_id = TreeNodeId(self.nodes.len());
        self.nodes.push(CVNode {
            parent_pointer: PPNode::child(data, parent),
            children: Vec::new(),
        });
        // Add this child to its parent's list.
        self.nodes[parent.0].children.push(node_id);
        node_id
    }

    fn map<F, U>(self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(Self::NodeData) -> U,
    {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| CVNode {
                parent_pointer: node.parent_pointer.map(&mut transform),
                children: node.children, // children remain the same id values.
            })
            .collect();
        ChildVecStore { nodes }
    }

    fn map_ref<F, U>(&self, mut transform: F) -> Self::Store<U>
    where
        F: FnMut(&Self::NodeData) -> U,
    {
        let nodes = self
            .nodes
            .iter()
            .map(|node| CVNode {
                parent_pointer: node.parent_pointer.map_ref(&mut transform),
                children: node.children.clone(),
            })
            .collect();
        ChildVecStore { nodes }
    }
}

impl<V> ForestNodeStoreDown for ChildVecStore<V> {
    fn iter_leaves(&self) -> impl Iterator<Item = TreeNodeId> {
        self.nodes.iter().enumerate().filter_map(|(i, node)| {
            if node.children.is_empty() {
                Some(TreeNodeId(i))
            } else {
                None
            }
        })
    }

    fn iter_children(&self, node_id: TreeNodeId) -> impl Iterator<Item = TreeNodeId> {
        // Clone the children vector (or borrow as iterator) – they are stored in order.
        self.nodes[node_id.0].children.clone().into_iter()
    }
}

//
// Conversions
//

// 1. Conversion from ParentPointerStore to ChildVecStore.
//    In a ParentPointerStore, the nodes only have a PPNode (with parent pointer and data).
//    To build the children vector, we iterate over all nodes and add each node as a child
//    of its parent if applicable.
impl<V> From<ParentPointerStore<V>> for ChildVecStore<V> {
    fn from(pp_store: ParentPointerStore<V>) -> Self {
        let n = pp_store.nodes.len();
        let mut some_pp = pp_store.map(|a| Some(a));
        let mut nodes = Vec::with_capacity(n);
        for pp in some_pp.nodes.iter_mut() {
            let data = pp.data.take().unwrap();
            let parent_pointer = PPNode {
                data,
                parent: pp.parent,
            };
            nodes.push(CVNode {
                parent_pointer,
                children: vec![],
            });
        }

        for (i, pp) in some_pp.nodes.iter().enumerate() {
            if let ParentId::Node(n) = pp.parent {
                nodes[n.0].children.push(TreeNodeId(i));
            }
        }

        ChildVecStore { nodes }
    }
}

// 2. Conversion from ParentChildStore to ChildVecStore.
//    The ParentChildStore stores child pointers via cyclic neighbor links. We can recover
//    an ordered children vector by using its provided methods.
impl<V> From<ParentChildStore<V>> for ChildVecStore<V> {
    fn from(pc_store: ParentChildStore<V>) -> Self {
        let n = pc_store.nodes.len();
        let mut some_pc = pc_store.map(|a| Some(a));
        let mut nodes = Vec::with_capacity(n);
        for nid in (0..n).map(TreeNodeId) {
            let data = some_pc[nid].take().unwrap();
            let parent_pointer = PPNode {
                data,
                parent: some_pc.nodes[nid.0].parent_pointer.parent,
            };

            let children = some_pc.iter_children(nid).collect();

            nodes.push(CVNode {
                parent_pointer,
                children,
            });
        }

        ChildVecStore { nodes }
    }
}

// 3. Conversion from ChildVecStore to ParentPointerStore.
//    This conversion simply drops the children vector and keeps only the PPNode.
impl<V> From<ChildVecStore<V>> for ParentPointerStore<V> {
    fn from(store: ChildVecStore<V>) -> Self {
        store
            .nodes
            .into_iter()
            .map(|cv_node| cv_node.parent_pointer)
            .collect()
    }
}

// 4. Conversion from ChildVecStore to ParentChildStore.
//    Here we build a PCNode from each CVNode. For each node, if its children vec is non-empty,
//    we set its 'child' field to the first child, and, for each child, we compute neighbor links
//    (cyclically) using the ordering in the vector.
impl<V> From<ChildVecStore<V>> for ParentChildStore<V> {
    fn from(store: ChildVecStore<V>) -> Self {
        let n = store.nodes.len();
        let mut some_store = store.map(|a| Some(a));
        let mut nodes = Vec::with_capacity(n);

        for nid in (0..n).map(TreeNodeId) {
            let data = some_store[nid].take().unwrap();
            let parent_pointer = PPNode {
                data,
                parent: some_store.nodes[nid.0].parent_pointer.parent,
            };

            let child = some_store.nodes[nid.0].children.iter().cloned().next();

            nodes.push(PCNode {
                parent_pointer,
                child,
                neighbor_left: nid,
                neighbor_right: nid,
            });
        }

        for node in some_store.nodes.iter() {
            let len = node.children.len();
            for (j, &child) in node.children.iter().enumerate() {
                let left = node.children[(j + len - 1) % len];
                let right = node.children[(j + 1) % len];
                nodes[child.0].neighbor_left = left;
                nodes[child.0].neighbor_right = right;
            }
        }

        ParentChildStore { nodes }
    }
}
