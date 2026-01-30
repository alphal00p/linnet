use std::cell::RefCell;
use std::collections::BTreeMap;

use linnet::half_edge::involution::{EdgeData, EdgeIndex, Flow, Hedge, HedgePair, Orientation};
use linnet::half_edge::subgraph::{Inclusion, ModifySubSet, SuBitGraph, SubSetLike, SubSetOps};
use linnet::half_edge::tree::SimpleTraversalTree;
use linnet::half_edge::{HedgeGraphError, NodeIndex};
use linnet::parser::{DotEdgeData, DotGraph, DotHedgeData, DotVertexData};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple, PyType};

#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PyHedge {
    hedge: Hedge,
}

#[pymethods]
impl PyHedge {
    #[new]
    fn new(value: usize) -> Self {
        Self {
            hedge: Hedge(value),
        }
    }

    #[getter]
    fn value(&self) -> usize {
        self.hedge.0
    }

    fn __int__(&self) -> usize {
        self.hedge.0
    }

    fn __repr__(&self) -> String {
        format!("Hedge({})", self.hedge.0)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PyNodeIndex {
    node: NodeIndex,
}

#[pymethods]
impl PyNodeIndex {
    #[new]
    fn new(value: usize) -> Self {
        Self {
            node: NodeIndex(value),
        }
    }

    #[getter]
    fn value(&self) -> usize {
        self.node.0
    }

    fn __int__(&self) -> usize {
        self.node.0
    }

    fn __repr__(&self) -> String {
        format!("NodeIndex({})", self.node.0)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PyEdgeIndex {
    edge: EdgeIndex,
}

#[pymethods]
impl PyEdgeIndex {
    #[new]
    fn new(value: usize) -> Self {
        Self {
            edge: EdgeIndex::from(value),
        }
    }

    #[getter]
    fn value(&self) -> usize {
        self.edge.0
    }

    fn __int__(&self) -> usize {
        self.edge.0
    }

    fn __repr__(&self) -> String {
        format!("EdgeIndex({})", self.edge.0)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PyFlow {
    flow: Flow,
}

#[pymethods]
impl PyFlow {
    #[staticmethod]
    fn source() -> Self {
        Self { flow: Flow::Source }
    }

    #[staticmethod]
    fn sink() -> Self {
        Self { flow: Flow::Sink }
    }

    fn __repr__(&self) -> String {
        match self.flow {
            Flow::Source => "Flow.Source".to_string(),
            Flow::Sink => "Flow.Sink".to_string(),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PyOrientation {
    orientation: Orientation,
}

#[pymethods]
impl PyOrientation {
    #[staticmethod]
    fn default() -> Self {
        Self {
            orientation: Orientation::Default,
        }
    }

    #[staticmethod]
    fn reversed() -> Self {
        Self {
            orientation: Orientation::Reversed,
        }
    }

    #[staticmethod]
    fn undirected() -> Self {
        Self {
            orientation: Orientation::Undirected,
        }
    }

    fn __repr__(&self) -> String {
        match self.orientation {
            Orientation::Default => "Orientation.Default".to_string(),
            Orientation::Reversed => "Orientation.Reversed".to_string(),
            Orientation::Undirected => "Orientation.Undirected".to_string(),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PyDotVertexData {
    data: DotVertexData,
}

#[pymethods]
impl PyDotVertexData {
    #[new]
    fn new(
        name: Option<String>,
        index: Option<usize>,
        statements: Option<BTreeMap<String, String>>,
    ) -> Self {
        Self {
            data: DotVertexData {
                name,
                index: index.map(NodeIndex),
                statements: statements.unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.data.name.clone()
    }

    #[getter]
    fn index(&self) -> Option<PyNodeIndex> {
        self.data.index.map(|i| PyNodeIndex { node: i })
    }

    #[getter]
    fn statements<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.data.statements {
            let _ = dict.set_item(k, v);
        }
        dict
    }

    fn add_statement(&mut self, key: String, value: String) {
        self.data.add_statement(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "DotVertexData(name={:?}, index={:?}, statements={})",
            self.data.name,
            self.data.index.map(|i| i.0),
            self.data.statements.len()
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PyDotHedgeData {
    data: DotHedgeData,
}

#[pymethods]
impl PyDotHedgeData {
    #[new]
    fn new(
        statement: Option<String>,
        id: Option<usize>,
        port_label: Option<String>,
        compasspt: Option<String>,
    ) -> Self {
        let compasspt = compasspt.and_then(|value| parse_compasspt(&value));
        Self {
            data: DotHedgeData {
                statement,
                id: id.map(Hedge),
                port_label,
                compasspt,
            },
        }
    }

    #[getter]
    fn statement(&self) -> Option<String> {
        self.data.statement.clone()
    }

    #[getter]
    fn id(&self) -> Option<PyHedge> {
        self.data.id.map(|h| PyHedge { hedge: h })
    }

    #[getter]
    fn port_label(&self) -> Option<String> {
        self.data.port_label.clone()
    }

    #[getter]
    fn compasspt(&self) -> Option<String> {
        self.data.compasspt.as_ref().map(|c| format!("{c:?}"))
    }

    fn __repr__(&self) -> String {
        format!(
            "DotHedgeData(statement={:?}, id={:?}, port_label={:?}, compasspt={:?})",
            self.data.statement,
            self.data.id.map(|h| h.0),
            self.data.port_label,
            self.data.compasspt.as_ref().map(|c| format!("{c:?}"))
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PyDotEdgeData {
    data: DotEdgeData,
}

#[pymethods]
impl PyDotEdgeData {
    #[new]
    fn new(
        statements: Option<BTreeMap<String, String>>,
        local_statements: Option<BTreeMap<String, String>>,
        edge_id: Option<usize>,
    ) -> Self {
        Self {
            data: DotEdgeData {
                statements: statements.unwrap_or_default(),
                local_statements: local_statements.unwrap_or_default(),
                edge_id: edge_id.map(EdgeIndex::from),
            },
        }
    }

    #[getter]
    fn statements<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.data.statements {
            let _ = dict.set_item(k, v);
        }
        dict
    }

    #[getter]
    fn local_statements<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        for (k, v) in &self.data.local_statements {
            let _ = dict.set_item(k, v);
        }
        dict
    }

    #[getter]
    fn edge_id(&self) -> Option<PyEdgeIndex> {
        self.data.edge_id.map(|i| PyEdgeIndex { edge: i })
    }

    fn add_statement(&mut self, key: String, value: String) {
        self.data.add_statement(key, value);
    }

    fn __repr__(&self) -> String {
        format!(
            "DotEdgeData(statements={}, local_statements={}, edge_id={:?})",
            self.data.statements.len(),
            self.data.local_statements.len(),
            self.data.edge_id.map(|i| i.0)
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PyEdgeData {
    data: EdgeData<DotEdgeData>,
}

#[pymethods]
impl PyEdgeData {
    #[new]
    fn new(orientation: &Bound<'_, PyAny>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let orientation = extract_orientation(orientation)?;
        let data = extract_dot_edge_data(data)?;
        Ok(Self {
            data: EdgeData { orientation, data },
        })
    }

    #[getter]
    fn orientation(&self) -> PyOrientation {
        PyOrientation {
            orientation: self.data.orientation,
        }
    }

    #[getter]
    fn data(&self) -> PyDotEdgeData {
        PyDotEdgeData {
            data: self.data.data.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("EdgeData(orientation={:?})", self.data.orientation)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PyHedgePair {
    pair: HedgePair,
}

#[pymethods]
impl PyHedgePair {
    #[getter]
    fn kind(&self) -> String {
        match self.pair {
            HedgePair::Paired { .. } => "paired",
            HedgePair::Unpaired { .. } => "unpaired",
            HedgePair::Split { .. } => "split",
        }
        .to_string()
    }

    #[getter]
    fn source(&self) -> Option<PyHedge> {
        match self.pair {
            HedgePair::Paired { source, .. } => Some(PyHedge { hedge: source }),
            HedgePair::Split { source, .. } => Some(PyHedge { hedge: source }),
            HedgePair::Unpaired { .. } => None,
        }
    }

    #[getter]
    fn sink(&self) -> Option<PyHedge> {
        match self.pair {
            HedgePair::Paired { sink, .. } => Some(PyHedge { hedge: sink }),
            HedgePair::Split { sink, .. } => Some(PyHedge { hedge: sink }),
            HedgePair::Unpaired { .. } => None,
        }
    }

    #[getter]
    fn hedge(&self) -> Option<PyHedge> {
        match self.pair {
            HedgePair::Unpaired { hedge, .. } => Some(PyHedge { hedge }),
            _ => None,
        }
    }

    #[getter]
    fn flow(&self) -> Option<PyFlow> {
        match self.pair {
            HedgePair::Unpaired { flow, .. } => Some(PyFlow { flow }),
            _ => None,
        }
    }

    #[getter]
    fn split(&self) -> Option<PyFlow> {
        match self.pair {
            HedgePair::Split { split, .. } => Some(PyFlow { flow: split }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("HedgePair({})", self.kind())
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct PySubgraph {
    subgraph: SuBitGraph,
}

#[pymethods]
impl PySubgraph {
    #[classmethod]
    fn empty(_cls: &Bound<'_, PyType>, size: usize) -> Self {
        Self {
            subgraph: SuBitGraph::empty(size),
        }
    }

    #[classmethod]
    fn full(_cls: &Bound<'_, PyType>, size: usize) -> Self {
        Self {
            subgraph: SuBitGraph::full(size),
        }
    }

    #[classmethod]
    fn from_hedges(
        _cls: &Bound<'_, PyType>,
        size: usize,
        hedges: Vec<Py<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let mut subgraph = SuBitGraph::empty(size);
        for h in hedges {
            subgraph.add(extract_hedge(&h.bind(py))?);
        }
        Ok(Self { subgraph })
    }

    fn to_hedges(&self) -> Vec<PyHedge> {
        self.subgraph
            .included_iter()
            .map(|h| PyHedge { hedge: h })
            .collect()
    }

    fn size(&self) -> usize {
        self.subgraph.size()
    }

    fn n_included(&self) -> usize {
        self.subgraph.n_included()
    }

    fn includes(&self, hedge: &Bound<'_, PyAny>) -> PyResult<bool> {
        let hedge = extract_hedge(hedge)?;
        Ok(self.subgraph.includes(&hedge))
    }

    fn union(&self, other: &PySubgraph) -> Self {
        Self {
            subgraph: self.subgraph.union(&other.subgraph),
        }
    }

    fn intersection(&self, other: &PySubgraph) -> Self {
        Self {
            subgraph: self.subgraph.intersection(&other.subgraph),
        }
    }

    fn sym_diff(&self, other: &PySubgraph) -> Self {
        Self {
            subgraph: self.subgraph.sym_diff(&other.subgraph),
        }
    }

    fn subtract(&self, other: &PySubgraph) -> Self {
        Self {
            subgraph: self.subgraph.subtract(&other.subgraph),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Subgraph(size={}, included={})",
            self.size(),
            self.n_included()
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
struct PyTraversalTree {
    tree: SimpleTraversalTree,
}

#[pymethods]
impl PyTraversalTree {
    fn tree_subgraph(&self) -> PySubgraph {
        PySubgraph {
            subgraph: self.tree.tree_subgraph.clone(),
        }
    }

    fn node_order(&self) -> Vec<PyNodeIndex> {
        self.tree
            .node_order()
            .into_iter()
            .map(|n| PyNodeIndex { node: n })
            .collect()
    }

    fn covers(&self, subgraph: &PySubgraph) -> PySubgraph {
        PySubgraph {
            subgraph: self.tree.covers(&subgraph.subgraph),
        }
    }

    fn iter_hedges(&self) -> Vec<(PyHedge, String, Option<PyHedge>)> {
        self.tree
            .iter_hedges()
            .map(|(h, root, root_hedge)| {
                let kind = match root {
                    linnet::half_edge::tree::TTRoot::Root => "root",
                    linnet::half_edge::tree::TTRoot::Child(_) => "child",
                    linnet::half_edge::tree::TTRoot::None => "none",
                }
                .to_string();
                (
                    PyHedge { hedge: h },
                    kind,
                    root_hedge.map(|rh| PyHedge { hedge: rh }),
                )
            })
            .collect()
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
struct PyDotGraph {
    graph: DotGraph,
}

#[pymethods]
impl PyDotGraph {
    #[classmethod]
    fn from_string(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let graph = DotGraph::from_string(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { graph })
    }

    #[classmethod]
    fn from_file(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let graph = DotGraph::from_file(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { graph })
    }

    fn debug_dot(&self) -> String {
        self.graph.debug_dot()
    }

    fn dot_of(&self, subgraph: &PySubgraph) -> String {
        self.graph.dot_of(&subgraph.subgraph)
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Ok(h) = key.extract::<PyRef<PyHedge>>() {
            let data = self.graph.graph[h.hedge].clone();
            let obj = Py::new(py, PyDotHedgeData { data })?;
            return Ok(obj.into_any());
        }
        if let Ok(n) = key.extract::<PyRef<PyNodeIndex>>() {
            let data = self.graph.graph[n.node].clone();
            let obj = Py::new(py, PyDotVertexData { data })?;
            return Ok(obj.into_any());
        }
        if let Ok(e) = key.extract::<PyRef<PyEdgeIndex>>() {
            let data = self.graph.graph[e.edge].clone();
            let obj = Py::new(py, PyDotEdgeData { data })?;
            return Ok(obj.into_any());
        }
        Err(PyValueError::new_err(
            "expected Hedge, NodeIndex, or EdgeIndex",
        ))
    }

    fn n_nodes(&self) -> usize {
        self.graph.n_nodes()
    }

    fn n_edges(&self) -> usize {
        self.graph.n_edges()
    }

    fn n_hedges(&self) -> usize {
        self.graph.n_hedges()
    }

    fn n_externals(&self) -> usize {
        self.graph.n_externals()
    }

    fn n_internals(&self) -> usize {
        self.graph.n_internals()
    }

    fn full_filter(&self) -> PySubgraph {
        PySubgraph {
            subgraph: self.graph.full_filter(),
        }
    }

    fn empty_subgraph(&self) -> PySubgraph {
        PySubgraph {
            subgraph: self.graph.empty_subgraph(),
        }
    }

    fn iter_edges_of(&self, subgraph: &PySubgraph) -> Vec<(PyHedgePair, PyEdgeIndex, PyEdgeData)> {
        self.graph
            .iter_edges_of(&subgraph.subgraph)
            .map(|(pair, eid, data)| {
                let owned = EdgeData {
                    orientation: data.orientation,
                    data: data.data.clone(),
                };
                (
                    PyHedgePair { pair },
                    PyEdgeIndex { edge: eid },
                    PyEdgeData { data: owned },
                )
            })
            .collect()
    }

    fn iter_nodes_of(
        &self,
        subgraph: &PySubgraph,
    ) -> Vec<(PyNodeIndex, Vec<PyHedge>, PyDotVertexData)> {
        self.graph
            .iter_nodes_of(&subgraph.subgraph)
            .map(|(node, neighbors, data)| {
                let hedges = neighbors.map(|h| PyHedge { hedge: h }).collect();
                (
                    PyNodeIndex { node },
                    hedges,
                    PyDotVertexData { data: data.clone() },
                )
            })
            .collect()
    }

    fn connected_components(&self, subgraph: &PySubgraph) -> Vec<PySubgraph> {
        self.graph
            .connected_components(&subgraph.subgraph)
            .into_iter()
            .map(|sg| PySubgraph { subgraph: sg })
            .collect()
    }

    fn count_connected_components(&self, subgraph: &PySubgraph) -> usize {
        self.graph.count_connected_components(&subgraph.subgraph)
    }

    fn is_connected(&self, subgraph: &PySubgraph) -> bool {
        self.graph.is_connected(&subgraph.subgraph)
    }

    fn depth_first_traverse(
        &self,
        subgraph: &PySubgraph,
        root_node: &Bound<'_, PyAny>,
        include_hedge: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyTraversalTree> {
        let root = extract_node_index(root_node)?;
        let include = match include_hedge {
            Some(h) => Some(extract_hedge(&h)?),
            None => None,
        };
        let tree = SimpleTraversalTree::depth_first_traverse(
            &self.graph.graph,
            &subgraph.subgraph,
            &root,
            include,
        )
        .map_err(to_py_err)?;
        Ok(PyTraversalTree { tree })
    }

    fn breadth_first_traverse(
        &self,
        subgraph: &PySubgraph,
        root_node: &Bound<'_, PyAny>,
        include_hedge: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyTraversalTree> {
        let root = extract_node_index(root_node)?;
        let include = match include_hedge {
            Some(h) => Some(extract_hedge(&h)?),
            None => None,
        };
        let tree = SimpleTraversalTree::breadth_first_traverse(
            &self.graph.graph,
            &subgraph.subgraph,
            &root,
            include,
        )
        .map_err(to_py_err)?;
        Ok(PyTraversalTree { tree })
    }

    fn join(
        &self,
        other: &PyDotGraph,
        matching_fn: Py<PyAny>,
        merge_fn: Py<PyAny>,
    ) -> PyResult<Self> {
        let error: RefCell<Option<PyErr>> = RefCell::new(None);
        let left = self.graph.graph.clone();
        let right = other.graph.graph.clone();
        let matching = matching_fn;
        let merging = merge_fn;

        let result = left.join(
            right,
            |f1, d1, f2, d2| {
                if error.borrow().is_some() {
                    return false;
                }
                Python::attach(|py| {
                    let py_f1 = PyFlow { flow: f1 };
                    let py_f2 = PyFlow { flow: f2 };
                    let py_d1 = PyEdgeData {
                        data: EdgeData {
                            orientation: d1.orientation,
                            data: d1.data.clone(),
                        },
                    };
                    let py_d2 = PyEdgeData {
                        data: EdgeData {
                            orientation: d2.orientation,
                            data: d2.data.clone(),
                        },
                    };

                    match matching.bind(py).call1((py_f1, py_d1, py_f2, py_d2)) {
                        Ok(val) => match val.extract::<bool>() {
                            Ok(b) => b,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                false
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            false
                        }
                    }
                })
            },
            |f1, d1, f2, d2| {
                if error.borrow().is_some() {
                    return (
                        Flow::Source,
                        EdgeData {
                            orientation: Orientation::Undirected,
                            data: DotEdgeData::empty(),
                        },
                    );
                }
                Python::attach(|py| {
                    let py_f1 = PyFlow { flow: f1 };
                    let py_f2 = PyFlow { flow: f2 };
                    let py_d1 = PyEdgeData { data: d1 };
                    let py_d2 = PyEdgeData { data: d2 };

                    match merging.bind(py).call1((py_f1, py_d1, py_f2, py_d2)) {
                        Ok(val) => match extract_merge_result(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                (
                                    Flow::Source,
                                    EdgeData {
                                        orientation: Orientation::Undirected,
                                        data: DotEdgeData::empty(),
                                    },
                                )
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            (
                                Flow::Source,
                                EdgeData {
                                    orientation: Orientation::Undirected,
                                    data: DotEdgeData::empty(),
                                },
                            )
                        }
                    }
                })
            },
        );

        if let Some(err) = error.into_inner() {
            return Err(err);
        }

        let graph = result.map_err(to_py_err)?;
        Ok(Self {
            graph: DotGraph {
                graph,
                global_data: self.graph.global_data.clone(),
            },
        })
    }

    fn join_mut(
        &mut self,
        other: &PyDotGraph,
        matching_fn: Py<PyAny>,
        merge_fn: Py<PyAny>,
    ) -> PyResult<()> {
        let error: RefCell<Option<PyErr>> = RefCell::new(None);
        let other_graph = other.graph.graph.clone();
        let matching = matching_fn;
        let merging = merge_fn;

        let result = self.graph.graph.join_mut(
            other_graph,
            |f1, d1, f2, d2| {
                if error.borrow().is_some() {
                    return false;
                }
                Python::attach(|py| {
                    let py_f1 = PyFlow { flow: f1 };
                    let py_f2 = PyFlow { flow: f2 };
                    let py_d1 = PyEdgeData {
                        data: EdgeData {
                            orientation: d1.orientation,
                            data: d1.data.clone(),
                        },
                    };
                    let py_d2 = PyEdgeData {
                        data: EdgeData {
                            orientation: d2.orientation,
                            data: d2.data.clone(),
                        },
                    };

                    match matching.bind(py).call1((py_f1, py_d1, py_f2, py_d2)) {
                        Ok(val) => match val.extract::<bool>() {
                            Ok(b) => b,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                false
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            false
                        }
                    }
                })
            },
            |f1, d1, f2, d2| {
                if error.borrow().is_some() {
                    return (
                        Flow::Source,
                        EdgeData {
                            orientation: Orientation::Undirected,
                            data: DotEdgeData::empty(),
                        },
                    );
                }
                Python::attach(|py| {
                    let py_f1 = PyFlow { flow: f1 };
                    let py_f2 = PyFlow { flow: f2 };
                    let py_d1 = PyEdgeData { data: d1 };
                    let py_d2 = PyEdgeData { data: d2 };

                    match merging.bind(py).call1((py_f1, py_d1, py_f2, py_d2)) {
                        Ok(val) => match extract_merge_result(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                (
                                    Flow::Source,
                                    EdgeData {
                                        orientation: Orientation::Undirected,
                                        data: DotEdgeData::empty(),
                                    },
                                )
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            (
                                Flow::Source,
                                EdgeData {
                                    orientation: Orientation::Undirected,
                                    data: DotEdgeData::empty(),
                                },
                            )
                        }
                    }
                })
            },
        );

        if let Some(err) = error.into_inner() {
            return Err(err);
        }

        result.map_err(to_py_err)
    }

    fn extract(
        &mut self,
        subgraph: &PySubgraph,
        split_edge_fn: Py<PyAny>,
        internal_data: Py<PyAny>,
        split_node: Py<PyAny>,
        owned_node: Py<PyAny>,
    ) -> PyResult<Self> {
        let error: RefCell<Option<PyErr>> = RefCell::new(None);
        let split_edge = split_edge_fn;
        let internal = internal_data;
        let split_node_fn = split_node;
        let owned_node_fn = owned_node;

        let extracted = self.graph.graph.extract(
            &subgraph.subgraph,
            |edge_ref| {
                if error.borrow().is_some() {
                    return EdgeData {
                        orientation: Orientation::Undirected,
                        data: DotEdgeData::empty(),
                    };
                }
                Python::attach(|py| {
                    let py_edge = PyEdgeData {
                        data: EdgeData {
                            orientation: edge_ref.orientation,
                            data: edge_ref.data.clone(),
                        },
                    };
                    match split_edge.bind(py).call1((py_edge,)) {
                        Ok(val) => match extract_edge_data(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                EdgeData {
                                    orientation: Orientation::Undirected,
                                    data: DotEdgeData::empty(),
                                }
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            EdgeData {
                                orientation: Orientation::Undirected,
                                data: DotEdgeData::empty(),
                            }
                        }
                    }
                })
            },
            |edge_owned| {
                if error.borrow().is_some() {
                    return EdgeData {
                        orientation: Orientation::Undirected,
                        data: DotEdgeData::empty(),
                    };
                }
                Python::attach(|py| {
                    let py_edge = PyEdgeData { data: edge_owned };
                    match internal.bind(py).call1((py_edge,)) {
                        Ok(val) => match extract_edge_data(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                EdgeData {
                                    orientation: Orientation::Undirected,
                                    data: DotEdgeData::empty(),
                                }
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            EdgeData {
                                orientation: Orientation::Undirected,
                                data: DotEdgeData::empty(),
                            }
                        }
                    }
                })
            },
            |node_ref| {
                if error.borrow().is_some() {
                    return DotVertexData::empty();
                }
                Python::attach(|py| {
                    let py_node = PyDotVertexData {
                        data: node_ref.clone(),
                    };
                    match split_node_fn.bind(py).call1((py_node,)) {
                        Ok(val) => match extract_dot_vertex_data(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                DotVertexData::empty()
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            DotVertexData::empty()
                        }
                    }
                })
            },
            |node_owned| {
                if error.borrow().is_some() {
                    return DotVertexData::empty();
                }
                Python::attach(|py| {
                    let py_node = PyDotVertexData { data: node_owned };
                    match owned_node_fn.bind(py).call1((py_node,)) {
                        Ok(val) => match extract_dot_vertex_data(&val) {
                            Ok(out) => out,
                            Err(e) => {
                                *error.borrow_mut() = Some(e);
                                DotVertexData::empty()
                            }
                        },
                        Err(e) => {
                            *error.borrow_mut() = Some(e);
                            DotVertexData::empty()
                        }
                    }
                })
            },
        );

        if let Some(err) = error.into_inner() {
            return Err(err);
        }

        Ok(Self {
            graph: DotGraph {
                graph: extracted,
                global_data: self.graph.global_data.clone(),
            },
        })
    }
}

#[pymodule]
fn linnet_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHedge>()?;
    m.add_class::<PyNodeIndex>()?;
    m.add_class::<PyEdgeIndex>()?;
    m.add_class::<PyFlow>()?;
    m.add_class::<PyOrientation>()?;
    m.add_class::<PyDotVertexData>()?;
    m.add_class::<PyDotHedgeData>()?;
    m.add_class::<PyDotEdgeData>()?;
    m.add_class::<PyEdgeData>()?;
    m.add_class::<PyHedgePair>()?;
    m.add_class::<PySubgraph>()?;
    m.add_class::<PyTraversalTree>()?;
    m.add_class::<PyDotGraph>()?;
    Ok(())
}

fn extract_hedge(obj: &Bound<'_, PyAny>) -> PyResult<Hedge> {
    if let Ok(h) = obj.extract::<PyRef<PyHedge>>() {
        Ok(h.hedge)
    } else {
        Ok(Hedge(obj.extract::<usize>()?))
    }
}

fn extract_node_index(obj: &Bound<'_, PyAny>) -> PyResult<NodeIndex> {
    if let Ok(n) = obj.extract::<PyRef<PyNodeIndex>>() {
        Ok(n.node)
    } else {
        Ok(NodeIndex(obj.extract::<usize>()?))
    }
}

fn extract_flow(obj: &Bound<'_, PyAny>) -> PyResult<Flow> {
    if let Ok(f) = obj.extract::<PyRef<PyFlow>>() {
        Ok(f.flow)
    } else if let Ok(s) = obj.extract::<&str>() {
        match s.to_ascii_lowercase().as_str() {
            "source" => Ok(Flow::Source),
            "sink" => Ok(Flow::Sink),
            _ => Err(PyValueError::new_err("invalid flow")),
        }
    } else {
        Err(PyValueError::new_err("invalid flow"))
    }
}

fn extract_orientation(obj: &Bound<'_, PyAny>) -> PyResult<Orientation> {
    if let Ok(o) = obj.extract::<PyRef<PyOrientation>>() {
        Ok(o.orientation)
    } else if let Ok(s) = obj.extract::<&str>() {
        match s.to_ascii_lowercase().as_str() {
            "default" => Ok(Orientation::Default),
            "reversed" => Ok(Orientation::Reversed),
            "undirected" => Ok(Orientation::Undirected),
            _ => Err(PyValueError::new_err("invalid orientation")),
        }
    } else {
        Err(PyValueError::new_err("invalid orientation"))
    }
}

fn extract_dot_edge_data(obj: &Bound<'_, PyAny>) -> PyResult<DotEdgeData> {
    Ok(obj.extract::<PyRef<PyDotEdgeData>>()?.data.clone())
}

fn extract_dot_vertex_data(obj: &Bound<'_, PyAny>) -> PyResult<DotVertexData> {
    Ok(obj.extract::<PyRef<PyDotVertexData>>()?.data.clone())
}

fn extract_edge_data(obj: &Bound<'_, PyAny>) -> PyResult<EdgeData<DotEdgeData>> {
    if let Ok(ed) = obj.extract::<PyRef<PyEdgeData>>() {
        Ok(ed.data.clone())
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        if tuple.len() != 2 {
            return Err(PyValueError::new_err("expected (orientation, data)"));
        }
        let item0 = tuple.get_item(0)?;
        let item1 = tuple.get_item(1)?;
        let orientation = extract_orientation(&item0)?;
        let data = extract_dot_edge_data(&item1)?;
        Ok(EdgeData { orientation, data })
    } else {
        Err(PyValueError::new_err("invalid edge data"))
    }
}

fn extract_merge_result(obj: &Bound<'_, PyAny>) -> PyResult<(Flow, EdgeData<DotEdgeData>)> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        if tuple.len() != 2 {
            return Err(PyValueError::new_err("expected (flow, edge_data)"));
        }
        let item0 = tuple.get_item(0)?;
        let item1 = tuple.get_item(1)?;
        let flow = extract_flow(&item0)?;
        let data = extract_edge_data(&item1)?;
        Ok((flow, data))
    } else {
        Err(PyValueError::new_err("expected (flow, edge_data)"))
    }
}

fn to_py_err(err: HedgeGraphError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_compasspt(value: &str) -> Option<dot_parser::ast::CompassPt> {
    use dot_parser::ast::CompassPt;
    match value.to_ascii_lowercase().as_str() {
        "n" => Some(CompassPt::N),
        "ne" => Some(CompassPt::NE),
        "e" => Some(CompassPt::E),
        "se" => Some(CompassPt::SE),
        "s" => Some(CompassPt::S),
        "sw" => Some(CompassPt::SW),
        "w" => Some(CompassPt::W),
        "nw" => Some(CompassPt::NW),
        "c" => Some(CompassPt::C),
        "_" => Some(CompassPt::Underscore),
        _ => None,
    }
}
