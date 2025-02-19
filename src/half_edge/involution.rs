use std::{
    fmt::Display,
    ops::{Index, IndexMut, Mul, Neg},
};

use crate::num_traits::RefZero;
use derive_more::{From, Into};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{nodestorage::NodeStorageOps, subgraph::SubGraph, GVEdgeAttrs, HedgeGraph, NodeIndex};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hedge(pub usize);

impl Hedge {
    pub fn to_edge_id<E>(self, involution: &Involution<E>) -> HedgePair {
        HedgePair::from_half_edge(self, involution)
    }

    pub fn to_edge_id_with_subgraph<E, S: SubGraph>(
        self,
        involution: &Involution<E>,
        subgraph: &S,
    ) -> Option<HedgePair> {
        HedgePair::from_half_edge_with_subgraph(self, involution, subgraph)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum HedgePair {
    Unpaired {
        hedge: Hedge,
        flow: Flow,
    },
    Paired {
        source: Hedge,
        sink: Hedge,
    },
    Split {
        source: Hedge,
        sink: Hedge,
        split: Flow,
    },
}

impl HedgePair {
    pub fn is_unpaired(&self) -> bool {
        matches!(self, HedgePair::Unpaired { .. })
    }
    pub fn any_hedge(&self) -> Hedge {
        match self {
            HedgePair::Unpaired { hedge, .. } => *hedge,
            HedgePair::Paired { source, .. } => *source,
            HedgePair::Split {
                source,
                sink,
                split,
            } => {
                if *split == Flow::Source {
                    *source
                } else {
                    *sink
                }
            }
        }
    }

    pub fn from_half_edge<E>(hedge: Hedge, involution: &Involution<E>) -> Self {
        match involution.hedge_data(hedge) {
            InvolutiveMapping::Source { sink_idx, .. } => HedgePair::Paired {
                source: hedge,
                sink: *sink_idx,
            },
            InvolutiveMapping::Sink { source_idx } => HedgePair::Paired {
                source: *source_idx,
                sink: hedge,
            },
            InvolutiveMapping::Identity { underlying, .. } => HedgePair::Unpaired {
                hedge,
                flow: *underlying,
            },
        }
    }

    pub fn from_source<E>(hedge: Hedge, involution: &Involution<E>) -> Option<Self> {
        match involution.hedge_data(hedge) {
            InvolutiveMapping::Source { sink_idx, .. } => Some(HedgePair::Paired {
                source: hedge,
                sink: *sink_idx,
            }),
            InvolutiveMapping::Identity { underlying, .. } => Some(HedgePair::Unpaired {
                hedge,
                flow: *underlying,
            }),
            _ => None,
        }
    }

    pub fn from_source_with_subgraph<E, S: SubGraph>(
        hedge: Hedge,
        involution: &Involution<E>,
        subgraph: &S,
    ) -> Option<Self> {
        if subgraph.includes(&hedge) {
            match involution.hedge_data(hedge) {
                InvolutiveMapping::Source { sink_idx, .. } => {
                    if subgraph.includes(sink_idx) {
                        Some(HedgePair::Paired {
                            source: hedge,
                            sink: *sink_idx,
                        })
                    } else {
                        Some(HedgePair::Split {
                            source: hedge,
                            sink: *sink_idx,
                            split: Flow::Source,
                        })
                    }
                }
                InvolutiveMapping::Sink { source_idx } => {
                    if subgraph.includes(source_idx) {
                        None
                    } else {
                        Some(HedgePair::Split {
                            source: *source_idx,
                            sink: hedge,
                            split: Flow::Sink,
                        })
                    }
                }
                InvolutiveMapping::Identity { underlying, .. } => Some(HedgePair::Unpaired {
                    hedge,
                    flow: *underlying,
                }),
            }
        } else {
            None
        }
    }

    // uses the present color for split edge background
    pub fn fill_color(&self, attr: GVEdgeAttrs) -> GVEdgeAttrs {
        match self {
            HedgePair::Unpaired { flow, .. } => {
                if attr.color.is_some() {
                    attr
                } else {
                    GVEdgeAttrs {
                        color: Some(format!("\"{}\"", flow.color())),
                        label: attr.label,
                        other: attr.other,
                    }
                }
            }
            HedgePair::Paired { .. } => {
                if attr.color.is_some() {
                    attr
                } else {
                    let color = format!("\"{}:{};0.5\"", Flow::Source.color(), Flow::Sink.color());
                    GVEdgeAttrs {
                        color: Some(color),
                        label: attr.label,
                        other: attr.other,
                    }
                }
            }
            HedgePair::Split { split, .. } => {
                let background = if let Some(color) = attr.color {
                    color
                } else {
                    "gray75".to_owned()
                };

                let color = match split {
                    Flow::Source => format!("{}:{};0.5", split.color(), background),
                    Flow::Sink => format!("{}:{};0.5", background, split.color()),
                };
                GVEdgeAttrs {
                    color: Some(color),
                    label: attr.label,
                    other: attr.other,
                }
            }
        }
    }

    pub fn dot<E, V, N: NodeStorageOps<NodeData = V>>(
        &self,
        graph: &HedgeGraph<E, V, N>,
        orientation: Orientation,
        attr: GVEdgeAttrs,
    ) -> String {
        let attr = self.fill_color(attr);
        match self {
            HedgePair::Unpaired { hedge, flow } => InvolutiveMapping::<()>::identity_dot(
                *hedge,
                graph.node_id(*hedge),
                Some(&attr),
                orientation,
                *flow,
            ),
            HedgePair::Paired { source, sink } => InvolutiveMapping::<()>::pair_dot(
                graph.node_id(*source),
                graph.node_id(*sink),
                Some(&attr),
                orientation,
            ),
            HedgePair::Split { source, sink, .. } => InvolutiveMapping::<()>::pair_dot(
                graph.node_id(*source),
                graph.node_id(*sink),
                Some(&attr),
                orientation,
            ),
        }
    }

    pub fn with_subgraph<S: SubGraph>(self, subgraph: &S) -> Option<Self> {
        match self {
            HedgePair::Paired { source, sink } => {
                match (subgraph.includes(&source), subgraph.includes(&sink)) {
                    (true, true) => Some(HedgePair::Paired { source, sink }),
                    (true, false) => Some(HedgePair::Split {
                        source,
                        sink,
                        split: Flow::Source,
                    }),
                    (false, true) => Some(HedgePair::Split {
                        source,
                        sink,
                        split: Flow::Sink,
                    }),
                    (false, false) => None,
                }
            }
            HedgePair::Split { source, sink, .. } => {
                match (subgraph.includes(&source), subgraph.includes(&sink)) {
                    (true, true) => Some(HedgePair::Paired { source, sink }),
                    (true, false) => Some(HedgePair::Split {
                        source,
                        sink,
                        split: Flow::Source,
                    }),
                    (false, true) => Some(HedgePair::Split {
                        source,
                        sink,
                        split: Flow::Sink,
                    }),
                    (false, false) => None,
                }
            }
            HedgePair::Unpaired { hedge, flow } => {
                if subgraph.includes(&hedge) {
                    Some(HedgePair::Unpaired { hedge, flow })
                } else {
                    None
                }
            }
        }
    }

    pub fn from_half_edge_with_subgraph<E, S: SubGraph>(
        hedge: Hedge,
        involution: &Involution<E>,
        subgraph: &S,
    ) -> Option<Self> {
        if subgraph.includes(&hedge) {
            Some(match involution.hedge_data(hedge) {
                InvolutiveMapping::Source { sink_idx, .. } => {
                    if subgraph.includes(sink_idx) {
                        HedgePair::Paired {
                            source: hedge,
                            sink: *sink_idx,
                        }
                    } else {
                        HedgePair::Split {
                            source: hedge,
                            sink: *sink_idx,
                            split: Flow::Source,
                        }
                    }
                }
                InvolutiveMapping::Sink { source_idx } => {
                    if subgraph.includes(source_idx) {
                        HedgePair::Paired {
                            source: *source_idx,
                            sink: hedge,
                        }
                    } else {
                        HedgePair::Split {
                            source: *source_idx,
                            sink: hedge,
                            split: Flow::Sink,
                        }
                    }
                }
                InvolutiveMapping::Identity { underlying, .. } => HedgePair::Unpaired {
                    hedge,
                    flow: *underlying,
                },
            })
        } else {
            None
        }
    }
}

impl Display for Hedge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InvolutiveMapping<E> {
    Identity { data: EdgeData<E>, underlying: Flow },
    Source { data: EdgeData<E>, sink_idx: Hedge },
    Sink { source_idx: Hedge },
}

impl<E> InvolutiveMapping<E> {
    pub fn flow(&self) -> Flow {
        match self {
            InvolutiveMapping::Identity { underlying, .. } => *underlying,
            InvolutiveMapping::Source { .. } => Flow::Source,
            InvolutiveMapping::Sink { .. } => Flow::Sink,
        }
    }

    fn dummy() -> Self {
        InvolutiveMapping::Sink {
            source_idx: Hedge(0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeData<E> {
    pub orientation: Orientation,
    pub data: E,
}

impl<E> From<Orientation> for EdgeData<Option<E>> {
    fn from(orientation: Orientation) -> Self {
        EdgeData::empty(orientation)
    }
}

impl<E> EdgeData<Option<E>> {
    pub fn is_set(&self) -> bool {
        self.data.is_some()
    }
    pub const fn none() -> Self {
        Self {
            orientation: Orientation::Undirected,
            data: None,
        }
    }

    pub fn permute(self) -> Option<EdgeData<E>> {
        Some(EdgeData {
            orientation: self.orientation,
            data: self.data?,
        })
    }
    pub fn empty(orientation: Orientation) -> Self {
        EdgeData {
            orientation,
            data: None,
        }
    }
    pub fn take(&mut self) -> Self {
        EdgeData {
            orientation: self.orientation,
            data: self.data.take(),
        }
    }
    pub fn and_then<U, F>(self, f: F) -> EdgeData<Option<U>>
    where
        F: FnOnce(E) -> U,
    {
        match self.data {
            Some(x) => EdgeData {
                data: Some(f(x)),
                orientation: self.orientation,
            },
            None => EdgeData {
                data: None,
                orientation: self.orientation,
            },
        }
    }
}

impl<E> EdgeData<E> {
    pub fn un_orient(&mut self) {
        self.orientation = Orientation::Undirected;
    }

    pub fn orient(&mut self) {
        if let Orientation::Undirected = self.orientation {
            self.orientation = Orientation::Default;
        }
    }
    pub fn flip_orientation(&mut self) {
        self.orientation = self.orientation.reverse();
    }
    pub fn new(data: E, orientation: Orientation) -> Self {
        EdgeData { data, orientation }
    }
    pub fn map<E2>(self, f: impl FnOnce(E) -> E2) -> EdgeData<E2> {
        EdgeData {
            orientation: self.orientation,
            data: f(self.data),
        }
    }

    pub fn map_option<E2>(self, f: impl FnOnce(E) -> Option<E2>) -> Option<EdgeData<E2>> {
        Some(EdgeData {
            orientation: self.orientation,
            data: f(self.data)?,
        })
    }

    pub fn map_result<E2, O>(self, f: impl FnOnce(E) -> Result<E2, O>) -> Result<EdgeData<E2>, O> {
        Ok(EdgeData {
            orientation: self.orientation,
            data: f(self.data)?,
        })
    }

    pub const fn as_ref(&self) -> EdgeData<&E> {
        EdgeData {
            orientation: self.orientation,
            data: &self.data,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    Default,
    Reversed,
    Undirected,
}

#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq, Hash)]
pub enum Flow {
    Source, // outgoing
    Sink,   // incoming
}

impl Flow {
    pub fn color(&self) -> &'static str {
        match self {
            Flow::Source => "red",
            Flow::Sink => "blue",
        }
    }
}

impl Neg for Flow {
    type Output = Flow;

    fn neg(self) -> Self::Output {
        match self {
            Flow::Source => Flow::Sink,
            Flow::Sink => Flow::Source,
        }
    }
}

impl From<bool> for Flow {
    fn from(value: bool) -> Self {
        if value {
            Flow::Source
        } else {
            Flow::Sink
        }
    }
}

impl From<Flow> for Orientation {
    fn from(value: Flow) -> Self {
        match value {
            Flow::Source => Orientation::Default,
            Flow::Sink => Orientation::Reversed,
        }
    }
}

impl From<Flow> for SignOrZero {
    fn from(value: Flow) -> Self {
        match value {
            Flow::Source => SignOrZero::Minus,
            Flow::Sink => SignOrZero::Plus,
        }
    }
}

impl TryFrom<Orientation> for Flow {
    type Error = &'static str;

    fn try_from(value: Orientation) -> Result<Self, Self::Error> {
        match value {
            Orientation::Default => Ok(Flow::Source),
            Orientation::Reversed => Ok(Flow::Sink),
            Orientation::Undirected => Err("Cannot convert undirected orientation to flow"),
        }
    }
}

impl Orientation {
    pub fn reverse(self) -> Orientation {
        match self {
            Orientation::Default => Orientation::Reversed,
            Orientation::Reversed => Orientation::Default,
            Orientation::Undirected => Orientation::Undirected,
        }
    }

    pub fn relative_to(&self, other: Flow) -> Orientation {
        match (self, other) {
            (Orientation::Default, Flow::Source) => Orientation::Default,
            (Orientation::Default, Flow::Sink) => Orientation::Reversed,
            (Orientation::Reversed, Flow::Source) => Orientation::Reversed,
            (Orientation::Reversed, Flow::Sink) => Orientation::Default,
            (Orientation::Undirected, _) => Orientation::Undirected,
        }
    }
}

impl From<bool> for Orientation {
    fn from(value: bool) -> Self {
        if value {
            Orientation::Default
        } else {
            Orientation::Undirected
        }
    }
}

#[derive(Error, Debug)]
pub enum SignError {
    #[error("Invalid value for Sign")]
    InvalidValue,
    #[error("Zero is not a valid value for Sign")]
    ZeroValue,
}
#[repr(i8)]
pub enum SignOrZero {
    Zero = 0,
    Plus = 1,
    Minus = -1,
}

impl TryFrom<i8> for SignOrZero {
    type Error = SignError;
    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SignOrZero::Zero),
            1 => Ok(SignOrZero::Plus),
            -1 => Ok(SignOrZero::Minus),
            _ => Err(SignError::InvalidValue),
        }
    }
}

impl Display for SignOrZero {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignOrZero::Zero => write!(f, "."),
            SignOrZero::Plus => write!(f, "+"),
            SignOrZero::Minus => write!(f, "-"),
        }
    }
}

impl From<SignOrZero> for Orientation {
    fn from(value: SignOrZero) -> Self {
        match value {
            SignOrZero::Zero => Orientation::Undirected,
            SignOrZero::Plus => Orientation::Default,
            SignOrZero::Minus => Orientation::Reversed,
        }
    }
}

#[allow(non_upper_case_globals)]
impl SignOrZero {
    pub fn is_zero(&self) -> bool {
        matches!(self, SignOrZero::Zero)
    }

    pub fn is_sign(&self) -> bool {
        matches!(self, SignOrZero::Plus | SignOrZero::Minus)
    }

    pub fn is_positive(&self) -> bool {
        matches!(self, SignOrZero::Plus)
    }

    pub fn is_negative(&self) -> bool {
        matches!(self, SignOrZero::Minus)
    }
}

impl<T: Neg<Output = T> + RefZero> Mul<T> for SignOrZero {
    type Output = T;
    fn mul(self, rhs: T) -> Self::Output {
        match self {
            SignOrZero::Plus => rhs,
            SignOrZero::Minus => -rhs,
            SignOrZero::Zero => rhs.ref_zero(),
        }
    }
}

impl Neg for SignOrZero {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            SignOrZero::Plus => SignOrZero::Minus,
            SignOrZero::Minus => SignOrZero::Plus,
            SignOrZero::Zero => SignOrZero::Zero,
        }
    }
}

impl From<Orientation> for SignOrZero {
    fn from(value: Orientation) -> Self {
        match value {
            Orientation::Default => SignOrZero::Plus,
            Orientation::Reversed => SignOrZero::Minus,
            Orientation::Undirected => SignOrZero::Zero,
        }
    }
}

impl<E> InvolutiveMapping<Option<E>> {}
impl<E> InvolutiveMapping<E> {
    pub fn is_identity(&self) -> bool {
        matches!(self, InvolutiveMapping::Identity { .. })
    }

    pub fn data(self) -> Option<EdgeData<E>> {
        match self {
            InvolutiveMapping::Identity { data, .. } => Some(data),
            InvolutiveMapping::Source { data, .. } => Some(data),
            _ => None,
        }
    }

    pub fn to_sign(&self) -> SignOrZero {
        match self {
            InvolutiveMapping::Identity { .. } => SignOrZero::Zero,
            InvolutiveMapping::Sink { .. } => SignOrZero::Plus, // incoming
            InvolutiveMapping::Source { .. } => SignOrZero::Minus,
        }
    }

    pub fn as_ref(&self) -> InvolutiveMapping<&E> {
        self.map_data_ref(|e| e)
    }

    pub fn map_data_ref<'a, E2>(&'a self, f: impl FnOnce(&'a E) -> E2) -> InvolutiveMapping<E2> {
        match self {
            InvolutiveMapping::Identity { data, underlying } => InvolutiveMapping::Identity {
                data: data.as_ref().map(f),
                underlying: *underlying,
            },
            InvolutiveMapping::Source { data, sink_idx } => InvolutiveMapping::Source {
                data: data.as_ref().map(f),
                sink_idx: *sink_idx,
            },
            InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink {
                source_idx: *source_idx,
            },
        }
    }

    pub fn map_data<E2>(self, f: impl FnOnce(E) -> E2) -> InvolutiveMapping<E2> {
        match self {
            InvolutiveMapping::Identity { data, underlying } => InvolutiveMapping::Identity {
                data: data.map(f),
                underlying,
            },
            InvolutiveMapping::Source { data, sink_idx } => InvolutiveMapping::Source {
                data: data.map(f),
                sink_idx,
            },
            InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink { source_idx },
        }
    }

    pub fn map_data_option<E2>(
        self,
        f: impl FnOnce(E) -> Option<E2>,
    ) -> Option<InvolutiveMapping<E2>> {
        match self {
            InvolutiveMapping::Identity { data, underlying } => Some(InvolutiveMapping::Identity {
                data: data.map_option(f)?,
                underlying,
            }),
            InvolutiveMapping::Source { data, sink_idx } => Some(InvolutiveMapping::Source {
                data: data.map_option(f)?,
                sink_idx,
            }),
            InvolutiveMapping::Sink { source_idx } => Some(InvolutiveMapping::Sink { source_idx }),
        }
    }

    pub fn map_data_result<E2, O>(
        self,
        f: impl FnOnce(E) -> Result<E2, O>,
    ) -> Result<InvolutiveMapping<E2>, O> {
        match self {
            InvolutiveMapping::Identity { data, underlying } => Ok(InvolutiveMapping::Identity {
                data: data.map_result(f)?,
                underlying,
            }),
            InvolutiveMapping::Source { data, sink_idx } => Ok(InvolutiveMapping::Source {
                data: data.map_result(f)?,
                sink_idx,
            }),
            InvolutiveMapping::Sink { source_idx } => Ok(InvolutiveMapping::Sink { source_idx }),
        }
    }

    pub fn map_empty<E2>(&self) -> InvolutiveMapping<Option<E2>> {
        self.map_data_ref(|_| None)
    }

    pub fn is_internal(&self) -> bool {
        !matches!(self, InvolutiveMapping::Identity { .. })
    }

    pub fn is_source(&self) -> bool {
        matches!(self, InvolutiveMapping::Source { .. })
    }

    pub fn is_sink(&self) -> bool {
        matches!(self, InvolutiveMapping::Sink { .. })
    }

    // pub fn make_source(&mut self, sink_idx: Hedge) -> Option<InvolutiveMapping<E>> {
    //     let data = match self {
    //         InvolutiveMapping::Identity { data, .. } => data.take(),
    //         _ => EdgeData::none(),
    //     };
    //     Some(InvolutiveMapping::Source { data, sink_idx })
    // }

    pub fn new_identity(data: E, orientation: impl Into<Orientation>, underlying: Flow) -> Self {
        let o = orientation.into();
        InvolutiveMapping::Identity {
            data: EdgeData::new(data, o),
            underlying,
        }
    }

    pub fn identity_dot(
        edge_id: Hedge,
        source: NodeIndex,
        attr: Option<&GVEdgeAttrs>,
        orientation: Orientation,
        flow: Flow,
    ) -> String {
        let mut out = "".to_string();
        out.push_str(&format!("ext{} [shape=none, label=\"\"];\n ", edge_id));

        out.push_str(&format!("ext{} -> {}[", edge_id, source));
        match (orientation, flow) {
            (Orientation::Default, Flow::Source) => {
                out.push_str("dir=back ");
            }
            (Orientation::Default, Flow::Sink) => {
                out.push_str("dir=forward ");
            }
            (Orientation::Reversed, Flow::Sink) => {
                out.push_str("dir=back ");
            }
            (Orientation::Reversed, Flow::Source) => {
                out.push_str("dir=forward ");
            }
            (Orientation::Undirected, _) => {
                out.push_str("dir=none ");
            }
        }
        if let Some(attr) = attr {
            out.push_str(&format!("{}", attr));
        }
        out.push_str("];\n");
        out
    }

    pub fn pair_dot(
        source: NodeIndex,
        sink: NodeIndex,
        attr: Option<&GVEdgeAttrs>,
        orientation: Orientation,
    ) -> String {
        let mut out = "".to_string();

        out.push_str(&format!("{} -> {}[", source, sink));
        match orientation {
            Orientation::Default => {
                out.push_str(" dir=forward ");
            }
            Orientation::Reversed => {
                out.push_str(" dir=back ");
            }
            Orientation::Undirected => {
                out.push_str(" dir=none ");
            }
        }
        if let Some(attr) = attr {
            out.push_str(&format!("{}];\n", attr));
        } else {
            out.push_str(" color=\"red:blue;0.5 \" ];\n");
        }
        out
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Error)]
pub enum InvolutionError {
    #[error("Should have been identity")]
    NotIdentity,
    #[error("Should have been an paired hedge")]
    NotPaired,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Involution<E = EdgeIndex> {
    pub(super) inv: Vec<InvolutiveMapping<E>>,
}

impl<E> AsRef<Involution<E>> for Involution<E> {
    fn as_ref(&self) -> &Involution<E> {
        self
    }
}

impl<E> Default for Involution<E> {
    fn default() -> Self {
        Involution::new()
    }
}

impl<E> Involution<E> {
    /// The fundamental function of involutions is to pair edges. This function provides this pairing.
    /// It is a bijective map inv: H -> H, where H is the set of half-edges.
    /// The map is involutive, meaning that inv(inv(h)) = h.
    /// The fixed points of this function correspond to the unpaired hedges, interpreted as dangling or external edges.
    pub fn inv(&self, hedge: Hedge) -> Hedge {
        match self.hedge_data(hedge) {
            InvolutiveMapping::Sink { source_idx } => *source_idx,
            InvolutiveMapping::Source { sink_idx, .. } => *sink_idx,
            _ => hedge,
        }
    }

    pub fn len(&self) -> usize {
        self.inv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inv.is_empty()
    }

    pub fn new() -> Self {
        Involution { inv: Vec::new() }
    }

    /// add a new dangling edge to the involution
    /// returns the index of the new edge
    pub fn add_identity(
        &mut self,
        data: E,
        orientation: impl Into<Orientation>,
        underlying: Flow,
    ) -> Hedge {
        let index = self.inv.len();
        self.inv.push(InvolutiveMapping::new_identity(
            data,
            orientation,
            underlying,
        ));
        Hedge(index)
    }

    /// adds a new hedge that connects to an identity hedge
    pub fn connect_to_identity(&mut self, connect: Hedge) -> Result<Hedge, InvolutionError> {
        let sink = Hedge(self.inv.len());
        if !self.is_identity(connect) {
            return Err(InvolutionError::NotIdentity);
        }
        self.inv.push(InvolutiveMapping::Sink {
            source_idx: connect,
        });

        let flow = self.flow(connect);
        if let Some(data) = self.get_data_owned(connect) {
            self.set(
                connect,
                InvolutiveMapping::Source {
                    data,
                    sink_idx: sink,
                },
            );
            if let Flow::Sink = flow {
                self.flip_underlying(sink);
            }
            Ok(sink)
        } else {
            panic!("Should have data");
        }
    }

    /// add a connected pair of hedges to the involution
    /// returns the pair of hedges: (source, sink)
    pub fn add_pair(&mut self, data: E, directed: impl Into<Orientation>) -> (Hedge, Hedge) {
        let orientation = directed.into();
        let source = self.add_identity(data, orientation, Flow::Sink);
        let sink = self.connect_to_identity(source).unwrap();
        (source, sink)
    }

    pub fn find_from_data(&self, data: &E) -> Option<HedgePair>
    where
        E: PartialEq,
    {
        self.iter_edge_data().find_map(|(i, d)| {
            if &d.data == data {
                Some(self.hedge_pair(i))
            } else {
                None
            }
        })
    }

    pub fn last(&self) -> Option<Hedge> {
        self.len().checked_sub(1).map(Hedge)
    }

    fn validate(&self) -> bool {
        self.iter_idx().all(|h| {
            let pair = self.inv(h);
            if h == pair {
                self.is_identity(h)
            } else {
                self.inv(pair) == h
                    && ((self.is_sink(h) && self.is_source(pair))
                        || (self.is_source(h) && self.is_sink(pair)))
            }
        })
    }

    /// modify the involution to set hedge to sink.
    /// this might make the structure invalid
    fn set(&mut self, hedge: Hedge, mapping: InvolutiveMapping<E>) -> Option<EdgeData<E>> {
        if self.is_sink(hedge) {
            self.inv[hedge.0] = mapping;
            return None;
        }
        let data = self.inv.swap_remove(hedge.0);
        self.inv.push(mapping);
        let last = self.inv.len().checked_sub(1).unwrap();
        self.inv.swap(hedge.0, last);

        data.data()
    }

    /// extract the data from the hedge replacing it with a dummy mapping
    fn get_data_owned(&mut self, hedge: Hedge) -> Option<EdgeData<E>> {
        self.set(hedge, InvolutiveMapping::dummy())
    }

    /// connect a pair of identity hedges,
    /// taking orientation and data from source_idx
    /// errors if the hedges are not identity
    pub fn connect_identities(
        &mut self,
        source_idx: Hedge,
        sink_idx: Hedge,
        merge_fn: impl FnOnce(Flow, EdgeData<E>, Flow, EdgeData<E>) -> (Flow, EdgeData<E>),
    ) -> Result<(), InvolutionError>
    where
        E: Clone,
    {
        if self.is_identity(source_idx) && self.is_identity(sink_idx) {
            let source_flow = self.flow(source_idx);
            let sink_flow = self.flow(sink_idx);
            let source_data = self.get_data_owned(source_idx);
            let sink_data = self.get_data_owned(sink_idx);
            if let (Some(so_d), Some(si_d)) = (source_data, sink_data) {
                let (new_flow, new_data) = merge_fn(source_flow, so_d, sink_flow, si_d);

                match new_flow {
                    Flow::Source => {
                        self.set(
                            source_idx,
                            InvolutiveMapping::Source {
                                data: new_data,
                                sink_idx,
                            },
                        );
                        self.set(sink_idx, InvolutiveMapping::Sink { source_idx });
                    }
                    Flow::Sink => {
                        self.set(
                            source_idx,
                            InvolutiveMapping::Sink {
                                source_idx: sink_idx,
                            },
                        );
                        self.set(
                            sink_idx,
                            InvolutiveMapping::Source {
                                data: new_data,
                                sink_idx: source_idx,
                            },
                        );
                    }
                }
            }
            Ok(())
        } else {
            Err(InvolutionError::NotIdentity)
        }
    }

    /// Splits the edge that hedge is a part of into two dangling hedges, adding the data to the side given by hedge.
    /// The underlying orientation of the new edges is the same as the original edge, i.e. the source will now have `Flow::Source` and the sink will have `Flow::Sink`.
    /// The superficial orientation has to be given knowing this.
    pub fn split_edge(&mut self, hedge: Hedge, data: EdgeData<E>) -> Result<(), InvolutionError> {
        if self.is_identity(hedge) {
            Err(InvolutionError::NotPaired)
        } else {
            let invh = self.inv(hedge);
            if let Some(replacing_data) = self.get_data_owned(hedge) {
                //hedge is a source, replace its data with the new data
                self.set(
                    hedge,
                    InvolutiveMapping::Identity {
                        data,
                        underlying: Flow::Source,
                    },
                );
                self.set(
                    invh,
                    InvolutiveMapping::Identity {
                        data: replacing_data,
                        underlying: Flow::Sink,
                    },
                );
            } else {
                // hedge is a sink, give it the new data
                self.set(
                    hedge,
                    InvolutiveMapping::Identity {
                        data,
                        underlying: Flow::Sink,
                    },
                );
                let data = self.get_data_owned(invh).unwrap(); //extract the data from the source
                self.set(
                    invh,
                    InvolutiveMapping::Identity {
                        data,
                        underlying: Flow::Source,
                    },
                );
            }
            Ok(())
        }
    }

    pub fn hedge_pair(&self, hedge: Hedge) -> HedgePair {
        HedgePair::from_half_edge(hedge, self)
    }

    pub(crate) fn smart_data<S: SubGraph>(
        &self,
        hedge: Hedge,
        subgraph: &S,
    ) -> Option<&EdgeData<E>> {
        if subgraph.includes(&hedge) {
            match self.hedge_data(hedge) {
                InvolutiveMapping::Identity { .. } => Some(self.edge_data(hedge)),
                InvolutiveMapping::Source { .. } => Some(self.edge_data(hedge)),
                InvolutiveMapping::Sink { source_idx } => {
                    if subgraph.includes(source_idx) {
                        None
                    } else {
                        Some(self.edge_data(*source_idx))
                    }
                }
            }
        } else {
            None
        }
    }

    fn smart_data_mut<S: SubGraph>(
        &mut self,
        hedge: Hedge,
        subgraph: &S,
    ) -> Option<&mut EdgeData<E>> {
        if subgraph.includes(&hedge) {
            match self.hedge_data(hedge) {
                InvolutiveMapping::Identity { .. } => Some(self.edge_data_mut(hedge)),
                InvolutiveMapping::Source { .. } => Some(self.edge_data_mut(hedge)),
                InvolutiveMapping::Sink { source_idx } => {
                    if subgraph.includes(source_idx) {
                        None
                    } else {
                        Some(self.edge_data_mut(*source_idx))
                    }
                }
            }
        } else {
            None
        }
    }

    pub fn first_internal<S: SubGraph>(&self, subgraph: &S) -> Option<Hedge> {
        self.iter_idx().find(|e| self.is_internal(*e, subgraph))
    }

    pub fn n_internals<S: SubGraph>(&self, subgraph: &S) -> usize {
        subgraph
            .included_iter()
            .filter(|i| self.is_internal(*i, subgraph))
            .count()
            / 2
    }

    /// check if the edge is internal and totally included in the subgraph
    pub fn is_internal<S: SubGraph>(&self, index: Hedge, subgraph: &S) -> bool {
        if !subgraph.includes(&index) {
            return false;
        }
        match &self.inv[index.0] {
            InvolutiveMapping::Identity { .. } => false,
            InvolutiveMapping::Source { sink_idx, .. } => subgraph.includes(sink_idx),
            InvolutiveMapping::Sink { source_idx } => subgraph.includes(source_idx),
        }
    }

    pub fn orientation(&self, hedge: Hedge) -> Orientation {
        match self.hedge_data(hedge) {
            InvolutiveMapping::Identity { data, .. } => data.orientation,
            InvolutiveMapping::Source { data, .. } => data.orientation,
            InvolutiveMapping::Sink { source_idx } => self.orientation(*source_idx),
        }
    }

    pub fn flow(&self, hedge: Hedge) -> Flow {
        self.hedge_data(hedge).flow()
    }

    pub fn iter_edge_data(&self) -> impl Iterator<Item = (Hedge, &EdgeData<E>)> {
        self.iter().filter_map(|(e, m)| match m {
            InvolutiveMapping::Source { data, .. } => Some((e, data)),
            InvolutiveMapping::Identity { data, .. } => Some((e, data)),
            _ => None,
        })
    }

    pub fn iter_edge_data_mut(&mut self) -> impl Iterator<Item = &mut EdgeData<E>> {
        self.iter_mut().filter_map(|(_, m)| match m {
            InvolutiveMapping::Source { data, .. } => Some(data),
            InvolutiveMapping::Identity { data, .. } => Some(data),
            _ => None,
        })
    }

    pub fn iter(&self) -> InvolutionIter<E> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> InvolutionIterMut<E> {
        self.into_iter()
    }

    pub fn iter_idx(&self) -> impl Iterator<Item = Hedge> {
        (0..self.inv.len()).map(Hedge)
    }

    pub fn as_ref(&self) -> Involution<&E> {
        let inv = self.inv.iter().map(|e| e.as_ref()).collect();
        Involution { inv }
    }

    pub fn map_data_ref<'a, G, E2>(&'a self, g: &G) -> Involution<E2>
    where
        G: FnMut(&'a E) -> E2 + Clone,
    {
        let inv = self
            .inv
            .iter()
            .map(|e| (e.map_data_ref(g.clone())))
            .collect();

        Involution { inv }
    }

    pub fn map_data<F, G, E2>(self, g: &G) -> Involution<E2>
    where
        G: FnMut(E) -> E2 + Clone,
    {
        let inv = self
            .inv
            .into_iter()
            .map(|e| (e.map_data(g.clone())))
            .collect();

        Involution { inv }
    }

    pub fn map_data_option<F, G, E2>(self, g: &G) -> Option<Involution<E2>>
    where
        G: FnMut(E) -> Option<E2> + Clone,
    {
        let inv = self
            .inv
            .into_iter()
            .map(|e| (e.map_data_option(g.clone())))
            .collect::<Option<Vec<_>>>()?;

        Some(Involution { inv })
    }

    pub fn map_data_result<F, G, E2, O>(self, g: &G) -> Result<Involution<E2>, O>
    where
        G: FnMut(E) -> Result<E2, O> + Clone,
    {
        let inv = self
            .inv
            .into_iter()
            .map(|e| (e.map_data_result(g.clone())))
            .collect::<Result<Vec<_>, O>>()?;

        Ok(Involution { inv })
    }

    pub fn map_full<E2>(
        self,
        mut g: impl FnMut(HedgePair, EdgeData<E>) -> EdgeData<E2>,
    ) -> Involution<E2> {
        let inv = self
            .into_iter()
            .map(|(i, e)| match e {
                InvolutiveMapping::Identity { data, underlying } => InvolutiveMapping::Identity {
                    data: g(
                        HedgePair::Unpaired {
                            hedge: i,
                            flow: underlying,
                        },
                        data,
                    ),
                    underlying,
                },
                InvolutiveMapping::Source { data, sink_idx } => InvolutiveMapping::Source {
                    data: g(
                        HedgePair::Paired {
                            source: i,
                            sink: sink_idx,
                        },
                        data,
                    ),

                    sink_idx,
                },
                InvolutiveMapping::Sink { source_idx } => InvolutiveMapping::Sink { source_idx },
            })
            .collect();

        Involution { inv }
    }

    pub fn is_sink(&self, hedge: Hedge) -> bool {
        self.hedge_data(hedge).is_sink()
    }

    pub fn is_source(&self, hedge: Hedge) -> bool {
        self.hedge_data(hedge).is_source()
    }

    pub fn is_identity(&self, hedge: Hedge) -> bool {
        self.hedge_data(hedge).is_identity()
    }

    /// If the data at `hedge` was a sink, turn it into a source.
    /// Otherwise do nothing.
    pub fn set_as_source(&mut self, hedge: Hedge) {
        let is_sink = self.is_sink(hedge);
        if is_sink {
            self.flip_underlying(hedge)
        }
    }

    /// If the data at `hedge` was a source, turn it into a sink.
    /// Otherwise do nothing.
    pub fn set_as_sink(&mut self, hedge: Hedge) {
        let is_source = self.is_source(hedge);
        if is_source {
            self.flip_underlying(hedge)
        }
    }

    /// Swap the data carrier in a pair of hedges.
    pub(super) fn flip_underlying(&mut self, hedge: Hedge) {
        let pair = self.inv(hedge);

        self.inv.swap(hedge.0, pair.0);

        match self.hedge_data_mut(hedge) {
            InvolutiveMapping::Identity { underlying, .. } => {
                *underlying = -*underlying;
            }
            InvolutiveMapping::Source { data, sink_idx } => {
                *sink_idx = pair;
                data.orientation = data.orientation.reverse();
            }
            InvolutiveMapping::Sink { source_idx } => {
                *source_idx = pair;
            }
        }

        match self.hedge_data_mut(pair) {
            InvolutiveMapping::Identity { underlying, .. } => {
                *underlying = -*underlying;
            }
            InvolutiveMapping::Source { data, sink_idx } => {
                *sink_idx = hedge;
                data.orientation = data.orientation.reverse();
            }
            InvolutiveMapping::Sink { source_idx } => {
                *source_idx = hedge;
            }
        }
    }

    pub(crate) fn random(len: usize, seed: u64) -> Involution<()> {
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut inv = Involution::new();

        for _ in 0..len {
            let r = rng.gen_bool(0.1);
            if r {
                inv.add_identity((), Orientation::Undirected, Flow::Sink);
            } else {
                inv.add_pair((), false);
            }
        }

        inv
    }

    pub fn edge_data(&self, index: Hedge) -> &EdgeData<E> {
        match &self.inv[index.0] {
            InvolutiveMapping::Source { data, .. } => data,
            InvolutiveMapping::Identity { data, .. } => data,
            InvolutiveMapping::Sink { source_idx } => self.edge_data(*source_idx),
        }
    }

    #[allow(clippy::needless_return)]
    pub fn edge_data_mut(&mut self, index: Hedge) -> &mut EdgeData<E> {
        if let InvolutiveMapping::Sink { source_idx } = self.inv[index.0] {
            return self.edge_data_mut(source_idx);
        }
        match &mut self.inv[index.0] {
            InvolutiveMapping::Source { data, .. } => return data,
            InvolutiveMapping::Identity { data, .. } => return data,
            _ => unreachable!(),
        };
    }

    pub(super) fn hedge_data(&self, hedge: Hedge) -> &InvolutiveMapping<E> {
        &self.inv[hedge.0]
    }

    pub(super) fn hedge_data_mut(&mut self, hedge: Hedge) -> &mut InvolutiveMapping<E> {
        &mut self.inv[hedge.0]
    }

    fn data_inv(&self, hedge: Hedge) -> Hedge {
        match self.hedge_data(hedge) {
            InvolutiveMapping::Sink { source_idx } => *source_idx,
            _ => hedge,
        }
    }

    pub fn print<S: SubGraph>(
        &self,
        subgraph: &S,
        h_label: &impl Fn(&E) -> Option<String>,
    ) -> String {
        let mut out = "".to_string();
        for (i, e) in self.iter() {
            if !subgraph.includes(&i) {
                continue;
            }
            match e {
                InvolutiveMapping::Identity { .. } => {
                    out.push_str(&format!("{}\n", i));
                }
                InvolutiveMapping::Source { data, sink_idx } => {
                    let d = &data.data;
                    if let Some(l) = h_label(d) {
                        out.push_str(&format!("{}-{}->{}\n", i, sink_idx, l));
                    } else {
                        out.push_str(&format!("{}->{}\n", i, sink_idx));
                    }
                }
                InvolutiveMapping::Sink { source_idx } => {
                    out.push_str(&format!("{}<-{}\n", i, source_idx));
                }
            }
        }
        out
    }
}

impl<E> FromIterator<InvolutiveMapping<E>> for Involution<E> {
    fn from_iter<I: IntoIterator<Item = InvolutiveMapping<E>>>(iter: I) -> Self {
        Involution {
            inv: iter.into_iter().collect(),
        }
    }
}

impl<E> Index<Hedge> for Involution<E> {
    type Output = E;

    fn index(&self, index: Hedge) -> &Self::Output {
        match &self.inv[index.0] {
            InvolutiveMapping::Identity { data, .. } => &data.data,
            InvolutiveMapping::Source { data, .. } => &data.data,
            InvolutiveMapping::Sink { source_idx } => &self[*source_idx],
        }
    }
}

impl<E> IndexMut<Hedge> for Involution<E> {
    fn index_mut(&mut self, index: Hedge) -> &mut Self::Output {
        let invh = self.data_inv(index);

        match self.hedge_data_mut(invh) {
            InvolutiveMapping::Identity { data, .. } => &mut data.data,
            InvolutiveMapping::Source { data, .. } => &mut data.data,
            _ => panic!("should have gotten data inv"),
        }
    }
}

#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash,
)]
pub struct EdgeIndex(pub(crate) usize);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeVec<E> {
    inv: Involution<usize>,
    data: Vec<E>,
}

pub struct DrainingInvolutionIter<E> {
    into: std::vec::IntoIter<InvolutiveMapping<E>>,
    current: Hedge,
}

impl<E> Iterator for DrainingInvolutionIter<E> {
    type Item = (Hedge, InvolutiveMapping<E>);
    fn next(&mut self) -> Option<Self::Item> {
        self.into.next().map(|e| {
            let out = (self.current, e);
            self.current.0 += 1;
            out
        })
    }
}

impl<E> IntoIterator for Involution<E> {
    type Item = (Hedge, InvolutiveMapping<E>);
    type IntoIter = DrainingInvolutionIter<E>;
    fn into_iter(self) -> Self::IntoIter {
        DrainingInvolutionIter {
            into: self.inv.into_iter(),
            current: Hedge(0),
        }
    }
}

pub struct InvolutionIter<'a, E> {
    into: std::slice::Iter<'a, InvolutiveMapping<E>>,
    current: Hedge,
}

impl<'a, E> Iterator for InvolutionIter<'a, E> {
    type Item = (Hedge, &'a InvolutiveMapping<E>);
    fn next(&mut self) -> Option<Self::Item> {
        self.into.next().map(|e| {
            let out = (self.current, e);
            self.current.0 += 1;
            out
        })
    }
}

impl<'a, E> IntoIterator for &'a Involution<E> {
    type Item = (Hedge, &'a InvolutiveMapping<E>);
    type IntoIter = InvolutionIter<'a, E>;
    fn into_iter(self) -> Self::IntoIter {
        InvolutionIter {
            into: self.inv.iter(),
            current: Hedge(0),
        }
    }
}

pub struct InvolutionIterMut<'a, E> {
    into: std::slice::IterMut<'a, InvolutiveMapping<E>>,
    current: Hedge,
}

impl<'a, E> Iterator for InvolutionIterMut<'a, E> {
    type Item = (Hedge, &'a mut InvolutiveMapping<E>);
    fn next(&mut self) -> Option<Self::Item> {
        self.into.next().map(|e| {
            let out = (self.current, e);
            self.current.0 += 1;
            out
        })
    }
}

impl<'a, E> IntoIterator for &'a mut Involution<E> {
    type Item = (Hedge, &'a mut InvolutiveMapping<E>);
    type IntoIter = InvolutionIterMut<'a, E>;
    fn into_iter(self) -> Self::IntoIter {
        InvolutionIterMut {
            into: self.inv.iter_mut(),
            current: Hedge(0),
        }
    }
}

// pub trait Get<H> {
//     type Output<'a>
//     where
//         Self: 'a;

//     type MutOutput<'a>
//     where
//         Self: 'a;
//     fn get(&self, h: H) -> Self::Output<'_>;

//     fn get_mut(&mut self, h: H) -> Self::MutOutput<'_>;
// }

// impl<N, E> Get<Hedge> for Involution<N, E> {
//     type Output<'a>
//         = &'a InvolutiveMapping<E>
//     where
//         Self: 'a;
//     type MutOutput<'a>
//         = &'a mut InvolutiveMapping<E>
//     where
//         Self: 'a;
//     fn get(&self, h: Hedge) -> Self::Output<'_> {
//         &self.inv[h.0]
//     }

//     fn get_mut(&mut self, h: Hedge) -> Self::MutOutput<'_> {
//         &mut self.inv[h.0]
//     }
// }

// impl<N, E> Get<HedgePair> for Involution<N, E> {
//     type Output<'a>
//         = Option<(Flow, &'a EdgeData<E>)>
//     where
//         Self: 'a;

//     type MutOutput<'a>
//         = Option<(Flow, &'a mut EdgeData<E>)>
//     where
//         Self: 'a;

//     fn get(&self, h: HedgePair) -> Self::Output<'_> {
//         let (source, flow) = match h {
//             HedgePair::Paired { source, .. } => (source, Flow::Source),
//             HedgePair::Unpaired { hedge, flow } => (hedge, flow),
//             HedgePair::Split { source, split, .. } => (source, split),
//         };

//         match &self[source] {
//             InvolutiveMapping::Source { data, .. } => Some((flow, data)),
//             InvolutiveMapping::Identity { data, underlying } => Some((*underlying, data)),
//             _ => None,
//         }
//     }

//     fn get_mut(&mut self, h: HedgePair) -> Self::MutOutput<'_> {
//         let (source, flow) = match h {
//             HedgePair::Paired { source, .. } => (source, Flow::Source),
//             HedgePair::Unpaired { hedge, flow } => (hedge, flow),
//             HedgePair::Split { source, split, .. } => (source, split),
//         };

//         match &mut self[source] {
//             InvolutiveMapping::Source { data, .. } => Some((flow, data)),
//             InvolutiveMapping::Identity { data, underlying } => Some((*underlying, data)),
//             _ => None,
//         }
//     }
// }

impl<E> Display for Involution<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = "".to_string();
        for (i, e) in self.inv.iter().enumerate() {
            match e {
                InvolutiveMapping::Identity { .. } => {
                    out.push_str(&format!("{}\n", i));
                }
                InvolutiveMapping::Source { sink_idx, .. } => {
                    out.push_str(&format!("{}->{}\n", i, sink_idx));
                }
                InvolutiveMapping::Sink { source_idx } => {
                    out.push_str(&format!("{}<-{}\n", i, source_idx));
                }
            }
        }
        write!(f, "{}", out)
    }
}
