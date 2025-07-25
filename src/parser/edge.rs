use std::{borrow::Borrow, collections::BTreeMap, fmt::Display, str::FromStr};

use itertools::Either;

use crate::half_edge::{
    builder::HedgeData,
    involution::{EdgeIndex, Flow, Hedge, Orientation},
};

use super::{DotHedgeData, GlobalData, NodeIdOrDangling};

pub struct ParsingEdge {
    edge: DotEdgeData,
    source_id: Option<Hedge>,
    sink_id: Option<Hedge>,
}

impl FromIterator<(String, String)> for ParsingEdge {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(iter: T) -> Self {
        let mut source_id = None;
        let mut sink_id = None;
        let mut edge_id = None;
        let statements = iter
            .into_iter()
            .filter_map(|(k, v)| match k.as_str() {
                "sink_id" => {
                    sink_id = Some(Hedge(v.parse::<usize>().unwrap()));
                    None
                }
                "source_id" => {
                    source_id = Some(Hedge(v.parse::<usize>().unwrap()));
                    None
                }
                "id" => {
                    edge_id = Some(EdgeIndex::from(v.parse::<usize>().unwrap()));
                    None
                }
                _ => Some((k, v)),
            })
            .collect();

        ParsingEdge {
            edge: DotEdgeData {
                statements,
                edge_id,
            },
            source_id,
            sink_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotEdgeData {
    pub statements: BTreeMap<String, String>,
    pub edge_id: Option<EdgeIndex>,
}

impl Display for DotEdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (key, value) in &self.statements {
            write!(f, "{key}={value},")?;
        }
        Ok(())
    }
}

impl DotEdgeData {
    pub fn empty() -> Self {
        DotEdgeData {
            statements: BTreeMap::new(),
            edge_id: None,
        }
    }

    pub fn extend(&mut self, other: Self) {
        self.statements.extend(other.statements);
    }

    pub fn format(&self, template: impl AsRef<str>) -> String {
        let mut result = template.as_ref().to_owned();

        // Find all occurrences of {key} in the template
        for (key, value) in &self.statements {
            let placeholder = format!("{{{key}}} ");
            result = result.replace(&placeholder, value);
        }

        // Remove any remaining {whatever} patterns
        while let Some(start) = result.find('{') {
            if let Some(end) = result[start..].find('}') {
                result.replace_range(start..=start + end, "");
            } else {
                break;
            }
        }

        result
    }

    pub fn add_statement(&mut self, key: impl ToString, value: impl ToString) {
        self.statements.insert(key.to_string(), value.to_string());
    }

    pub fn get<Q: Ord + ?Sized, F: FromStr>(&self, key: &Q) -> Option<Result<F, F::Err>>
    where
        String: Borrow<Q>,
    {
        self.statements.get(key).map(|s| s.parse())
    }

    pub fn from_parser(
        edge: dot_parser::canonical::Edge<(String, String)>,
        map: &BTreeMap<String, NodeIdOrDangling>,
        orientation: impl Into<Orientation>,
        global_data: &GlobalData,
    ) -> (
        Self,
        Orientation,
        HedgeData<DotHedgeData>,
        Either<HedgeData<DotHedgeData>, Flow>,
    ) {
        let mut orientation = orientation.into();
        let mut source_h_data = None;
        let mut sink_h_data = None;

        let mut statements = global_data.edge_statements.clone();
        statements.extend(edge.attr.into_iter().filter_map(|(key, value)| {
            match key.as_str() {
                "dir" => match value.as_str() {
                    "forward" => orientation = Orientation::Default,
                    "back" => orientation = Orientation::Reversed,
                    "none" => orientation = Orientation::Undirected,
                    _ => panic!("Invalid edge direction"),
                },
                "source" => {
                    source_h_data = Some(value);
                }
                "sink" => {
                    sink_h_data = Some(value);
                }
                _ => {
                    return Some((key, value));
                }
            }
            None
        }));

        let source = map[&edge.from].clone();
        let target = map[&edge.to].clone();

        let (edge, source, target) = match (source, target) {
            (NodeIdOrDangling::Id(source), NodeIdOrDangling::Id(target)) => {
                //Full edge

                let edge: ParsingEdge = statements.into_iter().collect();

                let mut source_data: DotHedgeData = source_h_data.into();
                let mut sink_data: DotHedgeData = sink_h_data.into();

                if let Some(sink) = edge.sink_id {
                    sink_data = sink_data.with_id(sink)
                }
                if let Some(source) = edge.source_id {
                    source_data = source_data.with_id(source)
                }
                (
                    edge.edge,
                    source.add_data(source_data),
                    Either::Left(target.add_data(sink_data)),
                )
            }
            (
                NodeIdOrDangling::Id(source),
                NodeIdOrDangling::Dangling {
                    flow,
                    statements: states,
                },
            ) => {
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                let edge: ParsingEdge = statements.into_iter().collect();

                orientation = orientation.relative_to(-flow);
                let source = match flow {
                    Flow::Sink => {
                        let mut sink_data: DotHedgeData = sink_h_data.into();

                        if let Some(sink) = edge.sink_id {
                            sink_data = sink_data.with_id(sink)
                        }

                        if edge.source_id.is_some() {
                            panic!("Sink edge cannot have a source id");
                        }
                        source.add_data(sink_data)
                    }
                    Flow::Source => {
                        let mut source_data: DotHedgeData = source_h_data.into();

                        if let Some(source) = edge.source_id {
                            source_data = source_data.with_id(source)
                        }

                        if edge.sink_id.is_some() {
                            panic!("Source edge cannot have a sink id");
                        }
                        source.add_data(source_data)
                    }
                };
                (edge.edge, source, Either::Right(-flow))
            }
            (
                NodeIdOrDangling::Dangling {
                    flow,
                    statements: states,
                },
                NodeIdOrDangling::Id(source),
            ) => {
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                let edge: ParsingEdge = statements.into_iter().collect();

                orientation = orientation.relative_to(flow);
                let source = match flow {
                    Flow::Sink => {
                        let mut sink_data: DotHedgeData = sink_h_data.into();

                        if let Some(sink) = edge.sink_id {
                            sink_data = sink_data.with_id(sink)
                        }
                        if edge.source_id.is_some() {
                            panic!("Sink edge cannot have a source id");
                        }
                        source.add_data(sink_data)
                    }
                    Flow::Source => {
                        let mut source_data: DotHedgeData = source_h_data.into();

                        if let Some(source) = edge.source_id {
                            source_data = source_data.with_id(source)
                        }
                        if edge.sink_id.is_some() {
                            panic!("Source edge cannot have a sink id");
                        }
                        source.add_data(source_data)
                    }
                };
                (edge.edge, source, Either::Right(-flow))
            }
            _ => panic!("Cannot connect an edge to two external nodes"),
        };

        (edge, orientation, source, target)
    }
}
