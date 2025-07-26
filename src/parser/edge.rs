use std::{borrow::Borrow, collections::BTreeMap, fmt::Display, str::FromStr};

use itertools::Either;

use crate::half_edge::{
    builder::HedgeData,
    involution::{EdgeIndex, Flow, Orientation},
};

use super::{subgraph_free::Edge, DotHedgeData, GlobalData, NodeIdOrDangling};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotEdgeData {
    pub statements: BTreeMap<String, String>,
    pub edge_id: Option<EdgeIndex>,
}

impl FromIterator<(String, String)> for DotEdgeData {
    fn from_iter<T: IntoIterator<Item = (String, String)>>(iter: T) -> Self {
        let mut edge_id = None;
        let statements = iter
            .into_iter()
            .filter_map(|(k, v)| match k.as_str() {
                "id" => {
                    edge_id = Some(EdgeIndex::from(v.parse::<usize>().unwrap()));
                    None
                }
                _ => Some((k, v)),
            })
            .collect();

        DotEdgeData {
            statements,
            edge_id,
        }
    }
}

impl Display for DotEdgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (key, value) in &self.statements {
            if !first {
                write!(f, " ")?;
            }
            write!(f, "{key}={value}")?;
            first = false;
        }

        // if let Some(id) = self.edge_id {
        //     write!(f, " id={}", id.0)?;
        // }

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
        edge: Edge,
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

        let sink_id = edge.sink_id();
        let source_id = edge.source_id();
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

        let source = map[&edge.from.id].clone();
        let target = map[&edge.to.id].clone();

        let (edge, source, target) = match (source, target) {
            (NodeIdOrDangling::Id(source), NodeIdOrDangling::Id(target)) => {
                //Full edge

                let dot_edge: DotEdgeData = statements.into_iter().collect();

                let mut source_data: DotHedgeData = source_h_data.into();
                let mut sink_data: DotHedgeData = sink_h_data.into();

                if let Some(sink) = sink_id {
                    sink_data = sink_data.with_id(sink)
                }
                if let Some(source) = source_id {
                    source_data = source_data.with_id(source)
                }
                (
                    dot_edge,
                    source.add_data(source_data),
                    Either::Left(target.add_data(sink_data)),
                )
            }
            (NodeIdOrDangling::Id(source), NodeIdOrDangling::Dangling { statements: states }) => {
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                let dot_edge = statements.into_iter().collect();

                orientation = orientation.relative_to(Flow::Sink);

                let mut sink_data: DotHedgeData = sink_h_data.into();

                if let Some(sink) = source_id {
                    sink_data = sink_data.with_id(sink)
                }

                if sink_id.is_some() {
                    panic!("Sink edge cannot have a source id");
                }
                (
                    dot_edge,
                    source.add_data(sink_data),
                    Either::Right(Flow::Source),
                )
            }
            (NodeIdOrDangling::Dangling { statements: states }, NodeIdOrDangling::Id(source)) => {
                statements.extend(
                    states
                        .into_iter()
                        .filter(|(a, _)| !(a.as_str() == "shape" || a.as_str() == "label")),
                );
                let dot_edge = statements.into_iter().collect();

                orientation = orientation.relative_to(Flow::Source);

                let mut source_data: DotHedgeData = source_h_data.into();

                if let Some(source) = sink_id {
                    source_data = source_data.with_id(source)
                }
                if source_id.is_some() {
                    panic!("Source edge cannot have a sink id");
                }
                (
                    dot_edge,
                    source.add_data(source_data),
                    Either::Right(Flow::Sink),
                )
            }
            _ => panic!("Cannot connect an edge to two external nodes"),
        };

        (edge, orientation, source, target)
    }
}
