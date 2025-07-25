use std::{borrow::Borrow, collections::BTreeMap, fmt::Display, str::FromStr};

use itertools::Either;

use crate::half_edge::{involution::Flow, NodeIndex};

use super::GlobalData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotVertexData {
    pub id: Option<String>,
    pub index: Option<NodeIndex>,
    pub statements: BTreeMap<String, String>,
}
impl DotVertexData {
    pub fn id(&self) -> Option<&str> {
        if let Some(d) = self.statements.get("id") {
            Some(d.as_str())
        } else {
            self.id.as_deref()
        }
    }

    pub fn format(&self, template: impl AsRef<str>) -> String {
        let mut result = template.as_ref().to_owned();

        // Find all occurrences of {key} in the template
        for (key, value) in &self.statements {
            let placeholder = format!("{{{key}}}");
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

    pub fn empty() -> Self {
        DotVertexData {
            index: None,
            id: None,
            statements: BTreeMap::new(),
        }
    }

    pub fn extend(&mut self, other: Self) {
        self.statements.extend(other.statements);
    }

    pub fn get<Q: Ord + ?Sized, F: FromStr>(&self, key: &Q) -> Option<Result<F, F::Err>>
    where
        String: Borrow<Q>,
    {
        self.statements.get(key).map(|s| s.parse())
    }

    pub fn add_statement(&mut self, key: impl ToString, value: impl ToString) {
        self.statements.insert(key.to_string(), value.to_string());
    }

    pub fn from_parser(
        value: dot_parser::canonical::Node<(String, String)>,
        global: &GlobalData,
    ) -> Either<Self, (Flow, BTreeMap<String, String>)> {
        let mut flow = None;
        let mut index = None;
        let node_statements: BTreeMap<String, String> = value
            .attr
            .into_iter()
            .filter_map(|(key, value)| {
                match key.as_str() {
                    "flow" => match value.as_str() {
                        "source" => flow = Some(Flow::Source),
                        "sink" => flow = Some(Flow::Sink),
                        _ => panic!("Invalid flow"),
                    },
                    "index" => index = Some(NodeIndex(value.parse::<usize>().unwrap())),
                    _ => return Some((key, value)),
                }
                None
            })
            .collect();

        if let Some(flow) = flow {
            Either::Right((flow, node_statements))
        } else {
            let mut statements = global.node_statements.clone();
            statements.extend(node_statements);
            Either::Left(DotVertexData {
                id: Some(value.id),
                index,
                statements,
            })
        }
    }
}

impl Display for DotVertexData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(id) = self.id() {
            write!(f, "id={id},")?;
        }
        for (key, value) in &self.statements {
            if key == "id" {
                continue;
            }
            write!(f, "{key}={value},")?;
        }
        Ok(())
    }
}
