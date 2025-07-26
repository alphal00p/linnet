use std::{collections::BTreeMap, fmt::Display};

use dot_parser::ast::AttrStmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalData {
    pub statements: BTreeMap<String, String>,
    pub edge_statements: BTreeMap<String, String>,
    pub node_statements: BTreeMap<String, String>,
}

impl Display for GlobalData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.statements.is_empty() {
            write!(f, "graph\t [")?;
            let mut first = true;
            for (key, value) in &self.statements {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{key} = {value}")?;
                first = false;
            }
            writeln!(f, "]")?;
        }

        if !self.edge_statements.is_empty() {
            write!(f, "edge\t [")?;
            let mut first = true;
            for (key, value) in &self.edge_statements {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{key} = {value}")?;
                first = false;
            }
            writeln!(f, "]")?;
        }

        if !self.node_statements.is_empty() {
            write!(f, "node\t [")?;
            let mut first = true;
            for (key, value) in &self.node_statements {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{key} = {value}")?;
                first = false;
            }
            writeln!(f, "]")?;
        }

        Ok(())
    }
}

impl TryFrom<Vec<AttrStmt<(String, String)>>> for GlobalData {
    type Error = ();

    fn try_from(value: Vec<AttrStmt<(String, String)>>) -> Result<Self, Self::Error> {
        let mut statements = BTreeMap::new();
        let mut edge_statements = BTreeMap::new();
        let mut node_statements = BTreeMap::new();

        for attr_stmt in value {
            match attr_stmt {
                AttrStmt::Graph(l) => {
                    for l in l.elems {
                        statements.extend(l);
                    }
                }
                AttrStmt::Node(l) => {
                    for l in l.elems {
                        node_statements.extend(l);
                    }
                }
                AttrStmt::Edge(l) => {
                    for l in l.elems {
                        edge_statements.extend(l);
                    }
                }
            }
        }

        Ok(GlobalData {
            statements,
            edge_statements,
            node_statements,
        })
    }
}
