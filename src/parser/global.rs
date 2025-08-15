use std::{collections::BTreeMap, fmt::Display};

use dot_parser::{ast::AttrStmt, canonical::IDEq};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalData {
    pub name: String,
    pub statements: BTreeMap<String, String>,
    pub edge_statements: BTreeMap<String, String>,
    pub node_statements: BTreeMap<String, String>,
}

impl GlobalData {
    pub fn add_name(&mut self, name: String) {
        self.name = name;
    }
}

impl From<()> for GlobalData {
    fn from(_: ()) -> Self {
        GlobalData {
            name: String::new(),
            statements: BTreeMap::new(),
            edge_statements: BTreeMap::new(),
            node_statements: BTreeMap::new(),
        }
    }
}

impl Display for GlobalData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.statements.is_empty() {
            for (key, value) in &self.statements {
                if let Some(indent) = f.width() {
                    writeln!(f, "{}{key} = {value};", vec![" "; indent].join(""))?;
                } else {
                    write!(f, "\n{key} = {value};")?;
                }
            }
        }

        if !self.edge_statements.is_empty() {
            if let Some(indent) = f.width() {
                writeln!(f, "{}edge\t [", vec![" "; indent].join(""))?;
            } else {
                write!(f, "edge\t [")?;
            }

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
            if let Some(indent) = f.width() {
                writeln!(f, "{}node\t [", vec![" "; indent].join(""))?;
            } else {
                write!(f, "node\t [")?;
            }

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

impl TryFrom<(Vec<AttrStmt<(String, String)>>, Vec<IDEq>)> for GlobalData {
    type Error = ();

    fn try_from(value: (Vec<AttrStmt<(String, String)>>, Vec<IDEq>)) -> Result<Self, Self::Error> {
        let mut statements = BTreeMap::new();
        let mut edge_statements = BTreeMap::new();
        let mut node_statements = BTreeMap::new();

        for attr_stmt in value.0 {
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

        for e in value.1 {
            statements.insert(e.lhs, e.rhs);
        }

        let mut name = String::new();
        statements.retain(|k, v| {
            if k.as_str() == "name" {
                name = v.clone();
                false
            } else {
                true
            }
        });

        Ok(GlobalData {
            name,
            statements,
            edge_statements,
            node_statements,
        })
    }
}
