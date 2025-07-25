use std::collections::BTreeMap;

use dot_parser::canonical::AttrStmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalData {
    pub statements: BTreeMap<String, String>,
    pub edge_statements: BTreeMap<String, String>,
    pub node_statements: BTreeMap<String, String>,
}

impl TryFrom<&Vec<dot_parser::canonical::AttrStmt<(String, String)>>> for GlobalData {
    type Error = ();

    fn try_from(
        value: &Vec<dot_parser::canonical::AttrStmt<(String, String)>>,
    ) -> Result<Self, Self::Error> {
        let mut statements = BTreeMap::new();
        let mut edge_statements = BTreeMap::new();
        let mut node_statements = BTreeMap::new();

        for attr_stmt in value {
            match attr_stmt {
                AttrStmt::Graph((k, v)) => {
                    statements.insert(k.clone(), v.clone());
                }
                AttrStmt::Node((k, v)) => {
                    node_statements.insert(k.clone(), v.clone());
                }
                AttrStmt::Edge((k, v)) => {
                    edge_statements.insert(k.clone(), v.clone());
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
