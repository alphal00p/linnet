use std::fmt::Display;

use crate::half_edge::involution::{Flow, Hedge};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DotHedgeData {
    pub statement: Option<String>,
    pub id: Option<Hedge>,
}

impl DotHedgeData {
    #[allow(clippy::useless_asref)]
    pub fn dot_serialize(&self) -> Option<String> {
        let mut out = String::new();
        let mut info = false;
        if let Some(statement) = &self.statement {
            info = true;
            out.push_str(statement);
        }

        if let Some(id) = &self.id {
            info = true;
            out.push_str(&format!(" [id={id}]"));
        }
        if info {
            Some(out)
        } else {
            None
        }
    }

    pub fn with_id(mut self, id: Hedge) -> Self {
        self.id = Some(id);
        self
    }
}

impl Display for DotHedgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(statement) = &self.statement {
            write!(f, "{statement}")?;
        }
        Ok(())
    }
}

impl From<Option<String>> for DotHedgeData {
    fn from(statement: Option<String>) -> Self {
        DotHedgeData {
            statement,
            id: None,
        }
    }
}

pub enum ParsingHedgePair {
    Unpaired {
        hedge: Option<Hedge>,
        flow: Flow,
        data: DotHedgeData,
    },
    Paired {
        source: Option<Hedge>,
        source_data: DotHedgeData,
        sink: Option<Hedge>,
        sink_data: DotHedgeData,
    },
}
