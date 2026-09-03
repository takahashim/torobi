//! The engine's view of a GraphConfig: exactly the canonical JSON the Ruby
//! side emits, deserialized with serde. The engine trusts the structural
//! validation done at build time and re-checks only what execution needs.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct GraphConfig {
    #[allow(dead_code)]
    pub schema_version: u32,
    pub models: BTreeMap<String, Graph>,
    pub objective: Option<Graph>,
    /// Which models are differentiated. Everything else is frozen.
    pub train: Vec<String>,
}

#[derive(Deserialize)]
pub struct Graph {
    pub inputs: Vec<InputSpec>,
    pub parameters: Vec<ParameterSpec>,
    pub nodes: Vec<NodeSpec>,
    /// Named: {"logits": "node:12"}.
    pub outputs: BTreeMap<String, String>,
}

#[derive(Deserialize)]
pub struct InputSpec {
    pub name: String,
    /// Where the data comes from: {"batch": field} or
    /// {"model": name, "output": name}.
    pub source: BTreeMap<String, String>,
    /// A null dimension is symbolic: it may differ from batch to batch.
    pub shape: Vec<Option<i32>>,
    #[allow(dead_code)]
    pub dtype: String,
}

impl InputSpec {
    /// The batch field this input reads, if it reads one.
    pub fn batch_field(&self) -> Option<&str> {
        self.source.get("batch").map(String::as_str)
    }

    /// The model and output this input reads, if it reads one.
    pub fn model_output(&self) -> Option<(&str, &str)> {
        match (self.source.get("model"), self.source.get("output")) {
            (Some(model), Some(output)) => Some((model, output)),
            _ => None,
        }
    }
}

#[derive(Deserialize)]
pub struct ParameterSpec {
    pub path: String,
    pub shape: Vec<i32>,
    pub trainable: bool,
}

#[derive(Deserialize)]
pub struct NodeSpec {
    pub id: usize,
    pub op: String,
    /// A stable path a tap can ask for ("layers.3.attn"), or none.
    pub name: Option<String>,
    pub inputs: Vec<String>,
    pub parameters: Vec<usize>,
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

/// "input:3" -> (Input, 3); "node:7" -> (Node, 7).
pub enum Ref {
    Input(usize),
    Node(usize),
}

pub fn parse_ref(text: &str) -> anyhow::Result<Ref> {
    let (kind, id) = text
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("bad reference {text:?}"))?;
    let id: usize = id.parse()?;
    match kind {
        "input" => Ok(Ref::Input(id)),
        "node" => Ok(Ref::Node(id)),
        other => anyhow::bail!("bad reference kind {other:?}"),
    }
}
