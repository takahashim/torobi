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
    #[allow(dead_code)]
    pub objective: Option<Graph>,
}

#[derive(Deserialize)]
pub struct Graph {
    pub inputs: Vec<InputSpec>,
    pub parameters: Vec<ParameterSpec>,
    pub nodes: Vec<NodeSpec>,
    pub outputs: Vec<String>,
}

#[derive(Deserialize)]
pub struct InputSpec {
    pub name: String,
}

#[derive(Deserialize)]
pub struct ParameterSpec {
    pub path: String,
    pub shape: Vec<i32>,
    /// Read but not yet honoured: M1 differentiates every parameter.
    #[allow(dead_code)]
    pub trainable: bool,
}

#[derive(Deserialize)]
pub struct NodeSpec {
    pub id: usize,
    pub op: String,
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
