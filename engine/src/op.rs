//! The operation vocabulary, resolved once.
//!
//! A GraphConfig arrives as JSON: an op is a string, its attributes are a
//! `serde_json::Map`, and its inputs are strings like "node:7". None of
//! that changes while a run goes, so none of it should be read while one
//! does. [`Program::resolve`] turns a graph into values the interpreter can
//! match on, and does it when the plan opens.
//!
//! What that buys is not only speed. **A graph that cannot run fails when
//! it is opened, not on its first step**: a missing attribute, an
//! attribute of the wrong type, a reference to a node that does not exist
//! or has not been computed yet, an op given the wrong number of inputs.
//! Those used to surface from inside `value_and_grad`, as MLX exceptions,
//! at step one.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::graph::{parse_ref, Graph, InputSpec, Ref};

/// One node's operation, with its attributes already read.
///
/// The list is `config/ops.yml`, which is the single source of truth for
/// the vocabulary; this is the same list in the form the interpreter
/// dispatches on.
#[derive(Debug)]
pub enum Op {
    /// The model's parameter at this position in its own slice.
    Parameter(usize),

    Add,
    Sub,
    Mul,
    Div,
    AddScalar(f32),
    SubScalar(f32),
    MulScalar(f32),
    DivScalar(f32),

    Neg,
    Abs,
    Sqrt,
    Square,
    Exp,
    Log,
    Gelu,
    GeluTanh,
    Relu,
    Sigmoid,
    Tanh,

    /// The only op that consumes randomness (docs/plan.md section 11.1).
    Dropout(f32),
    StopGradient,

    Softmax { axis: i32 },
    Rope { theta: f32 },

    Transpose(Vec<i32>),
    Reshape(Vec<i32>),
    Slice { axis: i32, start: i32, length: i32 },
    Mean { axes: Option<Vec<i32>>, keepdims: bool },
    Sum { axes: Option<Vec<i32>>, keepdims: bool },
    Max { axes: Option<Vec<i32>>, keepdims: bool },

    Cast(String),
    Matmul,
    Take,

    /// Semantic ops, kept whole: the backend decides how to realize them,
    /// and the IR says what they mean.
    LayerNorm { eps: f32 },
    RmsNorm { eps: f32 },
    Sdpa { scale: Option<f32>, causal: bool },
    CrossEntropy,
}

impl Op {
    /// How many value inputs this op takes, as `config/ops.yml` declares.
    /// A range where the manifest gives one.
    fn arity(&self) -> (usize, usize) {
        match self {
            Op::Parameter(_) => (0, 0),
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Matmul
            | Op::Take
            | Op::CrossEntropy => (2, 2),
            Op::LayerNorm { .. } => (2, 3),
            Op::RmsNorm { .. } => (2, 2),
            Op::Sdpa { .. } => (3, 4),
            _ => (1, 1),
        }
    }

    fn resolve(op: &str, attributes: &Map<String, Value>) -> Result<Self> {
        Ok(match op {
            "parameter" => Op::Parameter(0), // filled in by the caller
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            "div" => Op::Div,
            "add_scalar" => Op::AddScalar(number(attributes, "value")?),
            "sub_scalar" => Op::SubScalar(number(attributes, "value")?),
            "mul_scalar" => Op::MulScalar(number(attributes, "value")?),
            "div_scalar" => Op::DivScalar(number(attributes, "value")?),
            "neg" => Op::Neg,
            "abs" => Op::Abs,
            "sqrt" => Op::Sqrt,
            "square" => Op::Square,
            "exp" => Op::Exp,
            "log" => Op::Log,
            "gelu" => Op::Gelu,
            "gelu_tanh" => Op::GeluTanh,
            "relu" => Op::Relu,
            "sigmoid" => Op::Sigmoid,
            "tanh" => Op::Tanh,
            "dropout" => {
                let p = number(attributes, "p")?;
                anyhow::ensure!(
                    (0.0..1.0).contains(&p),
                    "dropout: p must be in 0..1, got {p}"
                );
                Op::Dropout(p)
            }
            "stop_gradient" => Op::StopGradient,
            "softmax" => Op::Softmax {
                axis: integer(attributes, "axis")?,
            },
            "rope" => Op::Rope {
                theta: number(attributes, "theta")?,
            },
            "transpose" => Op::Transpose(integers(attributes, "axes")?),
            "reshape" => {
                let shape = integers(attributes, "shape")?;
                anyhow::ensure!(
                    shape.iter().filter(|d| **d == -1).count() <= 1,
                    "reshape: only one dimension may be -1, got {shape:?}"
                );
                anyhow::ensure!(
                    shape.iter().all(|d| *d > 0 || *d == -1),
                    "reshape: a shape is positive integers and at most one -1, got {shape:?}"
                );
                Op::Reshape(shape)
            }
            "slice" => Op::Slice {
                axis: integer(attributes, "axis")?,
                start: integer(attributes, "start")?,
                length: integer(attributes, "length")?,
            },
            "mean" => Op::Mean {
                axes: optional_integers(attributes, "axes")?,
                keepdims: boolean(attributes, "keepdims"),
            },
            "sum" => Op::Sum {
                axes: optional_integers(attributes, "axes")?,
                keepdims: boolean(attributes, "keepdims"),
            },
            "max" => Op::Max {
                axes: optional_integers(attributes, "axes")?,
                keepdims: boolean(attributes, "keepdims"),
            },
            "cast" => Op::Cast(string(attributes, "dtype")?),
            "matmul" => Op::Matmul,
            "take" => Op::Take,
            "layer_norm" => Op::LayerNorm {
                eps: number(attributes, "eps")?,
            },
            "rms_norm" => Op::RmsNorm {
                eps: number(attributes, "eps")?,
            },
            "sdpa" => Op::Sdpa {
                scale: optional_number(attributes, "scale")?,
                causal: boolean(attributes, "causal"),
            },
            "cross_entropy" => Op::CrossEntropy,
            other => anyhow::bail!(
                "op {other:?} is not in this engine's vocabulary. The manifest is \
                 Ruby's (config/ops.yml) and this list is compiled, so if the \
                 manifest declares it, what is stale is the extension: build it \
                 again (rake compile). Otherwise it is a typo."
            ),
        })
    }
}

/// One resolved node.
pub struct Node {
    pub op: Op,
    /// Where each input comes from, already parsed and checked.
    pub inputs: Vec<Ref>,
    /// A stable name a tap can ask for, or none.
    pub name: Option<String>,
}

/// A graph in the form the interpreter runs.
pub struct Program {
    /// Kept as declared: binding a batch reads the source, shape and dtype.
    pub inputs: Vec<InputSpec>,
    pub nodes: Vec<Node>,
    /// Named outputs, in the order the config declared them.
    pub outputs: Vec<(String, Ref)>,
}

impl Program {
    /// Reads a graph into runnable form, checking everything that can be
    /// checked without data.
    ///
    /// `parameters` is how many the model declares, so a `parameter` node
    /// that names one past the end says so here rather than panicking in
    /// the middle of a step.
    pub fn resolve(graph: Graph, parameters: usize, model: &str) -> Result<Self> {
        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for (position, spec) in graph.nodes.iter().enumerate() {
            let where_ = format!("{model}: node {} ({})", spec.id, spec.op);
            let mut op = Op::resolve(&spec.op, &spec.attributes).context(where_.clone())?;

            if let Op::Parameter(_) = op {
                let index = *spec.parameters.first().with_context(|| {
                    format!("{where_}: a parameter node names no parameter")
                })?;
                anyhow::ensure!(
                    index < parameters,
                    "{where_}: parameter {index} of {parameters}"
                );
                op = Op::Parameter(index);
            }

            let inputs = spec
                .inputs
                .iter()
                .map(|text| {
                    let reference = parse_ref(text).with_context(|| where_.clone())?;
                    check(&reference, position, graph.inputs.len(), &where_)?;
                    Ok(reference)
                })
                .collect::<Result<Vec<_>>>()?;

            let (least, most) = op.arity();
            anyhow::ensure!(
                (least..=most).contains(&inputs.len()),
                "{where_}: {} inputs, but this op takes {}",
                inputs.len(),
                if least == most {
                    least.to_string()
                } else {
                    format!("{least} to {most}")
                }
            );

            nodes.push(Node {
                op,
                inputs,
                name: spec.name.clone(),
            });
        }

        let outputs = graph
            .outputs
            .iter()
            .map(|(name, text)| {
                let where_ = format!("{model}: output {name:?}");
                let reference = parse_ref(text).with_context(|| where_.clone())?;
                check(&reference, nodes.len(), graph.inputs.len(), &where_)?;
                Ok((name.clone(), reference))
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!outputs.is_empty(), "{model}: no outputs");

        Ok(Self {
            inputs: graph.inputs,
            nodes,
            outputs,
        })
    }

    /// Every name a tap could ask for.
    pub fn node_names(&self) -> impl Iterator<Item = &String> {
        self.nodes.iter().filter_map(|n| n.name.as_ref())
    }
}

/// What the objective's own nodes and inputs are named by, since it is a
/// graph without being a model.
pub const OBJECTIVE: &str = "objective";

/// How a tap names a node: the graph it belongs to, then the node.
///
/// The same shape as a parameter path ("student.head.weight"), and for
/// the same reason: two models of one architecture share every name
/// inside them.
pub fn qualified(graph: &str, node: &str) -> String {
    format!("{graph}.{node}")
}

/// That a reference names something that exists, and that a node reference
/// names one already computed.
///
/// The second is what keeps the interpreter's `values[id]` in bounds. A
/// config whose nodes are not in topological order used to be a panic at
/// step one; it is a refusal at open now.
fn check(reference: &Ref, position: usize, inputs: usize, where_: &str) -> Result<()> {
    match *reference {
        Ref::Input(id) => anyhow::ensure!(
            id < inputs,
            "{where_}: reads input {id}, and there are {inputs}"
        ),
        Ref::Node(id) => anyhow::ensure!(
            id < position,
            "{where_}: reads node {id}, which is not computed before it. \
             Nodes run in the order they are declared."
        ),
    }
    Ok(())
}

fn number(attributes: &Map<String, Value>, key: &str) -> Result<f32> {
    attributes
        .get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .with_context(|| format!("attribute {key:?} must be a number"))
}

fn string(attributes: &Map<String, Value>, key: &str) -> Result<String> {
    attributes
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("attribute {key:?} must be a string"))
}

fn optional_number(attributes: &Map<String, Value>, key: &str) -> Result<Option<f32>> {
    match attributes.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => number(attributes, key).map(Some),
    }
}

fn integer(attributes: &Map<String, Value>, key: &str) -> Result<i32> {
    attributes
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .with_context(|| format!("attribute {key:?} must be an integer"))
}

fn integers(attributes: &Map<String, Value>, key: &str) -> Result<Vec<i32>> {
    let list = attributes
        .get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("attribute {key:?} must be a list of integers"))?;
    list.iter()
        .map(|v| {
            v.as_i64()
                .map(|i| i as i32)
                .with_context(|| format!("attribute {key:?} must be a list of integers"))
        })
        .collect()
}

fn optional_integers(attributes: &Map<String, Value>, key: &str) -> Result<Option<Vec<i32>>> {
    match attributes.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => integers(attributes, key).map(Some),
    }
}

fn boolean(attributes: &Map<String, Value>, key: &str) -> bool {
    attributes.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// A resolved graph's outputs, by name, as the caller reads them.
pub type Outputs = BTreeMap<String, mlx_rs::Array>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, NodeSpec};
    use serde_json::json;

    fn node(id: usize, op: &str, inputs: &[&str], attributes: Value) -> NodeSpec {
        serde_json::from_value(json!({
            "id": id,
            "op": op,
            "inputs": inputs,
            "parameters": [],
            "attributes": attributes,
        }))
        .unwrap()
    }

    fn graph(nodes: Vec<NodeSpec>, outputs: &[(&str, &str)]) -> Graph {
        let inputs = json!([{
            "id": 0, "name": "x", "source": {"batch": "x"},
            "shape": [null, 2], "dtype": "f32",
        }]);
        Graph {
            inputs: serde_json::from_value(inputs).unwrap(),
            parameters: Vec::new(),
            nodes,
            outputs: outputs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn refusal(nodes: Vec<NodeSpec>, outputs: &[(&str, &str)], parameters: usize) -> String {
        match Program::resolve(graph(nodes, outputs), parameters, "m") {
            Ok(_) => panic!("this should not have resolved"),
            Err(e) => format!("{e:#}"),
        }
    }

    #[test]
    fn a_graph_resolves_into_values_the_interpreter_matches_on() {
        let program = Program::resolve(
            graph(
                vec![
                    node(0, "square", &["input:0"], json!({})),
                    node(1, "mul_scalar", &["node:0"], json!({"value": 3.0})),
                ],
                &[("loss", "node:1")],
            ),
            0,
            "m",
        )
        .unwrap();

        assert!(matches!(program.nodes[0].op, Op::Square));
        assert!(matches!(program.nodes[1].op, Op::MulScalar(v) if v == 3.0));
        assert_eq!(program.outputs.len(), 1);
    }

    // Each of these used to surface from inside value_and_grad, as an MLX
    // exception, on the first step. They are refusals at open now.

    #[test]
    fn an_op_outside_the_vocabulary_is_named() {
        let e = refusal(
            vec![node(0, "convolve", &["input:0"], json!({}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("convolve"), "{e}");
        assert!(e.contains("vocabulary"), "{e}");
    }

    #[test]
    fn a_missing_attribute_is_named_with_its_node() {
        let e = refusal(
            vec![node(0, "mul_scalar", &["input:0"], json!({}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("node 0 (mul_scalar)"), "{e}");
        assert!(e.contains("\"value\" must be a number"), "{e}");
    }

    #[test]
    fn an_attribute_of_the_wrong_type_is_refused() {
        let e = refusal(
            vec![node(0, "softmax", &["input:0"], json!({"axis": "last"}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("must be an integer"), "{e}");
    }

    #[test]
    fn a_rate_outside_zero_to_one_is_refused_before_a_step_draws_one() {
        let e = refusal(
            vec![node(0, "dropout", &["input:0"], json!({"p": 1.5}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("p must be in 0..1"), "{e}");
    }

    #[test]
    fn an_op_given_the_wrong_number_of_inputs_is_refused() {
        let e = refusal(
            vec![node(0, "matmul", &["input:0"], json!({}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("1 inputs"), "{e}");
        assert!(e.contains("takes 2"), "{e}");
    }

    #[test]
    fn a_reshape_naming_two_free_dimensions_is_refused() {
        let e = refusal(
            vec![node(0, "reshape", &["input:0"], json!({"shape": [-1, -1]}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("only one dimension may be -1"), "{e}");
    }

    #[test]
    fn a_node_that_reads_one_not_yet_computed_is_refused() {
        // The interpreter indexes `values[id]`, which is only in bounds
        // because of this check.
        let e = refusal(
            vec![
                node(0, "square", &["node:1"], json!({})),
                node(1, "square", &["input:0"], json!({})),
            ],
            &[("loss", "node:1")],
            0,
        );
        assert!(e.contains("not computed before it"), "{e}");
    }

    #[test]
    fn a_reference_to_an_input_that_does_not_exist_is_refused() {
        let e = refusal(
            vec![node(0, "square", &["input:9"], json!({}))],
            &[("loss", "node:0")],
            0,
        );
        assert!(e.contains("reads input 9"), "{e}");
    }

    #[test]
    fn a_parameter_node_past_the_end_is_refused() {
        let mut spec = node(0, "parameter", &[], json!({}));
        spec.parameters = vec![3];
        let e = refusal(vec![spec], &[("loss", "node:0")], 2);
        assert!(e.contains("parameter 3 of 2"), "{e}");
    }

    #[test]
    fn a_graph_with_no_outputs_is_refused() {
        let e = refusal(vec![node(0, "square", &["input:0"], json!({}))], &[], 0);
        assert!(e.contains("no outputs"), "{e}");
    }
}
