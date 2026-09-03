//! The interpreter: one pass over a graph's nodes, dispatching to mlx-rs.
//! M1 implements only what the linear-regression spike needs; everything
//! else fails loudly by name. Runs inside value_and_grad, so errors are
//! mlx exceptions.

use std::collections::BTreeMap;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::graph::{parse_ref, Graph, Ref};

type Result<T> = std::result::Result<T, Exception>;

/// What a tap asks for: a node's name, and how much of it to bring back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stat {
    /// The whole tensor. Honest and expensive; for debugging.
    Full,
    Mean,
    /// The L2 norm, which is what a gradient or an activation is usually
    /// watched by.
    Norm,
    /// Smallest and largest, as a two-element tensor.
    Extent,
}

impl Stat {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "full" => Some(Stat::Full),
            "mean" => Some(Stat::Mean),
            "norm" => Some(Stat::Norm),
            "extent" => Some(Stat::Extent),
            _ => None,
        }
    }

    /// Reduces on the device, so a tap that is left on costs a scalar per
    /// step rather than a tensor (docs/plan.md section 8.3).
    fn apply(self, value: &Array) -> Result<Array> {
        match self {
            Stat::Full => Ok(value.clone()),
            Stat::Mean => value.mean(false),
            Stat::Norm => value.square()?.sum(false)?.sqrt(),
            Stat::Extent => {
                let min = value.min(false)?;
                let max = value.max(false)?;
                mlx_rs::ops::stack(&[min, max])
            }
        }
    }
}

/// Evaluates one graph and returns its outputs by name.
///
/// `rng` is both the randomness and the mode. A key means this is a
/// training pass and the random ops draw; no key means it is not, and they
/// stand aside (dropout becomes the identity). Inference has no other
/// meaning here, and a flag beside the key could disagree with it.
///
/// Collects the values `taps` asks for as it goes.
///
/// A tap is read-only: it adds an output, it does not change one. What it
/// costs is that the value must be kept rather than fused away, which is
/// why a standing tap should reduce (docs/plan.md section 8.3).
pub fn evaluate_tapped(
    graph: &Graph,
    params: &[Array],
    inputs: &BTreeMap<String, Array>,
    rng: Option<&Array>,
    taps: &BTreeMap<String, Stat>,
    collected: &mut BTreeMap<String, Array>,
) -> Result<BTreeMap<String, Array>> {
    // One key per graph, split per dropout node, so two dropouts in one
    // step draw different masks and the same step always draws the same
    // ones.
    let mut key = rng.cloned();
    let mut values: Vec<Array> = Vec::with_capacity(graph.nodes.len());
    let resolve = |text: &str, values: &Vec<Array>| -> Result<Array> {
        match parse_ref(text).map_err(|e| Exception::custom(e.to_string()))? {
            Ref::Input(id) => {
                let name = &graph.inputs[id].name;
                inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Exception::custom(format!("no binding for input {name:?}")))
            }
            Ref::Node(id) => Ok(values[id].clone()),
        }
    };

    for node in &graph.nodes {
        let ins: Vec<Array> = node
            .inputs
            .iter()
            .map(|r| resolve(r, &values))
            .collect::<Result<_>>()?;
        let fail = |what: &str| Exception::custom(format!("node {} ({}): {what}", node.id, &node.op));

        let out = match node.op.as_str() {
            "parameter" => params[node.parameters[0]].clone(),
            "transpose" => {
                let axes = int_list(node, "axes")?;
                ins[0].transpose_axes(&axes)?
            }
            "matmul" => ins[0].matmul(&ins[1])?,
            "add" => ins[0].add(&ins[1])?,
            "sub" => ins[0].subtract(&ins[1])?,
            "mul" => ins[0].multiply(&ins[1])?,
            "div" => ins[0].divide(&ins[1])?,
            "add_scalar" => ins[0].add(Array::from_f32(number(node, "value")?))?,
            "sub_scalar" => ins[0].subtract(Array::from_f32(number(node, "value")?))?,
            "mul_scalar" => ins[0].multiply(Array::from_f32(number(node, "value")?))?,
            "div_scalar" => ins[0].divide(Array::from_f32(number(node, "value")?))?,
            "square" => ins[0].square()?,
            // An embedding: rows of the table, selected by i32 ids. The
            // gradient reaches only the rows that were read.
            "take" => ins[0].take_axis(&ins[1], 0)?,
            "stop_gradient" => mlx_rs::stop_gradient(&ins[0])?,
            "dropout" => {
                let p = number(node, "p")?;
                if !(0.0..1.0).contains(&p) {
                    return Err(fail(&format!("p must be in 0..1, got {p}")));
                }
                // No key means this is not a training pass. Inverted
                // dropout already scales what survives, so standing aside
                // is the whole of what inference has to do.
                if p == 0.0 || key.is_none() {
                    ins[0].clone()
                } else {
                    let current = key.as_ref().expect("just checked");
                    let (next, draw) = mlx_rs::random::split(current, 2)?;
                    key = Some(next);
                    let keep = Array::from_f32(1.0 - p);
                    // Inverted dropout: scale what survives, so inference
                    // needs no correction.
                    let mask = mlx_rs::random::bernoulli(&keep, ins[0].shape(), &draw)?;
                    ins[0]
                        .multiply(mask.as_dtype(mlx_rs::Dtype::Float32)?)?
                        .divide(&keep)?
                }
            }
            "mean" => {
                if !node.attributes.get("axes").is_none_or(|v| v.is_null()) {
                    return Err(fail("mean over specific axes is not implemented in M1"));
                }
                ins[0].mean(false)?
            }
            other => return Err(fail(&format!("op {other:?} is not implemented in M1"))),
        };
        if let Some(name) = &node.name {
            if let Some(stat) = taps.get(name) {
                collected.insert(name.clone(), stat.apply(&out).map_err(|e| fail(&e.to_string()))?);
            }
        }
        values.push(out);
    }

    graph
        .outputs
        .iter()
        .map(|(name, r)| Ok((name.clone(), resolve(r, &values)?)))
        .collect()
}

fn number(node: &crate::graph::NodeSpec, key: &str) -> Result<f32> {
    node.attributes
        .get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .ok_or_else(|| Exception::custom(format!("{}: attribute {key:?} must be a number", node.op)))
}

fn int_list(node: &crate::graph::NodeSpec, key: &str) -> Result<Vec<i32>> {
    node.attributes
        .get(key)
        .and_then(|v| v.as_array())
        .map(|xs| xs.iter().filter_map(|x| x.as_i64().map(|i| i as i32)).collect())
        .ok_or_else(|| Exception::custom(format!("{}: attribute {key:?} must be an int list", node.op)))
}
