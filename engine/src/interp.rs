//! The interpreter: one pass over a graph's nodes, dispatching to mlx-rs.
//! M1 implements only what the linear-regression spike needs; everything
//! else fails loudly by name. Runs inside value_and_grad, so errors are
//! mlx exceptions.

use std::collections::BTreeMap;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::graph::{parse_ref, Graph, Ref};

type Result<T> = std::result::Result<T, Exception>;

/// Evaluates one graph and returns its outputs by name. `params` is this
/// graph's slice of the run's parameters, in declaration order; `inputs` is
/// keyed by input name (the caller resolved each input's source).
pub fn evaluate(
    graph: &Graph,
    params: &[Array],
    inputs: &BTreeMap<String, Array>,
    rng: Option<&Array>,
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
            "stop_gradient" => mlx_rs::stop_gradient(&ins[0])?,
            "dropout" => {
                let p = number(node, "p")?;
                if !(0.0..1.0).contains(&p) {
                    return Err(fail(&format!("p must be in 0..1, got {p}")));
                }
                if p == 0.0 {
                    ins[0].clone()
                } else {
                    let current = key
                        .as_ref()
                        .ok_or_else(|| fail("dropout needs the session's RNG state"))?;
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
