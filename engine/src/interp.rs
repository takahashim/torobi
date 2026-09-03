//! The interpreter: one pass over a resolved program's nodes, dispatching
//! to mlx-rs.
//!
//! Everything that could be decided without data was decided when the plan
//! opened (`crate::op`): the op is a variant, its attributes are fields,
//! and its inputs are indices that are known to be in range. What is left
//! here is arithmetic.

use std::collections::BTreeMap;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::graph::Ref;
use crate::op::{Node, Op, Program};

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

/// Evaluates one program and returns its outputs by name.
///
/// `rng` is both the randomness and the mode. A key means this is a
/// training pass and the random ops draw; no key means it is not, and they
/// stand aside (dropout becomes the identity). Inference has no other
/// meaning here, and a flag beside the key could disagree with it.
///
/// Collects the values `taps` asks for as it goes. A tap is read-only: it
/// adds an output, it does not change one. What it costs is that the value
/// must be kept rather than fused away, which is why a standing tap should
/// reduce (docs/plan.md section 8.3).
pub fn evaluate_tapped(
    program: &Program,
    params: &[Array],
    inputs: &BTreeMap<String, Array>,
    rng: Option<&Array>,
    taps: &BTreeMap<String, Stat>,
    collected: &mut BTreeMap<String, Array>,
) -> Result<BTreeMap<String, Array>> {
    // One key per random op, split from the graph's, so two dropouts in one
    // step draw different masks and the same step always draws the same
    // ones.
    let mut key = rng.cloned();
    let mut values: Vec<Array> = Vec::with_capacity(program.nodes.len());

    for node in &program.nodes {
        let ins: Vec<Array> = node
            .inputs
            .iter()
            .map(|r| read(r, program, inputs, &values))
            .collect::<Result<_>>()?;
        let out = apply(node, &ins, params, &mut key)?;
        if let Some(name) = &node.name {
            if let Some(stat) = taps.get(name) {
                collected.insert(name.clone(), stat.apply(&out)?);
            }
        }
        values.push(out);
    }

    program
        .outputs
        .iter()
        .map(|(name, r)| Ok((name.clone(), read(r, program, inputs, &values)?)))
        .collect()
}

/// One resolved reference. Both arms were bounds-checked when the program
/// was resolved, so this cannot be out of range.
fn read(
    reference: &Ref,
    program: &Program,
    inputs: &BTreeMap<String, Array>,
    values: &[Array],
) -> Result<Array> {
    match *reference {
        Ref::Input(id) => {
            let name = &program.inputs[id].name;
            inputs
                .get(name)
                .cloned()
                .ok_or_else(|| Exception::custom(format!("no binding for input {name:?}")))
        }
        Ref::Node(id) => Ok(values[id].clone()),
    }
}

/// One node's arithmetic. `key` is the graph's remaining randomness, split
/// by each op that draws.
fn apply(node: &Node, ins: &[Array], params: &[Array], key: &mut Option<Array>) -> Result<Array> {
    let scalar = Array::from_f32;
    Ok(match &node.op {
        Op::Parameter(index) => params[*index].clone(),

        Op::Add => ins[0].add(&ins[1])?,
        Op::Sub => ins[0].subtract(&ins[1])?,
        Op::Mul => ins[0].multiply(&ins[1])?,
        Op::Div => ins[0].divide(&ins[1])?,
        Op::AddScalar(v) => ins[0].add(scalar(*v))?,
        Op::SubScalar(v) => ins[0].subtract(scalar(*v))?,
        Op::MulScalar(v) => ins[0].multiply(scalar(*v))?,
        Op::DivScalar(v) => ins[0].divide(scalar(*v))?,

        Op::Neg => ins[0].negative()?,
        Op::Abs => ins[0].abs()?,
        Op::Sqrt => ins[0].sqrt()?,
        Op::Square => ins[0].square()?,
        Op::Exp => ins[0].exp()?,
        Op::Log => ins[0].log()?,
        Op::Relu => mlx_rs::ops::maximum(&ins[0], Array::from_f32(0.0))?,
        Op::Sigmoid => mlx_rs::ops::sigmoid(&ins[0])?,
        Op::Tanh => mlx_rs::ops::tanh(&ins[0])?,
        // The exact form, not the tanh approximation: erf is what the
        // reference implementations of these models use, and a parity
        // check against them is the point (docs/plan.md 9.2).
        Op::Gelu => {
            let half = ins[0].multiply(scalar(0.5))?;
            let inner = ins[0].divide(scalar(std::f32::consts::SQRT_2))?;
            half.multiply(mlx_rs::ops::erf(&inner)?.add(scalar(1.0))?)?
        }

        Op::Dropout(p) => {
            // No key means this is not a training pass. Inverted dropout
            // already scales what survives, so standing aside is the whole
            // of what inference has to do.
            match key.as_ref() {
                None => ins[0].clone(),
                Some(current) => {
                    let (next, draw) = mlx_rs::random::split(current, 2)?;
                    *key = Some(next);
                    let keep = scalar(1.0 - p);
                    let mask = mlx_rs::random::bernoulli(&keep, ins[0].shape(), &draw)?;
                    ins[0]
                        .multiply(mask.as_dtype(mlx_rs::Dtype::Float32)?)?
                        .divide(&keep)?
                }
            }
        }
        Op::StopGradient => mlx_rs::stop_gradient(&ins[0])?,

        Op::Softmax { axis } => mlx_rs::ops::softmax_axis(&ins[0], *axis, None)?,
        Op::Rope { theta } => rope(&ins[0], *theta)?,

        Op::Transpose(axes) => ins[0].transpose_axes(axes)?,
        Op::Reshape(shape) => ins[0].reshape(shape)?,
        Op::Slice {
            axis,
            start,
            length,
        } => slice(&ins[0], *axis, *start, *length)?,
        Op::Mean { axes, keepdims } => match axes {
            None => ins[0].mean(*keepdims)?,
            Some(axes) => ins[0].mean_axes(axes, *keepdims)?,
        },
        Op::Sum { axes, keepdims } => match axes {
            None => ins[0].sum(*keepdims)?,
            Some(axes) => ins[0].sum_axes(axes, *keepdims)?,
        },

        Op::Matmul => ins[0].matmul(&ins[1])?,
        // An embedding: rows of the table, selected by i32 ids. The
        // gradient reaches only the rows that were read.
        Op::Take => ins[0].take_axis(&ins[1], 0)?,

        Op::LayerNorm { eps } => layer_norm(ins, *eps)?,
        Op::RmsNorm { eps } => rms_norm(&ins[0], &ins[1], *eps)?,
        Op::Sdpa { scale } => sdpa(ins, *scale)?,
    })
}

/// `length` elements of `axis`, starting at `start`.
///
/// Through `take_axis` rather than a strided slice: mlx-rs keeps the
/// latter to itself, and selecting a contiguous run of indices is the same
/// function.
fn slice(x: &Array, axis: i32, start: i32, length: i32) -> Result<Array> {
    let indices: Vec<i32> = (start..start + length).collect();
    x.take_axis(&Array::from_slice(&indices, &[length]), axis)
}

/// Normalize over the last axis, then scale and shift.
///
/// Two or three inputs: the value, the gain, and optionally the bias. Kept
/// whole as a semantic op, so this is what it means rather than how a
/// backend must do it.
fn layer_norm(ins: &[Array], eps: f32) -> Result<Array> {
    let (x, weight) = (&ins[0], &ins[1]);
    let mean = x.mean_axes(&[-1], true)?;
    let centred = x.subtract(&mean)?;
    let variance = centred.square()?.mean_axes(&[-1], true)?;
    let normed = centred.divide(variance.add(Array::from_f32(eps))?.sqrt()?)?;
    let scaled = normed.multiply(weight)?;
    match ins.get(2) {
        Some(bias) => scaled.add(bias),
        None => Ok(scaled),
    }
}

/// Scale by the root mean square of the last axis, then by the gain. No
/// centring and no bias, which is what makes it not layer_norm.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let scale = x
        .square()?
        .mean_axes(&[-1], true)?
        .add(Array::from_f32(eps))?
        .rsqrt()?;
    x.multiply(scale)?.multiply(weight)
}

/// Rotary position embedding over the last axis, at the given base.
///
/// Positions are the second-to-last axis, which is the sequence, and the
/// last axis is split in half: the first half rotates against the second.
fn rope(x: &Array, theta: f32) -> Result<Array> {
    let shape = x.shape().to_vec();
    let rank = shape.len();
    let (positions, width) = (shape[rank - 2], shape[rank - 1]);
    let half = width / 2;

    // angle[p, i] = p / theta^(2i/width). Built on the host: it is
    // positions x half floats, which is nothing beside the matmuls this
    // sits between, and it keeps the device graph to the rotation itself.
    let angles: Vec<f32> = (0..positions)
        .flat_map(|p| {
            (0..half).map(move |i| {
                p as f32 / theta.powf(2.0 * i as f32 / width as f32)
            })
        })
        .collect();
    let angle = Array::from_slice(&angles, &[positions, half]);
    let (cos, sin) = (angle.cos()?, angle.sin()?);

    let first = slice(x, -1, 0, half)?;
    let second = slice(x, -1, half, half)?;
    let rotated_first = first.multiply(&cos)?.subtract(second.multiply(&sin)?)?;
    let rotated_second = second.multiply(&cos)?.add(first.multiply(&sin)?)?;
    mlx_rs::ops::concatenate_axis(&[rotated_first, rotated_second], -1)
}

/// Scaled dot-product attention: softmax(q k^T / sqrt(d) + mask) v.
///
/// Three or four inputs, the fourth being an additive mask. Kept whole so
/// a backend may run a fused kernel instead; what the IR says is the
/// meaning, and the numbers are held to a tolerance rather than to an
/// order of operations (docs/plan.md 9.2).
fn sdpa(ins: &[Array], scale: Option<f32>) -> Result<Array> {
    let (q, k, v) = (&ins[0], &ins[1], &ins[2]);
    let width = *q.shape().last().expect("a query has a last axis");
    let scale = scale.unwrap_or_else(|| 1.0 / (width as f32).sqrt());

    let rank = k.ndim() as i32;
    let mut axes: Vec<i32> = (0..rank).collect();
    axes.swap(rank as usize - 2, rank as usize - 1);
    let scores = q
        .matmul(k.transpose_axes(&axes)?)?
        .multiply(Array::from_f32(scale))?;
    let scores = match ins.get(3) {
        Some(mask) => scores.add(mask)?,
        None => scores,
    };
    mlx_rs::ops::softmax_axis(&scores, -1, None)?.matmul(v)
}

#[cfg(test)]
mod tests {
    use crate::fixtures;
    use crate::plan::Weights;
    use crate::session::Session;
    use crate::tensor::{Batch, Values};

    /// Runs `op` over one batch and returns what its tap saw. The tap is
    /// how a value inside the graph is read without a second graph to read
    /// it with.
    fn output(op: &str, shape: &[i32], attributes: serde_json::Value, inputs: &[&[f32]]) -> Vec<f32> {
        let extra = inputs.len() - 1;
        let (config, weights) = fixtures::one_op(op, serde_json::json!(shape), attributes, extra);
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        session.tap("seen", "full").unwrap();

        let batch: Batch = inputs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                let name = if i == 0 { "x".to_string() } else { format!("v{}", i - 1) };
                fixtures::field(&name, shape, data)
            })
            .collect();
        session.evaluate(&batch).unwrap();

        let seen = session.tapped().unwrap();
        match &seen[0].1.values {
            Values::F32(v) => v.clone(),
            _ => panic!("a tapped value is f32"),
        }
    }

    fn close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{got:?} against {want:?}");
        }
    }

    #[test]
    fn the_elementwise_ops_are_what_they_say() {
        let x = [-1.0f32, 0.0, 0.5, 2.0];
        let shape = [4];
        close(&output("neg", &shape, serde_json::json!({}), &[&x]), &[1.0, 0.0, -0.5, -2.0]);
        close(&output("abs", &shape, serde_json::json!({}), &[&x]), &[1.0, 0.0, 0.5, 2.0]);
        close(
            &output("relu", &shape, serde_json::json!({}), &[&x]),
            &[0.0, 0.0, 0.5, 2.0],
        );
        close(
            &output("exp", &shape, serde_json::json!({}), &[&x]),
            &[(-1.0f32).exp(), 1.0, 0.5f32.exp(), 2.0f32.exp()],
        );
        close(
            &output("tanh", &shape, serde_json::json!({}), &[&x]),
            &x.map(f32::tanh),
        );
        close(
            &output("sigmoid", &shape, serde_json::json!({}), &[&x]),
            &x.map(|v| 1.0 / (1.0 + (-v).exp())),
        );
        close(
            &output("sqrt", &[3], serde_json::json!({}), &[&[1.0, 4.0, 9.0]]),
            &[1.0, 2.0, 3.0],
        );
        close(
            &output("log", &[3], serde_json::json!({}), &[&[1.0, std::f32::consts::E, 10.0]]),
            &[0.0, 1.0, 10.0f32.ln()],
        );
    }

    #[test]
    fn gelu_is_the_exact_form_rather_than_the_tanh_approximation() {
        // 0.5 x (1 + erf(x / sqrt 2)). The reference implementations of
        // the models this has to match use erf, and a parity check is the
        // point (docs/plan.md 9.2).
        let x = [-2.0f32, -0.5, 0.0, 1.0];
        let want: Vec<f32> = x
            .iter()
            .map(|v| {
                // erf through the relation to the normal CDF, computed here
                // by a series good enough for a tolerance of 1e-5.
                let t = v / std::f32::consts::SQRT_2;
                0.5 * v * (1.0 + erf(t))
            })
            .collect();
        close(&output("gelu", &[4], serde_json::json!({}), &[&x]), &want);
    }

    /// Abramowitz and Stegun 7.1.26, accurate to about 1.5e-7.
    fn erf(x: f32) -> f32 {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        sign * y
    }

    #[test]
    fn softmax_sums_to_one_along_its_axis() {
        let got = output("softmax", &[2, 3], serde_json::json!({"axis": -1}),
                         &[&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]);
        let want = {
            let e: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|v| v.exp()).collect();
            let total: f32 = e.iter().sum();
            let mut v: Vec<f32> = e.iter().map(|x| x / total).collect();
            v.extend([1.0 / 3.0; 3]);
            v
        };
        close(&got, &want);
    }

    #[test]
    fn reshape_rearranges_without_moving_anything() {
        // What multi-head attention needs: [seq, heads * head] becomes
        // [seq, heads, head], reading in order.
        let got = output(
            "reshape",
            &[2, 4],
            serde_json::json!({"shape": [2, 2, 2]}),
            &[&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
        );
        close(&got, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn slice_takes_a_run_of_one_axis() {
        let got = output(
            "slice",
            &[2, 4],
            serde_json::json!({"axis": 1, "start": 1, "length": 2}),
            &[&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]],
        );
        close(&got, &[1.0, 2.0, 11.0, 12.0]);
    }

    #[test]
    fn sum_reduces_the_axes_it_is_given() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        close(
            &output("sum", &[2, 2], serde_json::json!({"axes": [1], "keepdims": false}), &[&data]),
            &[3.0, 7.0],
        );
        close(
            &output("sum", &[2, 2], serde_json::json!({"axes": null, "keepdims": false}), &[&data]),
            &[10.0],
        );
    }

    #[test]
    fn layer_norm_centres_and_scales_the_last_axis() {
        // Two rows, gain 1: each row becomes mean 0 and variance 1.
        let got = output(
            "layer_norm",
            &[2, 3],
            serde_json::json!({"eps": 0.0}),
            &[&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        );
        // (x - mean) / sqrt(variance). Both rows come out the same,
        // because normalizing is scale-free and the second row is the
        // first times ten.
        let s = (2.0f32 / 3.0).sqrt();
        let row = [-1.0 / s, 0.0, 1.0 / s];
        close(&got, &[row[0], row[1], row[2], row[0], row[1], row[2]]);
    }

    #[test]
    fn rms_norm_scales_without_centring() {
        // rms of [3, 4] is sqrt((9 + 16) / 2) = 3.5355, and no mean is
        // subtracted: that is what makes it not layer_norm.
        let got = output(
            "rms_norm",
            &[1, 2],
            serde_json::json!({"eps": 0.0}),
            &[&[3.0, 4.0], &[1.0, 1.0]],
        );
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        close(&got, &[3.0 / rms, 4.0 / rms]);
    }

    #[test]
    fn sdpa_with_one_key_returns_the_value_it_attends_to() {
        // One query, one key: softmax over a single score is 1, so the
        // answer is v exactly, whatever q and k are.
        let got = output(
            "sdpa",
            &[1, 2],
            serde_json::json!({"scale": null}),
            &[&[0.3, -0.7], &[1.1, 2.2], &[5.0, 6.0]],
        );
        close(&got, &[5.0, 6.0]);
    }

    #[test]
    fn rope_leaves_the_first_position_alone_and_rotates_the_next() {
        // At position 0 every angle is 0, so the identity. At position 1
        // the first pair rotates by 1 radian (theta^0 = 1).
        let got = output(
            "rope",
            &[2, 2],
            serde_json::json!({"theta": 10000.0}),
            &[&[1.0, 0.0, 1.0, 0.0]],
        );
        close(&got, &[1.0, 0.0, 1.0f32.cos(), 1.0f32.sin()]);
    }
}
