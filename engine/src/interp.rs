//! The interpreter: one pass over a resolved program's nodes, dispatching
//! to MLX through the engine's own binding (`crate::mlxc`).
//!
//! Everything that could be decided without data was decided when the plan
//! opened (`crate::op`): the op is a variant, its attributes are fields,
//! and its inputs are indices that are known to be in range. What is left
//! here is arithmetic.

use std::collections::BTreeMap;

use crate::mlxc::error::Exception;
use crate::mlxc::fast::ScaledDotProductAttentionMask as Mask;
use crate::mlxc::ops::indexing::take_along_axis;
use crate::mlxc::{Array, Dtype};

use crate::graph::Ref;
use crate::op::{qualified, Node, Op, Program};

type Result<T> = std::result::Result<T, Exception>;

/// What a tap asked for, by node name.
pub type Taps = BTreeMap<String, Stat>;

/// What the taps saw, by node name.
pub type Tapped = BTreeMap<String, Array>;

/// What is being watched, and where what it saw goes.
///
/// The two always travelled together and as two parameters: one to say
/// which nodes are wanted, one to put them in. Two chances to pass a
/// mismatched pair, and the deciding (is anything watching? is this node
/// one of them? reduce it how?) sat inline in the interpreter's innermost
/// loop, which is not what that loop is about.
pub struct Watch<'a> {
    taps: &'a Taps,
    collected: &'a mut Tapped,
}

impl<'a> Watch<'a> {
    pub fn new(taps: &'a Taps, collected: &'a mut Tapped) -> Self {
        Self { taps, collected }
    }

    /// Nothing is watched, whatever is passed in to collect into.
    ///
    /// What a traced pass wants: the values a `value_and_grad` produces
    /// belong to the trace, and reading them out is not what a tap means
    /// (`crate::executor::differentiate`).
    pub fn none(collected: &'a mut Tapped) -> Self {
        static NOTHING: std::sync::OnceLock<Taps> = std::sync::OnceLock::new();
        Self {
            taps: NOTHING.get_or_init(Taps::new),
            collected,
        }
    }

    /// Keeps what a node produced, if that node is one of the watched.
    ///
    /// Nothing is watched most of the time, and this is called once per
    /// node per pass, so the usual answer is the first line.
    fn saw(&mut self, qualifier: &str, name: &str, value: &Array) -> Result<()> {
        if self.taps.is_empty() {
            return Ok(());
        }
        let watched = qualified(qualifier, name);
        let Some(stat) = self.taps.get(&watched) else {
            return Ok(());
        };
        self.collected.insert(watched, stat.apply(value)?);
        Ok(())
    }
}

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
                crate::mlxc::ops::stack(&[min, max], 0)
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
///
/// `qualifier` is which graph this is (a model's name, or "objective"),
/// and a tap names a node through it: "student.hidden" rather than
/// "hidden". Two models of the same architecture have the same node names,
/// which is the ordinary shape of a distillation, so an unqualified tap
/// would report one of them and drop the other without saying so.
pub fn evaluate_tapped(
    program: &Program,
    params: &[Array],
    inputs: &BTreeMap<String, Array>,
    rng: Option<&Array>,
    qualifier: &str,
    watch: &mut Watch<'_>,
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
        // What was computed has the dtype the graph says, or the graph and
        // the engine are describing two different runs. MLX promotes a
        // bf16 array combined with an f32 one to f32 without a word, and
        // that is how a bf16 model came to compute in f32 from its first
        // norm on, while the graph went on saying bf16 (2026-11-09).
        if let Some(declared) = node.dtype {
            if out.dtype() != declared {
                return Err(Exception::custom(format!(
                    "{qualifier}: node {} ({:?}) computed {:?} where the graph declares {:?}. \
                     This is the engine's mistake, not the graph's: an op mixed a dtype into its \
                     arithmetic that the graph did not ask for.",
                    values.len(),
                    node.op,
                    out.dtype(),
                    declared
                )));
            }
        }
        if let Some(name) = &node.name {
            watch.saw(qualifier, name, &out)?;
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
    // A number from the graph, in the dtype of what it is combined with.
    let scalar = |value: f32| constant(value, &ins[0]);
    Ok(match &node.op {
        Op::Parameter(index) => params[*index].clone(),

        Op::Add => ins[0].add(&ins[1])?,
        Op::Sub => ins[0].subtract(&ins[1])?,
        Op::Mul => ins[0].multiply(&ins[1])?,
        Op::Div => ins[0].divide(&ins[1])?,
        Op::AddScalar(v) => ins[0].add(scalar(*v)?)?,
        Op::SubScalar(v) => ins[0].subtract(scalar(*v)?)?,
        Op::MulScalar(v) => ins[0].multiply(scalar(*v)?)?,
        Op::DivScalar(v) => ins[0].divide(scalar(*v)?)?,
        Op::RdivScalar(v) => scalar(*v)?.divide(&ins[0])?,

        Op::Neg => ins[0].negative()?,
        Op::Abs => ins[0].abs()?,
        Op::Sqrt => ins[0].sqrt()?,
        Op::Square => ins[0].square()?,
        Op::Exp => ins[0].exp()?,
        Op::Log => ins[0].log()?,
        Op::Relu => crate::mlxc::ops::maximum(&ins[0], scalar(0.0)?)?,
        Op::Sigmoid => crate::mlxc::ops::sigmoid(&ins[0])?,
        Op::Tanh => crate::mlxc::ops::tanh(&ins[0])?,
        // The exact form, not the tanh approximation: erf is what the
        // reference implementations of these models use, and a parity
        // check against them is the point (docs/plan.md 9.2).
        Op::Gelu => {
            let half = ins[0].multiply(scalar(0.5)?)?;
            let inner = ins[0].divide(scalar(std::f32::consts::SQRT_2)?)?;
            half.multiply(crate::mlxc::ops::erf(&inner)?.add(scalar(1.0)?)?)?
        }
        // And the tanh approximation, which is a different function:
        // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))). Gemma is trained
        // with it, so a graph that reached for the exact one above would
        // be a model that runs and is not the published one.
        Op::GeluTanh => crate::mlxc::nn::gelu_approximate(&ins[0])?,

        Op::Dropout(p) => {
            // No key means this is not a training pass. Inverted dropout
            // already scales what survives, so standing aside is the whole
            // of what inference has to do.
            match key.as_ref() {
                None => ins[0].clone(),
                Some(current) => {
                    let (next, draw) = crate::mlxc::random::split(current, 2)?;
                    *key = Some(next);
                    let chance = Array::from_f32(1.0 - p)?;
                    let mask = crate::mlxc::random::bernoulli(&chance, ins[0].shape(), &draw)?;
                    ins[0]
                        .multiply(mask.as_dtype(ins[0].dtype())?)?
                        .divide(scalar(1.0 - p)?)?
                }
            }
        }
        Op::StopGradient => crate::mlxc::stop_gradient(&ins[0])?,

        Op::Softmax { axis } => crate::mlxc::ops::softmax_axis(&ins[0], *axis, None)?,
        Op::Rope { theta, freqs } => rope(&ins[0], *theta, freqs.as_deref())?,

        Op::Transpose(axes) => ins[0].transpose_axes(axes)?,
        Op::Reshape(shape) => ins[0].reshape(&kept(shape, ins[0].shape())?)?,
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
        Op::Max { axes, keepdims } => match axes {
            None => ins[0].max(*keepdims)?,
            Some(axes) => ins[0].max_axes(axes, *keepdims)?,
        },

        // The one op that changes a dtype. Named rather than inferred:
        // the graph says where a model's precision changes.
        Op::Cast(to) => ins[0].as_dtype(*to)?,
        Op::Matmul => ins[0].matmul(&ins[1])?,
        // The weight is a linear's [d_out, d_in], so it is always read
        // transposed; the IR has no attribute for it (config/ops.yml).
        Op::QuantizedMatmul(quantization) => {
            crate::mlxc::quantized::quantized_matmul(&ins[0], &ins[1], &ins[2], ins.get(3), true, *quantization)?
        }
        // An embedding: rows of the table, selected by i32 ids. The
        // gradient reaches only the rows that were read.
        Op::Take => ins[0].take_axis(&ins[1], 0)?,

        Op::LayerNorm { eps } => layer_norm(ins, *eps)?,
        Op::RmsNorm { eps } => rms_norm(&ins[0], &ins[1], *eps)?,
        Op::Sdpa { scale, causal } => sdpa(ins, *scale, *causal)?,
        Op::CrossEntropy => cross_entropy(&ins[0], &ins[1])?,
    })
}

/// `length` elements of `axis`, starting at `start`.
///
/// Through `take_axis` rather than a strided slice: mlx-rs kept the latter
/// to itself when this was written, and selecting a contiguous run of
/// indices is the same function.
fn slice(x: &Array, axis: i32, start: i32, length: i32) -> Result<Array> {
    let indices: Vec<i32> = (start..start + length).collect();
    x.take_axis(&Array::from_slice(&indices, &[length])?, axis)
}

/// `value` as an array of `like`'s dtype, when that is a floating dtype
/// narrower than f32; as f32 otherwise.
///
/// MLX's arrays carry their dtype strongly: a bf16 array times an f32
/// one is f32, where a Python scalar would have stayed out of the way.
/// So a constant made as f32 and combined with a bf16 value promotes the
/// value, and everything downstream of it, to f32.
fn constant(value: f32, like: &Array) -> Result<Array> {
    let array = Array::from_f32(value)?;
    match like.dtype() {
        Dtype::Bfloat16 | Dtype::Float16 => array.as_dtype(like.dtype()),
        _ => Ok(array),
    }
}

/// `x` in f32, for arithmetic whose precision should not be `x`'s own
/// (a norm's statistics, a rotation's angles).
fn widened(x: &Array) -> Result<Array> {
    if x.dtype() == Dtype::Float32 {
        Ok(x.clone())
    } else {
        x.as_dtype(Dtype::Float32)
    }
}

/// Normalize over the last axis, then scale and shift.
///
/// Two or three inputs: the value, the gain, and optionally the bias. Kept
/// whole as a semantic op, so this is what it means rather than how a
/// backend must do it.
///
/// In f32 whatever `x` is held in, and returned in `x`'s dtype, as MLX's
/// own fused norms do: the mean and variance of a bf16 row are too coarse
/// in bf16, and a result left in f32 would turn the rest of a bf16 model
/// into an f32 one.
fn layer_norm(ins: &[Array], eps: f32) -> Result<Array> {
    let (x, weight) = (widened(&ins[0])?, widened(&ins[1])?);
    let mean = x.mean_axes(&[-1], true)?;
    let centred = x.subtract(&mean)?;
    let variance = centred.square()?.mean_axes(&[-1], true)?;
    let normed = centred.divide(variance.add(Array::from_f32(eps)?)?.sqrt()?)?;
    let scaled = normed.multiply(&weight)?;
    let shifted = match ins.get(2) {
        Some(bias) => scaled.add(widened(bias)?)?,
        None => scaled,
    };
    narrowed(shifted, ins[0].dtype())
}

/// Scale by the root mean square of the last axis, then by the gain. No
/// centring and no bias, which is what makes it not layer_norm. In f32,
/// returned in `x`'s dtype, for the reason `layer_norm` is.
fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let wide = widened(x)?;
    let scale = wide
        .square()?
        .mean_axes(&[-1], true)?
        .add(Array::from_f32(eps)?)?
        .rsqrt()?;
    narrowed(wide.multiply(scale)?.multiply(widened(weight)?)?, x.dtype())
}

/// `x` back in `dtype`, after arithmetic done wider.
fn narrowed(x: Array, dtype: Dtype) -> Result<Array> {
    if x.dtype() == dtype {
        Ok(x)
    } else {
        x.as_dtype(dtype)
    }
}

/// A reshape target with its 0s filled in from the array in hand.
///
/// A 0 says "the dimension this already has", which is how a graph that
/// does not know its batch *or* its sequence still splits the last axis
/// into heads. The Ruby side settles this when the graph is built; the
/// engine resolves it again here because it is handed a shape, not a
/// promise about one.
fn kept(target: &[i32], from: &[i32]) -> Result<Vec<i32>> {
    target
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim != 0 {
                return Ok(dim);
            }
            from.get(axis).copied().ok_or_else(|| {
                Exception::custom(format!(
                    "a reshape to {target:?} keeps axis {axis} of a shape with {} of them",
                    from.len()
                ))
            })
        })
        .collect()
}

/// Rotary position embedding over the last axis, at the given base.
///
/// Positions are the second-to-last axis, which is the sequence, and the
/// last axis is split in half: the first half rotates against the second.
fn rope(x: &Array, theta: f32, freqs: Option<&[f32]>) -> Result<Array> {
    let shape = x.shape().to_vec();
    let rank = shape.len();
    let (positions, width) = (shape[rank - 2], shape[rank - 1]);
    let half = width / 2;

    // angle[p, i] = p / theta^(2i/width), or p * freqs[i] where the model
    // scaled them (Op::Rope). Built on the host: it is positions x half
    // floats, which is nothing beside the matmuls this sits between, and
    // it keeps the device graph to the rotation itself.
    let angles: Vec<f32> = match freqs {
        Some(given) => {
            if given.len() != half as usize {
                return Err(Exception::custom(format!(
                    "rope: {} frequencies for {width} dimensions, which wants {half}",
                    given.len()
                )));
            }
            (0..positions)
                .flat_map(|p| given.iter().map(move |f| p as f32 * f))
                .collect()
        }
        None => (0..positions)
            .flat_map(|p| {
                (0..half).map(move |i| {
                    p as f32 / theta.powf(2.0 * i as f32 / width as f32)
                })
            })
            .collect(),
    };
    let angle = Array::from_slice(&angles, &[positions, half])?;
    let (cos, sin) = (angle.cos()?, angle.sin()?);

    // Rotated in f32 and returned in `x`'s dtype, as MLX's fused rope
    // does: the angles are f32, and multiplying a bf16 `x` by them would
    // otherwise leave the queries and keys in f32.
    let wide = widened(x)?;
    let first = slice(&wide, -1, 0, half)?;
    let second = slice(&wide, -1, half, half)?;
    let rotated_first = first.multiply(&cos)?.subtract(second.multiply(&sin)?)?;
    let rotated_second = second.multiply(&cos)?.add(first.multiply(&sin)?)?;
    narrowed(crate::mlxc::ops::concatenate(&[rotated_first, rotated_second], -1)?, x.dtype())
}

/// Scaled dot-product attention: softmax(q k^T / sqrt(d) + mask) v.
///
/// Three or four inputs, the fourth being an additive mask. Kept whole so
/// a backend may run a fused kernel instead; what the IR says is the
/// meaning, and the numbers are held to a tolerance rather than to an
/// order of operations (docs/plan.md 9.2).
/// Attention, through the backend's own kernel.
///
/// The fused one rather than the decomposition it replaced (docs/plan.md
/// 6.1: a semantic op uses the backend's kernel when there is one). What
/// that buys is not only speed: it takes k and v untiled, which is what
/// makes grouped-query attention expressible at all, and it has a causal
/// mode, so a decoder does not hand over a triangle of the same number
/// every step. It differentiates, which is the thing to check before
/// trusting a fused kernel in a trainer, and there is a test that it does.
fn sdpa(ins: &[Array], scale: Option<f32>, causal: bool) -> Result<Array> {
    let width = *ins[0].shape().last().expect("a query has a last axis");
    let scale = scale.unwrap_or_else(|| 1.0 / (width as f32).sqrt());

    // The kernel takes [batch, heads, positions, head_dim]. A graph that
    // writes attention without a head axis means one head, and saying so
    // is an axis of 1 rather than a different op.
    let flat = ins[0].ndim() == 3;
    let lift = |a: &Array| if flat { a.expand_dims_axes(&[1]) } else { Ok(a.clone()) };
    let (q, k, v) = (lift(&ins[0])?, lift(&ins[1])?, lift(&ins[2])?);
    let held;
    let mask = if causal {
        Some(Mask::Causal)
    } else {
        match ins.get(3) {
            // A mask is broadcast against the scores, so one written for
            // three axes needs the head axis these just grew. And it is in
            // the scores' dtype, as mlx-lm casts it: an f32 mask added to
            // bf16 scores would hand back f32 attention.
            Some(mask) => {
                let lifted = if flat && mask.ndim() == 3 { mask.expand_dims_axes(&[1])? } else { mask.clone() };
                held = narrowed(lifted, q.dtype())?;
                Some(Mask::Array(&held))
            }
            None => None,
        }
    };
    // The last argument is attention sinks, which nothing here uses.
    let out = crate::mlxc::fast::scaled_dot_product_attention(&q, &k, &v, scale, mask, None)?;
    Ok(if flat { out.squeeze_axes(&[1])? } else { out })
}

/// The loss at each position: `logsumexp(logits) - logits[target]`.
///
/// The largest logit comes out before anything is exponentiated, which is
/// what `logsumexp` is; a vocabulary's logits overflow f32 without it.
/// The same two lines as mlx-rs's `CrossEntropy` with no reduction,
/// written here because the reduction is the objective's to choose.
fn cross_entropy(logits: &Array, targets: &Array) -> Result<Array> {
    let wanted = take_along_axis(logits, &targets.expand_dims_axes(&[-1])?, -1)?
        .squeeze_axes(&[-1])?;
    Ok(crate::mlxc::ops::logsumexp_axes(logits, &[-1], None)?.subtract(wanted)?)
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
        session.tap("m.seen", "full").unwrap();

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

    /// A graph built knowing neither its batch nor its sequence still
    /// splits the last axis: the 0s take whatever the array turns out to
    /// have (docs/plan.md 15.63).
    #[test]
    fn a_reshape_keeps_the_dimensions_the_graph_never_knew() {
        let (config, weights) = fixtures::one_op(
            "reshape",
            serde_json::json!([null, null, 4]),
            serde_json::json!({"shape": [0, 0, 2, 2]}),
            0,
        );
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        session.tap("m.seen", "full").unwrap();

        for rows in [3, 7] {
            let batch: Batch = [fixtures::field("x", &[rows, 5, 4], &vec![1.0f32; (rows * 20) as usize])]
                .into_iter()
                .collect();
            session.evaluate(&batch).unwrap();
            let seen = session.tapped().unwrap();
            assert_eq!(seen[0].1.shape, vec![rows, 5, 2, 2]);
        }
    }

    /// `value / x`, and at an infinite x that is 0, not NaN: the reason
    /// it is an op rather than a number spread to x's shape by x * 0.
    #[test]
    fn a_number_divided_by_the_value_is_zero_where_the_value_is_infinite() {
        let got = output(
            "rdiv_scalar",
            &[3],
            serde_json::json!({"value": 2.0}),
            &[&[4.0, -0.5, f32::INFINITY]],
        );
        close(&got, &[0.5, -4.0, 0.0]);
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
        // answer is v exactly, whatever q and k are. Rank 4, which is
        // what attention is: [batch, heads, positions, head_dim].
        let got = output(
            "sdpa",
            &[1, 1, 1, 2],
            serde_json::json!({"scale": null, "causal": false}),
            &[&[0.3, -0.7], &[1.1, 2.2], &[5.0, 6.0]],
        );
        close(&got, &[5.0, 6.0]);
    }

    /// Why the fused kernel is trusted in a trainer.
    ///
    /// A fused op is only usable here if it differentiates, and this is
    /// the check that it does. It is also where grouped-query attention
    /// is answered: four query heads over two key heads, with k and v
    /// untiled, which is the arrangement every decoder since Llama 2 has.
    #[test]
    fn the_fused_attention_differentiates_and_groups_queries() {
        use crate::mlxc::transforms::value_and_grad_with_argnums;
        use crate::mlxc::Array;

        let q = Array::from_slice(&[0.1f32; 24], &[1, 4, 2, 3]).unwrap();
        let k = Array::from_slice(&[0.2f32; 12], &[1, 2, 2, 3]).unwrap();
        let v = Array::from_slice(&[0.3f32; 12], &[1, 2, 2, 3]).unwrap();

        let forward = |args: &[Array]| -> Vec<Array> {
            let out = crate::mlxc::fast::scaled_dot_product_attention(
                &args[0], &args[1], &args[2], 0.5, None, None,
            )
            .expect("fused attention");
            vec![out.sum(false).expect("sum")]
        };
        let mut vg = value_and_grad_with_argnums(forward, &[0, 1, 2][..]);
        let (value, grads) = vg(&[q, k, v][..]).expect("a gradient");

        // Every value is 0.3, so any weighted average of them is 0.3, and
        // there are 4 heads * 2 positions * 3 wide of them.
        assert!((value[0].item::<f32>().unwrap() - 0.3 * 24.0).abs() < 1e-5);
        assert_eq!(grads[0].shape(), &[1, 4, 2, 3]);
        assert_eq!(grads[1].shape(), &[1, 2, 2, 3], "k stays untiled");
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

    /// A scaled rope turns by the frequencies it was handed.
    ///
    /// The base is still there and is still ignored: what a model scaled
    /// its frequencies to is the model's arithmetic, and the engine is
    /// told the answer rather than the formula (`Op::Rope`).
    #[test]
    fn rope_turns_by_the_frequencies_it_is_given_rather_than_the_base() {
        let got = output(
            "rope",
            &[2, 2],
            serde_json::json!({"theta": 10000.0, "freqs": [0.25]}),
            &[&[1.0, 0.0, 1.0, 0.0]],
        );
        // Position 0 is still the identity; position 1 turns by 0.25
        // rather than by the 1 radian the base alone would have given.
        close(&got, &[1.0, 0.0, 0.25f32.cos(), 0.25f32.sin()]);
    }

    #[test]
    fn a_rope_given_the_wrong_number_of_frequencies_says_so() {
        // Four dimensions want two frequencies. A description handing
        // over another count has miscounted its head, and the difference
        // between being told and not being told is a wrong model that
        // trains.
        let (config, weights) = fixtures::one_op(
            "rope",
            serde_json::json!([2, 4]),
            serde_json::json!({"theta": 10000.0, "freqs": [0.25, 0.5, 0.75]}),
            0,
        );
        let mut session = Session::open(&config, Weights::Inline(&weights)).unwrap();
        let batch: Batch = [fixtures::field("x", &[2, 4], &[1.0f32; 8])].into_iter().collect();

        let said = session
            .evaluate(&batch)
            .expect_err("three frequencies for four dimensions")
            .to_string();

        assert!(said.contains("frequencies"), "{said}");
    }

    /// One node over bf16 inputs, by the interpreter alone, with the dtype
    /// the graph declares for it. `params` are the model's parameters, read
    /// by `parameter` nodes ahead of the op.
    fn through(
        op: &str,
        attributes: serde_json::Value,
        inputs: &[(&str, Vec<i32>)],
        params: Vec<crate::mlxc::Array>,
        declared: &str,
    ) -> Result<crate::mlxc::Array, String> {
        use crate::mlxc::{Array, Dtype};
        use serde_json::json;
        crate::runtime::runtime()
            .execute(|| {
                let specs: Vec<_> = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, (name, shape))| fixtures::input(i, name, json!(shape), "bf16"))
                    .collect();
                let parameters: Vec<_> = params
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let mut spec = fixtures::parameter(i, &format!("p{i}"), json!(a.shape()), false);
                        spec["dtype"] = json!("bf16");
                        spec
                    })
                    .collect();
                let mut nodes: Vec<_> = (0..params.len())
                    .map(|i| fixtures::node(i, "parameter", json!([]), json!([i])))
                    .collect();
                let reads: Vec<String> = (0..inputs.len())
                    .map(|i| format!("input:{i}"))
                    .chain((0..params.len()).map(|i| format!("node:{i}")))
                    .collect();
                let mut under_test = fixtures::node(nodes.len(), op, json!(reads), json!([]));
                under_test["attributes"] = attributes.clone();
                under_test["dtype"] = json!(declared);
                nodes.push(under_test);
                let graph: crate::graph::Graph = serde_json::from_value(json!({
                    "inputs": specs, "parameters": parameters, "nodes": nodes,
                    "outputs": {"y": format!("node:{}", nodes.len() - 1)},
                }))
                .unwrap();
                let program = crate::op::Program::resolve(graph, params.len(), "m").unwrap();
                let fed: std::collections::BTreeMap<String, Array> = inputs
                    .iter()
                    .map(|(name, shape)| {
                        let count: i32 = shape.iter().product();
                        let values: Vec<f32> = (0..count).map(|i| (i as f32 * 0.37).sin()).collect();
                        let array = Array::from_slice(&values, shape)?.as_dtype(Dtype::Bfloat16)?;
                        Ok((name.to_string(), array))
                    })
                    .collect::<super::Result<_>>()?;
                let mut collected = super::Tapped::new();
                let mut watch = super::Watch::none(&mut collected);
                Ok(super::evaluate_tapped(&program, &params, &fed, None, "m", &mut watch)
                    .map(|mut out| out.remove("y").unwrap())
                    .map_err(|e| e.what().to_string()))
            })
            .expect("the runtime should let this through")
    }

    fn bf16(shape: &[i32]) -> crate::mlxc::Array {
        crate::mlxc::Array::ones::<f32>(shape).unwrap().as_dtype(crate::mlxc::Dtype::Bfloat16).unwrap()
    }

    /// A bf16 value through any op stays bf16. Every one of these used to
    /// come back f32, because the number it was combined with was made
    /// as an f32 array, and MLX promotes rather than narrows: the graph
    /// said bf16, and the engine computed the rest of the model in f32
    /// (2026-11-09).
    #[test]
    fn a_bf16_value_stays_bf16_through_every_op_that_adds_a_number_of_its_own() {
        use crate::mlxc::Dtype;
        use serde_json::json;
        let row = |name: &'static str| (name, vec![2, 4]);
        let cases: Vec<(&str, serde_json::Value, Vec<(&str, Vec<i32>)>, Vec<crate::mlxc::Array>)> = vec![
            ("add_scalar", json!({"value": 1.0}), vec![row("x")], vec![]),
            ("sub_scalar", json!({"value": 1.0}), vec![row("x")], vec![]),
            ("mul_scalar", json!({"value": 2.0}), vec![row("x")], vec![]),
            ("div_scalar", json!({"value": 2.0}), vec![row("x")], vec![]),
            ("rdiv_scalar", json!({"value": 2.0}), vec![row("x")], vec![]),
            ("relu", json!({}), vec![row("x")], vec![]),
            ("gelu", json!({}), vec![row("x")], vec![]),
            ("gelu_tanh", json!({}), vec![row("x")], vec![]),
            ("rms_norm", json!({"eps": 1e-6}), vec![row("x")], vec![bf16(&[4])]),
            ("layer_norm", json!({"eps": 1e-5}), vec![row("x")], vec![bf16(&[4]), bf16(&[4])]),
            ("rope", json!({"theta": 10000.0}), vec![row("x")], vec![]),
            (
                "sdpa",
                json!({"scale": null, "causal": false}),
                vec![("q", vec![1, 2, 4]), ("k", vec![1, 2, 4]), ("v", vec![1, 2, 4]), ("mask", vec![1, 2, 2])],
                vec![],
            ),
        ];
        for (op, attributes, inputs, params) in cases {
            let out = through(op, attributes, &inputs, params, "bf16").unwrap_or_else(|e| panic!("{op}: {e}"));
            assert_eq!(out.dtype(), Dtype::Bfloat16, "{op}");
        }
    }

    /// The norms take their statistics in f32 and hand back bf16: over a
    /// row of 4096, the result is the f32 answer to bf16's own precision,
    /// which a mean of squares summed in bf16 would not be.
    #[test]
    fn a_norm_of_a_bf16_row_is_computed_wide_and_returned_narrow() {
        use crate::mlxc::Dtype;
        let out = through("rms_norm", serde_json::json!({"eps": 0.0}), &[("x", vec![1, 4096])], vec![bf16(&[4096])], "bf16")
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        let values = crate::runtime::runtime()
            .execute(|| Ok(out.as_dtype(Dtype::Float32)?.as_slice::<f32>()?.to_vec()))
            .unwrap();
        // The input is sin(0.37 i) held in bf16; the answer is computed
        // here in f32 from the same bf16 values.
        let rms = ((0..4096).map(|i| (i as f32 * 0.37).sin()).map(|v| {
            let b = half_round(v);
            b * b
        }).sum::<f32>() / 4096.0).sqrt();
        for (i, got) in values.iter().enumerate() {
            let want = half_round((i as f32 * 0.37).sin()) / rms;
            assert!((got - want).abs() <= 1e-2 * (1.0 + want.abs()), "{i}: {got} for {want}");
        }
    }

    /// bf16 by truncating an f32 to its top sixteen bits, rounded to nearest.
    fn half_round(v: f32) -> f32 {
        let bits = v.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
        f32::from_bits(rounded)
    }

    /// A node that computes another dtype than the graph declares is
    /// refused where it happens, naming the node and both dtypes, rather
    /// than carrying on as a different run from the one described.
    #[test]
    fn a_node_that_computes_another_dtype_than_declared_is_refused() {
        let said = through("mul_scalar", serde_json::json!({"value": 2.0}), &[("x", vec![2, 4])], vec![], "f32")
            .expect_err("a bf16 product declared f32");
        assert!(said.contains("computed Bfloat16 where the graph declares Float32"), "{said}");
        assert!(said.contains("node 0"), "{said}");
    }
}

