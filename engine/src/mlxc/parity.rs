//! The shim against mlx-rs, on the same inputs, while both are linked.
//!
//! Temporary by construction: it exists to show the calls chosen here are
//! the calls mlx-rs makes, and it goes when mlx-rs does (plan Phase 4).
//! What holds the numbers after that is what held them before: the
//! oracles, the closed-form check, and the recorded draws in `random`.
//!
//! Exact equality, not a tolerance. The same MLX computes both sides, so
//! any difference is a different call, which is what this is looking for.

use crate::mlxc;
use crate::runtime;

/// Values in, as both bindings take them, and values out as f32.
fn both(
    data: &[f32],
    shape: &[i32],
    shim: impl FnOnce(&mlxc::Array) -> mlxc::error::Result<mlxc::Array>,
    reference: impl FnOnce(&mlx_rs::Array) -> mlx_rs::error::Result<mlx_rs::Array>,
) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>) {
    runtime::runtime()
        .execute(|| {
            let ours = shim(&mlxc::Array::from_slice(data, shape))?
                .as_dtype(mlxc::Dtype::Float32)?
                .contiguous()?;
            let theirs = reference(&mlx_rs::Array::from_slice(data, shape))?
                .as_dtype(mlx_rs::Dtype::Float32)?
                .contiguous()?;
            Ok((
                ours.as_slice::<f32>().to_vec(),
                theirs.as_slice::<f32>().to_vec(),
                ours.shape().to_vec(),
                theirs.shape().to_vec(),
            ))
        })
        .expect("the runtime should let this through")
}

fn same(what: &str, (ours, theirs, our_shape, their_shape): (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>)) {
    assert_eq!(our_shape, their_shape, "{what}: shapes");
    let ours: Vec<u32> = ours.iter().map(|v| v.to_bits()).collect();
    let theirs: Vec<u32> = theirs.iter().map(|v| v.to_bits()).collect();
    assert_eq!(ours, theirs, "{what}: values");
}

fn wave(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect()
}

#[test]
fn the_free_functions_agree() {
    let x = wave(24);
    let s = [2, 3, 4];
    same("tanh", both(&x, &s, |a| mlxc::ops::tanh(a), |a| mlx_rs::ops::tanh(a)));
    same("sigmoid", both(&x, &s, |a| mlxc::ops::sigmoid(a), |a| mlx_rs::ops::sigmoid(a)));
    same("erf", both(&x, &s, |a| mlxc::ops::erf(a), |a| mlx_rs::ops::erf(a)));
    same("stop_gradient", both(&x, &s, |a| mlxc::stop_gradient(a), |a| mlx_rs::stop_gradient(a)));
    same("zeros_like", both(&x, &s, |a| mlxc::ops::zeros_like(a), |a| mlx_rs::ops::zeros_like(a)));
    same(
        "relu",
        both(&x, &s, |a| mlxc::ops::maximum(a, mlxc::Array::from_f32(0.0)), |a| {
            mlx_rs::ops::maximum(a, mlx_rs::Array::from_f32(0.0))
        }),
    );
    same(
        "softmax",
        both(&x, &s, |a| mlxc::ops::softmax_axis(a, -1, None), |a| mlx_rs::ops::softmax_axis(a, -1, None)),
    );
    same(
        "logsumexp",
        both(&x, &s, |a| mlxc::ops::logsumexp_axes(a, &[-1], None), |a| {
            mlx_rs::ops::logsumexp_axes(a, &[-1], None)
        }),
    );
    same(
        "concatenate",
        both(&x, &s, |a| mlxc::ops::concatenate(&[a, a], -1), |a| mlx_rs::ops::concatenate(&[a, a], -1)),
    );
    same("stack", both(&x, &s, |a| mlxc::ops::stack(&[a, a], 0), |a| mlx_rs::ops::stack(&[a, a], 0)));
    same(
        "gelu_approximate",
        both(&x, &s, |a| mlxc::nn::gelu_approximate(a), |a| mlx_rs::nn::gelu_approximate(a)),
    );
    // mlx-rs compiles this one and the shim does not, so it is checked in
    // the precision a published model may run it in as well.
    same(
        "gelu_approximate in bf16",
        both(
            &x,
            &s,
            |a| mlxc::nn::gelu_approximate(a.as_dtype(mlxc::Dtype::Bfloat16)?),
            |a| mlx_rs::nn::gelu_approximate(a.as_dtype(mlx_rs::Dtype::Bfloat16)?),
        ),
    );
}

#[test]
fn take_along_axis_agrees() {
    let x = wave(12);
    same(
        "take_along_axis",
        both(
            &x,
            &[3, 4],
            |a| {
                let at = mlxc::Array::from_slice(&[3i32, 0, 2], &[3, 1]);
                mlxc::ops::indexing::take_along_axis(a, &at, -1)
            },
            |a| {
                let at = mlx_rs::Array::from_slice(&[3i32, 0, 2], &[3, 1]);
                mlx_rs::ops::indexing::take_along_axis(a, &at, -1)
            },
        ),
    );
}

#[test]
fn attention_agrees_with_each_kind_of_mask() {
    // [batch 1, heads 2, positions 3, width 4]; k is q scaled, and the
    // additive mask blanks one score per row, so a mask that is ignored
    // shows in the output.
    let q = wave(24);
    let shape = [1, 2, 3, 4];
    let additive: Vec<f32> = (0..9).map(|i| if i % 4 == 0 { -1e9 } else { 0.0 }).collect();
    for kind in ["none", "array", "causal"] {
        let ours = |a: &mlxc::Array| {
            let k = a.multiply(mlxc::Array::from_f32(0.5))?;
            let mask = mlxc::Array::from_slice(&additive, &[1, 1, 3, 3]);
            let mode = match kind {
                "none" => None,
                "array" => Some(mlxc::fast::ScaledDotProductAttentionMask::Array(&mask)),
                _ => Some(mlxc::fast::ScaledDotProductAttentionMask::Causal),
            };
            mlxc::fast::scaled_dot_product_attention(a, &k, a, 0.5, mode, None)
        };
        let theirs = |a: &mlx_rs::Array| {
            let k = a.multiply(mlx_rs::Array::from_f32(0.5))?;
            let mask = mlx_rs::Array::from_slice(&additive, &[1, 1, 3, 3]);
            let mode = match kind {
                "none" => None,
                "array" => Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(&mask)),
                _ => Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal),
            };
            mlx_rs::fast::scaled_dot_product_attention(a, &k, a, 0.5, mode, None)
        };
        same(&format!("sdpa, mask {kind}"), both(&q, &shape, ours, theirs));
    }
}

/// The gradient through the fused kernel, which is what training uses.
#[test]
fn the_gradient_through_attention_agrees() {
    let q = wave(24);
    let shape = [1, 2, 3, 4];
    let ours = |a: &mlxc::Array| {
        let f = |args: &[mlxc::Array]| -> mlxc::error::Result<Vec<mlxc::Array>> {
            let out = mlxc::fast::scaled_dot_product_attention(
                &args[0],
                &args[0],
                &args[0],
                0.5,
                mlxc::fast::ScaledDotProductAttentionMask::Causal,
                None,
            )?;
            Ok(vec![out.sum(false)?])
        };
        let (_, grads) = mlxc::transforms::value_and_grad_with_argnums(f, &[0])(&[a.clone()])?;
        Ok(grads[0].clone())
    };
    let theirs = |a: &mlx_rs::Array| {
        let f = |args: &[mlx_rs::Array]| -> mlx_rs::error::Result<Vec<mlx_rs::Array>> {
            let out = mlx_rs::fast::scaled_dot_product_attention(
                &args[0],
                &args[0],
                &args[0],
                0.5,
                mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
                None,
            )?;
            Ok(vec![out.sum(false)?])
        };
        let (_, grads) = mlx_rs::transforms::value_and_grad_with_argnums(f, &[0])(&[a.clone()])?;
        Ok(grads[0].clone())
    };
    same("d sdpa / d q", both(&q, &shape, ours, theirs));
}

/// Files written by one binding load in the other, both ways, because a
/// checkpoint written before the move has to resume after it.
#[test]
fn safetensors_cross_between_the_bindings() {
    let dir = tempfile::tempdir().unwrap();
    let (from_theirs, from_ours) = runtime::runtime()
        .execute(|| {
            let theirs = dir.path().join("theirs.safetensors");
            let ours = dir.path().join("ours.safetensors");
            let data = wave(6);
            mlx_rs::Array::save_safetensors(
                [("w", mlx_rs::Array::from_slice(&data, &[2, 3]))],
                None,
                &theirs,
            )?;
            mlxc::Array::save_safetensors([("w", mlxc::Array::from_slice(&data, &[2, 3]))], None, &ours)?;
            let read_ours = mlxc::Array::load_safetensors(&theirs)?;
            let read_theirs = mlx_rs::Array::load_safetensors(&ours)?;
            Ok((
                (read_ours["w"].shape().to_vec(), read_ours["w"].as_slice::<f32>().to_vec()),
                (read_theirs["w"].shape().to_vec(), read_theirs["w"].as_slice::<f32>().to_vec()),
            ))
        })
        .expect("the runtime should let this through");
    assert_eq!(from_theirs, (vec![2, 3], wave(6)));
    assert_eq!(from_ours, (vec![2, 3], wave(6)));
}
