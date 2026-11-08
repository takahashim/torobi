//! Small graphs for the engine's own tests.
//!
//! The Ruby side builds a GraphConfig through the DSL; a Rust test wants
//! one without that machinery, and wants it small enough to reason about.
//! These are hand-written, which is also the point: they pin the JSON the
//! engine actually accepts, independently of what the DSL happens to emit.

use serde_json::{json, Value};

/// An input reading a batch field.
pub fn input(id: usize, name: &str, shape: Value, dtype: &str) -> Value {
    json!({
        "id": id,
        "name": name,
        "source": {"batch": name},
        "shape": shape,
        "dtype": dtype,
    })
}

/// An input reading another model's output.
pub fn from_model(id: usize, name: &str, model: &str, output: &str, shape: Value) -> Value {
    json!({
        "id": id,
        "name": name,
        "source": {"model": model, "output": output},
        "shape": shape,
        "dtype": "f32",
    })
}

pub fn parameter(id: usize, path: &str, shape: Value, trainable: bool) -> Value {
    json!({
        "id": id,
        "path": path,
        "shape": shape,
        "dtype": "f32",
        "initializer": {"type": "zeros"},
        "trainable": trainable,
    })
}

pub fn node(id: usize, op: &str, inputs: Value, parameters: Value) -> Value {
    json!({
        "id": id,
        "op": op,
        "inputs": inputs,
        "parameters": parameters,
        "attributes": {},
    })
}

/// The same, with a name a tap can ask for.
pub fn named(id: usize, op: &str, inputs: Value, parameters: Value, name: &str) -> Value {
    let mut node = node(id, op, inputs, parameters);
    node["name"] = json!(name);
    node
}

pub fn config(models: Value, objective: Value, train: Value) -> String {
    json!({
        "schema_version": 1,
        "semantics_version": 3,
        "models": models,
        "objective": objective,
        "train": train,
    })
    .to_string()
}

/// `loss = mean(x * w)`, with `w` a two-element trainable parameter.
///
/// Small enough that its gradient is arithmetic: d(loss)/dw is the column
/// mean of x, so a test can say what a step must produce.
pub fn scaled_mean() -> (String, String) {
    let graph = json!({
        "inputs": [input(0, "x", json!([null, 2]), "f32")],
        "parameters": [parameter(0, "w", json!([2]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            named(1, "mul", json!(["input:0", "node:0"]), json!([]), "scaled"),
            node(2, "mean", json!(["node:1"]), json!([])),
        ],
        "outputs": {"loss": "node:2"},
    });
    let config = config(json!({"m": graph}), Value::Null, json!(["m"]));
    let weights = json!({"params": {"m.w": {"shape": [2], "data": [1.0, 2.0]}}}).to_string();
    (config, weights)
}

/// Two models and an objective over them: a frozen teacher, a trained
/// student, and a loss that reads both. What the parameter ordering, the
/// freeze window and the checkpoint contract are all about.
pub fn teacher_and_student() -> (String, String) {
    let body = |path: &str| {
        json!({
            "inputs": [input(0, "x", json!([null, 2]), "f32")],
            "parameters": [parameter(0, path, json!([2]), true)],
            "nodes": [
                node(0, "parameter", json!([]), json!([0])),
                // Named, and named the same in both models, which is what
                // two models of one architecture look like.
                named(1, "mul", json!(["input:0", "node:0"]), json!([]), "scaled"),
            ],
            "outputs": {"out": "node:1"},
        })
    };
    let objective = json!({
        "inputs": [
            from_model(0, "s", "student", "out", json!([null, 2])),
            from_model(1, "t", "teacher", "out", json!([null, 2])),
        ],
        "parameters": [],
        "nodes": [
            node(0, "sub", json!(["input:0", "input:1"]), json!([])),
            node(1, "square", json!(["node:0"]), json!([])),
            node(2, "mean", json!(["node:1"]), json!([])),
        ],
        "outputs": {"loss": "node:2"},
    });
    let config = config(
        json!({"student": body("scale"), "teacher": body("scale")}),
        objective,
        json!(["student"]),
    );
    let weights = json!({"params": {
        "student.scale": {"shape": [2], "data": [1.0, 1.0]},
        "teacher.scale": {"shape": [2], "data": [3.0, 4.0]},
    }})
    .to_string();
    (config, weights)
}

/// `loss = mean(x / w)` with `w` starting at zero: the first step divides
/// by zero, so the loss and its gradients are not finite. What the guard in
/// `TrainState::advance` is for.
pub fn divides_by_zero() -> (String, String) {
    let graph = json!({
        "inputs": [input(0, "x", json!([null, 2]), "f32")],
        "parameters": [parameter(0, "w", json!([2]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            node(1, "div", json!(["input:0", "node:0"]), json!([])),
            node(2, "mean", json!(["node:1"]), json!([])),
        ],
        "outputs": {"loss": "node:2"},
    });
    let config = config(json!({"m": graph}), Value::Null, json!(["m"]));
    let weights = json!({"params": {"m.w": {"shape": [2], "data": [0.0, 0.0]}}}).to_string();
    (config, weights)
}

/// One model with dropout in the middle, so a training pass and an
/// evaluation differ.
pub fn with_dropout(p: f64) -> (String, String) {
    let mut dropout = node(1, "dropout", json!(["input:0"]), json!([]));
    dropout["attributes"] = json!({"p": p});
    let graph = json!({
        "inputs": [input(0, "x", json!([null, 2]), "f32")],
        "parameters": [parameter(0, "w", json!([2]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            dropout,
            named(2, "mul", json!(["node:1", "node:0"]), json!([]), "scaled"),
            node(3, "mean", json!(["node:2"]), json!([])),
        ],
        "outputs": {"loss": "node:3"},
    });
    let config = config(json!({"m": graph}), Value::Null, json!(["m"]));
    let weights = json!({"params": {"m.w": {"shape": [2], "data": [1.0, 1.0]}}}).to_string();
    (config, weights)
}

/// One model that is a single named op over the batch, so a test can hold
/// its arithmetic against a number worked out by hand.
///
/// `shape` is the input's, `attributes` the op's. The parameter is a
/// scalar the run can differentiate, because a plan needs something to
/// train; the op under test does not read it.
pub fn one_op(op: &str, shape: Value, attributes: Value, extra_inputs: usize) -> (String, String) {
    let mut inputs = vec![input(0, "x", shape.clone(), "f32")];
    for i in 0..extra_inputs {
        inputs.push(input(i + 1, &format!("v{i}"), shape.clone(), "f32"));
    }
    let mut under_test = node(
        1,
        op,
        json!((0..=extra_inputs).map(|i| format!("input:{i}")).collect::<Vec<_>>()),
        json!([]),
    );
    under_test["attributes"] = attributes;
    under_test["name"] = json!("seen");
    let graph = json!({
        "inputs": inputs,
        "parameters": [parameter(0, "w", json!([1]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            under_test,
            node(2, "mean", json!(["node:1"]), json!([])),
            node(3, "mul", json!(["node:2", "node:0"]), json!([])),
            node(4, "mean", json!(["node:3"]), json!([])),
        ],
        "outputs": {"loss": "node:4"},
    });
    let config = config(json!({"m": graph}), Value::Null, json!(["m"]));
    let weights = json!({"params": {"m.w": {"shape": [1], "data": [1.0]}}}).to_string();
    (config, weights)
}

/// A named batch field of any shape.
pub fn field(name: &str, shape: &[i32], data: &[f32]) -> (String, crate::tensor::Tensor) {
    (
        name.to_string(),
        crate::tensor::Tensor {
            shape: shape.to_vec(),
            values: crate::tensor::Values::F32(data.to_vec()),
        },
    )
}

/// A batch of two rows for the graphs above.
pub fn batch_x(rows: &[f32]) -> crate::tensor::Batch {
    let n = rows.len() as i32 / 2;
    [(
        "x".to_string(),
        crate::tensor::Tensor {
            shape: vec![n, 2],
            values: crate::tensor::Values::F32(rows.to_vec()),
        },
    )]
    .into_iter()
    .collect()
}

/// Two models where the one that reads runs first: "reader" sorts before
/// "source", and models run in name order. What the ordering rule refuses.
pub fn reader_before_producer() -> (String, String) {
    let source = json!({
        "inputs": [input(0, "x", json!([null, 2]), "f32")],
        "parameters": [parameter(0, "w", json!([2]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            node(1, "mul", json!(["input:0", "node:0"]), json!([])),
        ],
        "outputs": {"out": "node:1"},
    });
    let reader = json!({
        "inputs": [from_model(0, "s", "source", "out", json!([null, 2]))],
        "parameters": [parameter(0, "w", json!([2]), true)],
        "nodes": [
            node(0, "parameter", json!([]), json!([0])),
            node(1, "mul", json!(["input:0", "node:0"]), json!([])),
            node(2, "mean", json!(["node:1"]), json!([])),
        ],
        "outputs": {"loss": "node:2"},
    });
    let config = config(
        json!({"reader": reader, "source": source}),
        Value::Null,
        json!(["reader"]),
    );
    let weights = json!({"params": {
        "reader.w": {"shape": [2], "data": [1.0, 1.0]},
        "source.w": {"shape": [2], "data": [1.0, 1.0]},
    }})
    .to_string();
    (config, weights)
}

/// A linear held quantized (docs/qlora.md), and the file it is read from.
///
/// `proj` is three frozen parameters (a packed u32 weight, scales and
/// biases, 4 bits in groups of 32 over a width of 64) read by one
/// `quantized_matmul`; `emb` is declared as f32 numbers; `head` is the
/// one trainable parameter, so a step has something to move and the
/// base something to be left alone against.
pub mod quantized {
    use super::*;
    use crate::mlxc::quantized::{quantize, Mode, Quantization};
    use crate::mlxc::Array;

    pub const D_IN: usize = 64;
    pub const D_OUT: usize = 8;
    pub const EMB_ROWS: usize = 4;
    pub const QUANTIZATION: Quantization = Quantization { group_size: 32, bits: 4, mode: Mode::Affine };

    fn typed(id: usize, path: &str, shape: Value, dtype: &str, trainable: bool) -> Value {
        let mut spec = parameter(id, path, shape, trainable);
        spec["dtype"] = json!(dtype);
        spec
    }

    pub fn config() -> String {
        let graph = json!({
            "inputs": [input(0, "x", json!([null, D_IN]), "f32")],
            "parameters": [
                typed(0, "proj.weight", json!([D_OUT, D_IN * 4 / 32]), "u32", false),
                typed(1, "proj.scales", json!([D_OUT, D_IN / 32]), "f32", false),
                typed(2, "proj.biases", json!([D_OUT, D_IN / 32]), "f32", false),
                typed(3, "emb.weight", json!([EMB_ROWS, D_IN]), "f32", false),
                parameter(4, "head.weight", json!([D_OUT]), true),
            ],
            "nodes": [
                node(0, "parameter", json!([]), json!([0])),
                node(1, "parameter", json!([]), json!([1])),
                node(2, "parameter", json!([]), json!([2])),
                node(3, "parameter", json!([]), json!([3])),
                node(4, "parameter", json!([]), json!([4])),
                {
                    "id": 5, "op": "quantized_matmul", "name": "projected",
                    "inputs": ["input:0", "node:0", "node:1", "node:2"], "parameters": [],
                    "attributes": {"group_size": 32, "bits": 4, "mode": "affine"},
                },
                node(6, "mul", json!(["node:5", "node:4"]), json!([])),
                node(7, "mean", json!(["node:6"]), json!([])),
            ],
            "outputs": {"loss": "node:7"},
        });
        super::config(json!({"m": graph}), Value::Null, json!(["m"]))
    }

    /// The numbers `proj` stands for, before quantization: row r in
    /// `[-(r + 1) / D_OUT, (r + 1) / D_OUT]`, off the grid.
    pub fn weight(rows: usize) -> Vec<f32> {
        (0..rows * D_IN)
            .map(|i| {
                let (r, c) = (i / D_IN, i % D_IN);
                (r + 1) as f32 / rows as f32 * ((c as f32 * 0.37 + r as f32 * 1.3).sin())
            })
            .collect()
    }

    /// Writes the file a published checkpoint of this model would be:
    /// `proj` quantized, `head` as numbers, and `emb` as numbers or, as
    /// mlx-lm publishes an embedding, quantized too. Returns the path.
    pub fn file(dir: &std::path::Path, embedding_packed: bool) -> std::path::PathBuf {
        let proj = Array::from_slice(&weight(D_OUT), &[D_OUT as i32, D_IN as i32]).unwrap();
        let proj = quantize(proj, QUANTIZATION).unwrap();
        let emb = Array::from_slice(&weight(EMB_ROWS), &[EMB_ROWS as i32, D_IN as i32]).unwrap();
        let mut named = vec![
            ("proj.weight", proj.weight),
            ("proj.scales", proj.scales),
            ("proj.biases", proj.biases.unwrap()),
            ("head.weight", Array::from_slice(&[1.0f32; D_OUT], &[D_OUT as i32]).unwrap()),
        ];
        if embedding_packed {
            let packed = quantize(emb, QUANTIZATION).unwrap();
            named.push(("emb.weight", packed.weight));
            named.push(("emb.scales", packed.scales));
            named.push(("emb.biases", packed.biases.unwrap()));
        } else {
            named.push(("emb.weight", emb));
        }
        let path = dir.join("model.safetensors");
        Array::save_safetensors(named.iter().map(|(n, a)| (*n, a)), None, &path).unwrap();
        path
    }

    /// The same file with `emb` replaced: what a file that is wrong in
    /// one way looks like.
    pub fn file_with(dir: &std::path::Path, emb: Vec<(&str, Array)>) -> std::path::PathBuf {
        let proj = Array::from_slice(&weight(D_OUT), &[D_OUT as i32, D_IN as i32]).unwrap();
        let proj = quantize(proj, QUANTIZATION).unwrap();
        let mut named = vec![
            ("proj.weight", proj.weight),
            ("proj.scales", proj.scales),
            ("proj.biases", proj.biases.unwrap()),
            ("head.weight", Array::from_slice(&[1.0f32; D_OUT], &[D_OUT as i32]).unwrap()),
        ];
        named.extend(emb);
        let path = dir.join("model.safetensors");
        Array::save_safetensors(named.iter().map(|(n, a)| (*n, a)), None, &path).unwrap();
        path
    }
}
