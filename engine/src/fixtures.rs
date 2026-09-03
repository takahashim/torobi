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
                node(1, "mul", json!(["input:0", "node:0"]), json!([])),
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

/// A batch of two rows for the graphs above.
pub fn batch_x(rows: &[f32]) -> crate::tensor::Batch {
    let n = rows.len() as i32 / 2;
    [(
        "x".to_string(),
        crate::tensor::Tensor {
            dtype: mlx_rs::Dtype::Float32,
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
