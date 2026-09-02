//! torobi-engine, M1 spike: interpret a GraphConfig on MLX.
//!
//!   torobi-engine grad  <graph.json> <bindings.json>
//!   torobi-engine train <graph.json> <bindings.json> <steps> <lr>
//!
//! `grad` prints the loss and the gradient for every trainable parameter;
//! `train` runs plain SGD and prints one JSON line every 10 steps. Both
//! read the model named "spike", or the only model in the file.

mod graph;
mod interp;

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::Array;
use serde::Deserialize;

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<i32>,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct Bindings {
    inputs: BTreeMap<String, Tensor>,
    params: BTreeMap<String, Tensor>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("grad") if args.len() == 3 => grad(&args[1], &args[2]),
        Some("train") if args.len() == 5 => {
            train(&args[1], &args[2], args[3].parse()?, args[4].parse()?)
        }
        _ => bail!("usage: torobi-engine grad <graph> <bindings> | train <graph> <bindings> <steps> <lr>"),
    }
}

struct Loaded {
    graph: graph::Graph,
    params: Vec<Array>,
    inputs: BTreeMap<String, Array>,
}

fn load(graph_path: &str, bindings_path: &str) -> Result<Loaded> {
    let config: graph::GraphConfig =
        serde_json::from_str(&std::fs::read_to_string(graph_path).context("reading graph")?)?;
    let mut models = config.models;
    let graph = models
        .remove("spike")
        .or_else(|| models.pop_first().map(|(_, g)| g))
        .context("the graph file has no models")?;

    let bindings: Bindings =
        serde_json::from_str(&std::fs::read_to_string(bindings_path).context("reading bindings")?)?;
    let params = graph
        .parameters
        .iter()
        .map(|p| {
            let t = bindings
                .params
                .get(&p.path)
                .with_context(|| format!("bindings are missing parameter {:?}", p.path))?;
            anyhow::ensure!(t.shape == p.shape, "parameter {:?}: shape mismatch", p.path);
            Ok(Array::from_slice(&t.data, &t.shape))
        })
        .collect::<Result<Vec<_>>>()?;
    let inputs = bindings
        .inputs
        .iter()
        .map(|(name, t)| (name.clone(), Array::from_slice(&t.data, &t.shape)))
        .collect();
    Ok(Loaded { graph, params, inputs })
}

fn grad(graph_path: &str, bindings_path: &str) -> Result<()> {
    let loaded = load(graph_path, bindings_path)?;
    let (loss, grads) = loss_and_grads(&loaded)?;

    let mut out = serde_json::Map::new();
    out.insert("loss".into(), loss.item::<f32>().into());
    let mut grads_json = serde_json::Map::new();
    for (spec, grad) in loaded.graph.parameters.iter().zip(&grads) {
        // A gradient can come back strided (e.g. through a transpose);
        // reading it out requires contiguous memory.
        let grad = grad.contiguous()?;
        eval(std::iter::once(&grad))?;
        grads_json.insert(
            spec.path.clone(),
            serde_json::json!({
                "shape": grad.shape(),
                "data": grad.as_slice::<f32>(),
            }),
        );
    }
    out.insert("grads".into(), grads_json.into());
    println!("{}", serde_json::Value::Object(out));
    Ok(())
}

fn train(graph_path: &str, bindings_path: &str, steps: usize, lr: f32) -> Result<()> {
    let mut loaded = load(graph_path, bindings_path)?;
    let lr = Array::from_f32(lr);
    let mut last = f32::NAN;
    for step in 0..steps {
        let (loss, grads) = loss_and_grads(&loaded)?;
        loaded.params = loaded
            .params
            .iter()
            .zip(&grads)
            .map(|(p, g)| Ok(p.subtract(g.multiply(&lr)?)?))
            .collect::<Result<Vec<_>>>()?;
        eval(loaded.params.iter())?;
        last = loss.item::<f32>();
        if step % 10 == 0 {
            println!("{}", serde_json::json!({ "step": step, "loss": last }));
        }
    }
    println!("{}", serde_json::json!({ "final_loss": last }));
    Ok(())
}

fn loss_and_grads(loaded: &Loaded) -> Result<(Array, Vec<Array>)> {
    let argnums: Vec<i32> = (0..loaded.params.len() as i32).collect();
    let fun = |ps: &[Array]| -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
        interp::evaluate(&loaded.graph, ps, &loaded.inputs)
    };
    let mut vg = value_and_grad_with_argnums(fun, &argnums[..]);
    let (mut values, grads) = vg(&loaded.params)?;
    anyhow::ensure!(values.len() == 1, "the spike graph must output exactly the loss");
    let loss = values.remove(0);
    eval(std::iter::once(&loss).chain(grads.iter()))?;
    Ok((loss, grads))
}
