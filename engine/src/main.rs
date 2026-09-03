//! torobi-engine: the command-line face of the engine, for spikes and
//! debugging. The Ruby extension uses the same [`Session`] in-process.
//!
//!   torobi-engine grad  <graph.json> <bindings.json>
//!   torobi-engine train <graph.json> <bindings.json> <steps> <lr>
//!
//! `bindings.json` holds `{"inputs": {...}, "params": {...}}`: the CLI is a
//! one-batch tool, so it feeds the same batch to every step. A real caller
//! gives each step its own (docs/plan.md section 5A.2).

use anyhow::{bail, Result};
use torobi_engine::session::Batch;
use torobi_engine::Session;

use serde::Deserialize;

#[derive(Deserialize)]
struct Bindings {
    inputs: Batch,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("grad") if args.len() == 3 => grad(&args[1], &args[2]),
        Some("train") if args.len() == 5 => {
            train(&args[1], &args[2], args[3].parse()?, args[4].parse()?)
        }
        _ => bail!(
            "usage: torobi-engine grad <graph> <bindings> | \
             train <graph> <bindings> <steps> <lr>"
        ),
    }
}

fn open(graph_path: &str, bindings_path: &str) -> Result<(Session, Batch)> {
    let graph_json = std::fs::read_to_string(graph_path)?;
    let bindings_json = std::fs::read_to_string(bindings_path)?;
    let session = Session::open(&graph_json, &bindings_json)?;
    let bindings: Bindings = serde_json::from_str(&bindings_json)?;
    Ok((session, bindings.inputs))
}

fn grad(graph_path: &str, bindings_path: &str) -> Result<()> {
    let (session, batch) = open(graph_path, bindings_path)?;
    let (loss, _) = session.loss_and_grads(&batch)?;
    let grads = session
        .gradients(&batch)?
        .into_iter()
        .map(|(path, t)| (path, serde_json::json!({ "shape": t.shape, "data": t.data })))
        .collect::<serde_json::Map<_, _>>();
    println!(
        "{}",
        serde_json::json!({ "loss": loss.item::<f32>(), "grads": grads })
    );
    Ok(())
}

fn train(graph_path: &str, bindings_path: &str, steps: usize, lr: f32) -> Result<()> {
    let (mut session, batch) = open(graph_path, bindings_path)?;
    session.set_lr(lr);
    while session.step() < steps {
        let loss = session.run_step(&batch)?;
        if session.step() % 10 == 0 {
            println!("{}", serde_json::json!({ "step": session.step(), "loss": loss }));
        }
    }
    println!("{}", serde_json::json!({ "final_loss": session.loss() }));
    Ok(())
}
