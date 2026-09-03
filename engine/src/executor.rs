//! Running a plan: forward, and forward-with-gradients.
//!
//! Nothing here is stateful. Every function takes the plan, the current
//! parameters and the step's bound inputs, and returns values; what to do
//! with them is [`crate::state::TrainState`]'s business. Keeping the two
//! apart is what makes the transaction in `advance` reviewable: this file
//! decides what the numbers are, that one decides when they become real.

use std::collections::BTreeMap;

use anyhow::Result;
use mlx_rs::error::Exception;
use mlx_rs::transforms::{eval, value_and_grad_with_argnums};
use mlx_rs::Array;

use crate::graph::Graph;
use crate::interp::{self, Stat};
use crate::plan::{Model, Plan};
use crate::state::Pass;

/// What a tap asked for, by node name.
pub type Taps = BTreeMap<String, Stat>;

/// What the taps saw, by node name.
pub type Tapped = BTreeMap<String, Array>;

/// Runs every model, then the objective over their outputs.
///
/// `rng` is both the randomness and the mode: a key means a training pass,
/// and no key means an evaluation, where the random ops stand aside (see
/// [`crate::interp::evaluate_tapped`]).
pub fn forward(
    plan: &Plan,
    params: &[Array],
    fields: &BTreeMap<String, Array>,
    rng: Option<&Array>,
    taps: &Taps,
    collected: &mut Tapped,
) -> std::result::Result<Array, Exception> {
    let fail = |what: String| Exception::custom(what);
    let mut outputs: BTreeMap<(String, String), Array> = BTreeMap::new();

    // One key per graph, split from the step's, so a model and the
    // objective never draw the same numbers.
    let mut key = rng.cloned();
    for Model { name, graph, slice } in &plan.models {
        let inputs = resolve(graph, fields, &outputs, name)?;
        let mine = match &key {
            Some(current) => {
                let (next, mine) = mlx_rs::random::split(current, 2)?;
                key = Some(next);
                Some(mine)
            }
            None => None,
        };
        let produced = interp::evaluate_tapped(
            graph,
            &params[slice.clone()],
            &inputs,
            mine.as_ref(),
            taps,
            collected,
        )?;
        for (output, value) in produced {
            outputs.insert((name.clone(), output), value);
        }
    }

    let Some(objective) = plan.objective.as_ref() else {
        // No objective: a single model's only output is the loss.
        let mut values = outputs.into_values();
        return values
            .next()
            .ok_or_else(|| fail("the model produced no output".into()));
    };

    let inputs = resolve(objective, fields, &outputs, "objective")?;
    let produced =
        interp::evaluate_tapped(objective, &[], &inputs, key.as_ref(), taps, collected)?;
    produced
        .into_values()
        .next()
        .ok_or_else(|| fail("the objective produced no output".into()))
}

/// One graph's inputs, each read from its declared source.
fn resolve(
    graph: &Graph,
    fields: &BTreeMap<String, Array>,
    outputs: &BTreeMap<(String, String), Array>,
    where_: &str,
) -> std::result::Result<BTreeMap<String, Array>, Exception> {
    graph
        .inputs
        .iter()
        .map(|spec| {
            let value = if let Some(field) = spec.batch_field() {
                fields
                    .get(field)
                    .cloned()
                    .ok_or_else(|| Exception::custom(format!("{where_}: no batch field {field:?}")))?
            } else {
                let (model, output) = spec.model_output().ok_or_else(|| {
                    Exception::custom(format!("{where_}: input {:?} has no source", spec.name))
                })?;
                outputs
                    .get(&(model.to_string(), output.to_string()))
                    .cloned()
                    .ok_or_else(|| {
                        Exception::custom(format!(
                            "{where_}: {model}.{output} has not been produced; \
                             a model must be declared before what reads it"
                        ))
                    })?
            };
            Ok((spec.name.clone(), value))
        })
        .collect()
}

/// The loss and its gradients, and what the taps saw on the way.
///
/// The taps are collected from a forward pass of their own rather than
/// from inside `value_and_grad`: a traced pass produces values that
/// belong to the trace, and reading them out is not what a tap means.
/// The cost is one forward per step while a tap is on, which is why
/// taps are opt-in.
pub fn differentiate(
    plan: &Plan,
    pass: Pass<'_>,
    fields: &BTreeMap<String, Array>,
    taps: &Taps,
) -> Result<(Array, Vec<Array>, Tapped)> {
    let rng = pass.rng;
    let fun = |ps: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
        let mut inner = Tapped::new();
        forward(plan, ps, fields, Some(rng), &Taps::new(), &mut inner).map(|loss| vec![loss])
    };
    let mut vg = value_and_grad_with_argnums(fun, pass.argnums);
    let (mut values, grads) = vg(pass.params)?;
    let loss = values.remove(0);
    eval(std::iter::once(&loss).chain(grads.iter()))?;

    let mut collected = Tapped::new();
    if !taps.is_empty() {
        forward(plan, pass.params, fields, Some(rng), taps, &mut collected)?;
        eval(collected.values())?;
    }
    Ok((loss, grads, collected))
}

/// The loss for one batch, without gradients and without randomness.
///
/// What a validation set is read with. Running `differentiate` and throwing
/// the gradients away would pay for a backward pass nobody asked for, and
/// would sample dropout, which would make the number noise.
pub fn evaluate(
    plan: &Plan,
    params: &[Array],
    fields: &BTreeMap<String, Array>,
    taps: &Taps,
) -> Result<(Array, Tapped)> {
    let mut collected = Tapped::new();
    let loss = forward(plan, params, fields, None, taps, &mut collected)?;
    eval(std::iter::once(&loss).chain(collected.values()))?;
    Ok((loss, collected))
}
