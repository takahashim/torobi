//! A model's parameters in the layout a published checkpoint has, for a
//! serving consumer (docs/plan.md section 15.44).
//!
//! Not a checkpoint. A checkpoint is the run's record: optimizer state, the
//! RNG and the counters, under the run's own names. This is one model's
//! parameters under the names the published model uses, in fp32, with the
//! header a loader looks for, and nothing about the run. It reads the
//! parameters and touches nothing else, which is why it is not the
//! training state's business.

use std::path::Path;

use anyhow::{Context, Result};

use crate::mlxc::transforms::eval;
use crate::mlxc::{Array, Dtype};
use crate::plan::{Model, Plan};

/// Writes one model's parameters to `dir` as an HF-compatible fp32
/// safetensors checkpoint. The GraphConfig model name is stripped from
/// each qualified path, so a model declared with `encoder_prefix:`
/// keeps that prefix and the file's keys match the published layout.
///
/// Returns the (old, new) path pairs so the caller can write metadata
/// that names what was saved.
pub fn write(plan: &Plan, params: &[Array], model: &str, dir: &str) -> Result<Vec<(String, String)>> {
    let model = plan
        .models
        .iter()
        .find(|m| m.name == model)
        .with_context(|| {
            format!(
                "no model named {model:?} (this run has {:?})",
                plan.models.iter().map(|m| &m.name).collect::<Vec<_>>()
            )
        })?;
    let owned = published(plan, params, model)?;
    let renamed = plan.paths[model.slice.clone()]
        .iter()
        .cloned()
        .zip(owned.iter().map(|(path, _)| path.clone()))
        .collect();

    let dir = Path::new(dir);
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    // `{"format": "pt"}`, which is what a published checkpoint carries
    // and what a loader looks for: transformers reads the metadata and
    // refuses a file whose format it does not recognize. The bytes are
    // plain little-endian f32 in reading order, which is what "pt"
    // describes; the word names a layout, not a framework that wrote
    // it.
    let metadata = std::collections::HashMap::from([
        ("format".to_string(), "pt".to_string()),
    ]);
    let file = dir.join("model.safetensors");
    Array::save_safetensors(
        owned.iter().map(|(path, array)| (path.as_str(), array)),
        &metadata,
        &file,
    )
    .context("writing model.safetensors")?;

    // Read back before saying it is written, for the reason a
    // checkpoint does (docs/plan.md section 15.27): a file that does
    // not load is worse than no file, and finding out here costs one
    // read of what was just written.
    let read = Array::load_safetensors(&file)
        .context("the export just written does not read back")?;
    for (path, _) in &owned {
        anyhow::ensure!(
            read.contains_key(path.as_str()),
            "the export just written has no {path:?}"
        );
    }
    anyhow::ensure!(
        read.len() == owned.len(),
        "the export just written holds {} tensors and {} were saved",
        read.len(),
        owned.len()
    );
    Ok(renamed)
}

/// One model's parameters under the names a published checkpoint uses,
/// in fp32.
///
/// Pure: it decides what would be written and touches no disk, so what
/// goes in a file can be looked at without making one. The model's own
/// name is stripped, so a model declared with `encoder_prefix:` keeps
/// that prefix and the keys line up with the published layout.
fn published(plan: &Plan, params: &[Array], model: &Model) -> Result<Vec<(String, Array)>> {
    let prefix = format!("{}.", model.name);
    let owned: Vec<(String, Array)> = plan.paths[model.slice.clone()]
        .iter()
        .zip(&params[model.slice.clone()])
        .map(|(path, array)| {
            let array = if array.dtype() == Dtype::Float32 {
                array.clone()
            } else {
                array.as_dtype(Dtype::Float32)?
            };
            Ok((path.strip_prefix(&prefix).unwrap_or(path).to_string(), array))
        })
        .collect::<Result<_>>()?;
    eval(owned.iter().map(|(_, a)| a))?;
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::plan::Weights;
    use crate::tensor::{to_tensor, Values};

    fn open(which: (String, String)) -> (Plan, Vec<Array>) {
        let (config, weights) = which;
        Plan::open(&config, Weights::Inline(&weights)).unwrap()
    }

    fn values(array: &Array) -> Vec<f32> {
        match to_tensor(array).unwrap().values {
            Values::F32(v) => v,
            Values::I32(v) => v.iter().map(|&i| i as f32).collect(),
        }
    }

    #[test]
    fn one_models_parameters_are_written_with_its_name_stripped() {
        // student.scale exports as "scale": the GraphConfig model name is
        // the one thing a serving consumer does not carry.
        let (plan, params) = open(fixtures::teacher_and_student());
        let dir = tempfile::tempdir().unwrap();
        let export_to = dir.path().join("out").display().to_string();

        let renamed = write(&plan, &params, "student", &export_to).unwrap();
        assert_eq!(renamed, vec![("student.scale".to_string(), "scale".to_string())]);

        let file = std::path::Path::new(&export_to).join("model.safetensors");
        let arrays = Array::load_safetensors(&file).unwrap();
        assert_eq!(arrays.len(), 1, "only the student's parameter is exported");
        let array = &arrays["scale"];
        assert_eq!(array.dtype(), Dtype::Float32);
        assert_eq!(to_tensor(array).unwrap().shape, vec![2]);
        let held = &params[plan.index_of("student.scale").unwrap()];
        let (exported, held) = (values(array), values(held));
        for (got, want) in exported.iter().zip(&held) {
            assert!((got - want).abs() < 1e-6, "{exported:?} against {held:?}");
        }
    }

    #[test]
    fn an_export_says_what_format_it_is_in() {
        // A loader reads this before the tensors, and refuses a file whose
        // format it cannot name. A published checkpoint carries the same.
        let (plan, params) = open(fixtures::teacher_and_student());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        write(&plan, &params, "student", &out.display().to_string()).unwrap();

        let bytes = std::fs::read(out.join("model.safetensors")).unwrap();
        let length = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();

        assert_eq!(header["__metadata__"]["format"], "pt");
    }

    #[test]
    fn a_model_name_this_run_does_not_have_is_refused() {
        let (plan, params) = open(fixtures::scaled_mean());
        let dir = tempfile::tempdir().unwrap();
        let e = write(&plan, &params, "elsewhere", &dir.path().display().to_string()).unwrap_err();
        assert!(e.to_string().contains("no model named"), "{e}");
    }
}
