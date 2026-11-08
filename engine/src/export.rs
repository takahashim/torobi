//! A model's parameters in the layout a published checkpoint has, for a
//! serving consumer (docs/plan.md section 15.44).
//!
//! Not a checkpoint. A checkpoint is the run's record: optimizer state, the
//! RNG and the counters, under the run's own names. This is one model's
//! parameters under the names the published model uses, in fp32, with the
//! header a loader looks for, and nothing about the run. It reads the
//! parameters and touches nothing else, which is why it is not the
//! training state's business.
//!
//! Two things a published model does not have are folded away on the way
//! out (docs/qlora.md section 3.6). A weight held packed (`u32`, with its
//! scales and biases) is written as the numbers it stands for, and the
//! scales and biases are not written. An adapter (`lora_A`, `lora_B`
//! beside a weight) is merged into that weight, `W + scale · B A`, when
//! the caller says what the scale is, and refused otherwise: the base of
//! an adapted run never moved, and writing it alone would be a model that
//! looks trained and is not. [`write_adapter`] is the other shape a
//! fine-tune through an adapter produces: the adapter alone, named as
//! PEFT names it.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

use crate::mlxc::quantized::dequantize;
use crate::mlxc::transforms::eval;
use crate::mlxc::{Array, Dtype};
use crate::plan::{Model, Plan};

/// What an adapter's two matrices are called beside the weight they
/// adapt, as PEFT spells them and as `Torobi::LoRA` declares them.
const LORA_A: &str = ".lora_A.weight";
const LORA_B: &str = ".lora_B.weight";

/// Where PEFT puts a module's path when it saves an adapter: under the
/// model it wraps.
const PEFT_PREFIX: &str = "base_model.model.";

/// Writes one model's parameters to `dir` as an HF-compatible fp32
/// safetensors checkpoint. The GraphConfig model name is stripped from
/// each qualified path, so a model declared with `encoder_prefix:`
/// keeps that prefix and the file's keys match the published layout.
///
/// `merge` is the adapter's scale (`alpha / rank`), when the run holds
/// one and the caller wants it folded into the weights; without it, an
/// adapted run is refused. A packed weight is written as its numbers
/// either way.
///
/// Returns the (old, new) path pairs so the caller can write metadata
/// that names what was saved. Parameters folded away (an adapter's
/// matrices, a packed weight's scales and biases) are not in it.
pub fn write(plan: &Plan, params: &[Array], model: &str, dir: &str, merge: Option<f32>) -> Result<Vec<(String, String)>> {
    let model = find(plan, model)?;
    let owned = published(plan, params, model, merge)?;
    save(dir, "model.safetensors", owned)
}

/// Writes one model's adapter alone, as PEFT saves one: the `lora_A` and
/// `lora_B` of every adapted linear under `base_model.model.`, in fp32,
/// in `adapter_model.safetensors`. The `adapter_config.json` beside it
/// is the caller's (docs/qlora.md section 3.6): it says what the graph
/// said (`r`, `lora_alpha`, `target_modules`), and the graph is Ruby's.
///
/// Refused when the model holds no adapter: a file with no tensors
/// would load as an adapter that changes nothing.
pub fn write_adapter(plan: &Plan, params: &[Array], model: &str, dir: &str) -> Result<Vec<(String, String)>> {
    let model = find(plan, model)?;
    let prefix = format!("{}.", model.name);
    let owned: Vec<(String, String, Array)> = plan.paths[model.slice.clone()]
        .iter()
        .zip(&params[model.slice.clone()])
        .filter(|(path, _)| path.ends_with(LORA_A) || path.ends_with(LORA_B))
        .map(|(path, array)| {
            let own = path.strip_prefix(&prefix).unwrap_or(path);
            Ok((path.clone(), format!("{PEFT_PREFIX}{own}"), f32s(array)?))
        })
        .collect::<Result<_>>()?;
    anyhow::ensure!(
        !owned.is_empty(),
        "model {:?} holds no adapter (no parameter ends in {LORA_A} or {LORA_B}), so there \
         is no adapter to write; export_model writes the model itself",
        model.name
    );
    eval(owned.iter().map(|(_, _, a)| a))?;
    save(dir, "adapter_model.safetensors", owned)
}

fn find<'a>(plan: &'a Plan, model: &str) -> Result<&'a Model> {
    plan.models.iter().find(|m| m.name == model).with_context(|| {
        format!(
            "no model named {model:?} (this run has {:?})",
            plan.models.iter().map(|m| &m.name).collect::<Vec<_>>()
        )
    })
}

fn f32s(array: &Array) -> Result<Array> {
    Ok(if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32)?
    })
}

/// Writes `owned` ((old path, new path, array)) to `dir/name` with the
/// header a loader looks for, reads it back, and returns the renaming.
fn save(dir: &str, name: &str, owned: Vec<(String, String, Array)>) -> Result<Vec<(String, String)>> {
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
    let file = dir.join(name);
    Array::save_safetensors(
        owned.iter().map(|(_, path, array)| (path.as_str(), array)),
        &metadata,
        &file,
    )
    .with_context(|| format!("writing {name}"))?;

    // Read back before saying it is written, for the reason a
    // checkpoint does (docs/plan.md section 15.27): a file that does
    // not load is worse than no file, and finding out here costs one
    // read of what was just written.
    let read = Array::load_safetensors(&file)
        .context("the export just written does not read back")?;
    for (_, path, _) in &owned {
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
    Ok(owned.into_iter().map(|(old, new, _)| (old, new)).collect())
}

/// One model's parameters under the names a published checkpoint uses,
/// in fp32, with what a published model does not have folded away.
///
/// Pure: it decides what would be written and touches no disk, so what
/// goes in a file can be looked at without making one. The model's own
/// name is stripped, so a model declared with `encoder_prefix:` keeps
/// that prefix and the keys line up with the published layout.
///
/// A packed weight becomes the numbers it stands for, and its scales and
/// biases are left out. An adapter's `B A`, times `merge`, is added to
/// the weight it sits beside, and its two matrices are left out; an
/// adapter with no `merge` is refused.
fn published(plan: &Plan, params: &[Array], model: &Model, merge: Option<f32>) -> Result<Vec<(String, String, Array)>> {
    let prefix = format!("{}.", model.name);
    let slice = model.slice.clone();
    let index_of = |path: &str| plan.index_of(path).filter(|i| slice.contains(i));

    // The adapters, by the weight each one sits beside.
    let mut adapters: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for path in &plan.paths[slice.clone()] {
        let Some(name) = path.strip_suffix(LORA_A) else {
            continue;
        };
        let b = index_of(&format!("{name}{LORA_B}"))
            .with_context(|| format!("{path:?} has no {name}{LORA_B} beside it: half an adapter"))?;
        adapters.insert(format!("{name}.weight"), (index_of(path).expect("its own path"), b));
    }
    if let (Some((path, _)), None) = (adapters.iter().next(), merge) {
        anyhow::bail!(
            "model {:?} was trained through an adapter ({} weights like {path:?} have lora_A and \
             lora_B beside them), and its base weights have not moved. Writing them alone would be \
             a model that looks trained and is not: say the adapter's scale to merge it in, or \
             write the adapter by itself (export_adapter)",
            model.name,
            adapters.len()
        );
    }

    // What a packed weight's scales and biases are read by, so they can
    // be left out: every one of them belongs to a weight in `packed`.
    let folded: std::collections::HashSet<usize> = plan
        .packed
        .iter()
        .filter(|(weight, _)| slice.contains(weight))
        .flat_map(|(_, packed)| std::iter::once(packed.scales).chain(packed.biases))
        .collect();

    let mut owned = Vec::new();
    for (i, path) in plan.paths[slice.clone()].iter().enumerate() {
        let index = slice.start + i;
        if path.ends_with(LORA_A) || path.ends_with(LORA_B) || folded.contains(&index) {
            continue;
        }
        let array = &params[index];
        let mut value = if array.dtype() == Dtype::Uint32 {
            let packed = plan.packed.get(&index).with_context(|| {
                format!("parameter {path:?} is a packed u32 weight that no quantized_matmul reads")
            })?;
            dequantize(
                array,
                &params[packed.scales],
                packed.biases.map(|b| &params[b]),
                packed.quantization,
                Some(Dtype::Float32),
            )?
        } else {
            f32s(array)?
        };
        if let (Some(&(a, b)), Some(scale)) = (adapters.get(path), merge) {
            // W + scale · B A, in f32: A is [rank, d_in], B is [d_out, rank],
            // so B A is the weight's own [d_out, d_in].
            let delta = f32s(&params[b])?.matmul(f32s(&params[a])?)?.multiply(Array::from_f32(scale)?)?;
            value = value.add(&delta)?;
        }
        owned.push((path.clone(), path.strip_prefix(&prefix).unwrap_or(path).to_string(), value));
    }
    eval(owned.iter().map(|(_, _, a)| a))?;
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::plan::Weights;
    use crate::tensor::{to_tensor, Values};

    /// A quantized base is exported as the numbers its words stand for,
    /// in the weight's own shape, and its scales and biases stay behind:
    /// what comes out is a model a plain loader reads.
    #[test]
    fn a_quantized_model_exports_the_numbers_its_weights_stand_for() {
        use fixtures::quantized::{weight, D_IN, D_OUT};
        let dir = tempfile::tempdir().unwrap();
        let files = std::collections::BTreeMap::from([("m".to_string(), fixtures::quantized::file(dir.path(), false))]);
        let (plan, params) =
            Plan::open(&fixtures::quantized::config(), Weights::Pretrained { files: &files, fresh: &[] }).unwrap();
        let out = dir.path().join("out");
        let renamed = write(&plan, &params, "m", out.to_str().unwrap(), None).unwrap();
        assert_eq!(
            renamed,
            vec![
                ("m.proj.weight".to_string(), "proj.weight".to_string()),
                ("m.emb.weight".to_string(), "emb.weight".to_string()),
                ("m.head.weight".to_string(), "head.weight".to_string()),
            ]
        );
        let arrays = Array::load_safetensors(out.join("model.safetensors")).unwrap();
        assert_eq!(arrays.len(), 3, "no scales, no biases");
        let proj = &arrays["proj.weight"];
        assert_eq!(proj.dtype(), Dtype::Float32);
        assert_eq!(proj.shape(), &[D_OUT as i32, D_IN as i32]);
        for (got, want) in values(proj).iter().zip(weight(D_OUT)) {
            assert!((got - want).abs() <= 2.0 / 15.0, "{got} for {want}");
        }
    }

    /// An adapted run is refused without a scale, and with one its
    /// adapter is folded into the weight it sits beside: `W + scale · B A`,
    /// the matrices themselves left out. Hand arithmetic: W = [0.5, -1],
    /// A = [1, 2], B = [3], scale 2 gives [6.5, 11].
    #[test]
    fn an_adapter_is_merged_into_its_weight_when_the_scale_is_said() {
        let (plan, params) = open(fixtures::adapted());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");

        let said = write(&plan, &params, "m", out.to_str().unwrap(), None).unwrap_err().to_string();
        assert!(said.contains("trained through an adapter"), "{said}");
        assert!(said.contains("\"m.l.weight\""), "{said}");
        assert!(!out.join("model.safetensors").exists());

        let renamed = write(&plan, &params, "m", out.to_str().unwrap(), Some(2.0)).unwrap();
        assert_eq!(renamed, vec![("m.l.weight".to_string(), "l.weight".to_string())]);
        let arrays = Array::load_safetensors(out.join("model.safetensors")).unwrap();
        assert_eq!(arrays.len(), 1);
        assert_eq!(values(&arrays["l.weight"]), vec![6.5, 11.0]);
    }

    /// The adapter alone, as PEFT saves one: both matrices of every
    /// adapted linear, under `base_model.model.`, and nothing of the
    /// base. A model with no adapter has none to write.
    #[test]
    fn an_adapter_is_written_as_peft_names_it() {
        let (plan, params) = open(fixtures::adapted());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("adapter");
        let renamed = write_adapter(&plan, &params, "m", out.to_str().unwrap()).unwrap();
        assert_eq!(
            renamed,
            vec![
                ("m.l.lora_A.weight".to_string(), "base_model.model.l.lora_A.weight".to_string()),
                ("m.l.lora_B.weight".to_string(), "base_model.model.l.lora_B.weight".to_string()),
            ]
        );
        let arrays = Array::load_safetensors(out.join("adapter_model.safetensors")).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(values(&arrays["base_model.model.l.lora_A.weight"]), vec![1.0, 2.0]);
        assert_eq!(values(&arrays["base_model.model.l.lora_B.weight"]), vec![3.0]);

        let (plan, params) = open(fixtures::scaled_mean());
        let said = write_adapter(&plan, &params, "m", out.to_str().unwrap()).unwrap_err().to_string();
        assert!(said.contains("holds no adapter"), "{said}");
    }

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

        let renamed = write(&plan, &params, "student", &export_to, None).unwrap();
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
        write(&plan, &params, "student", &out.display().to_string(), None).unwrap();

        let bytes = std::fs::read(out.join("model.safetensors")).unwrap();
        let length = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();

        assert_eq!(header["__metadata__"]["format"], "pt");
    }

    #[test]
    fn a_model_name_this_run_does_not_have_is_refused() {
        let (plan, params) = open(fixtures::scaled_mean());
        let dir = tempfile::tempdir().unwrap();
        let e = write(&plan, &params, "elsewhere", &dir.path().display().to_string(), None).unwrap_err();
        assert!(e.to_string().contains("no model named"), "{e}");
    }
}
