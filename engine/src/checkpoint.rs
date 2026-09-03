//! Writing and reading a run's state.
//!
//! The format is the one docs/plan.md section 11.2 specifies:
//!
//! ```text
//! checkpoint/000042/
//! ├── manifest.json
//! ├── parameters.safetensors
//! ├── random.safetensors
//! └── optimizer.safetensors     (absent for an optimizer with no slots)
//! ```
//!
//! Two rules give it its value. It is written to a temporary directory and
//! renamed into place, so an interrupted write leaves no half-checkpoint.
//! And it is read back against the manifest - the config's digest, every
//! parameter's path and shape, the optimizer's kind - so a mismatch is
//! refused rather than silently absorbed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::optimizer::Config as OptimizerConfig;

pub const SCHEMA_VERSION: u32 = 1;

/// What a checkpoint claims about itself. Everything here is checked on the
/// way back in.
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    /// The GraphConfig this state belongs to. A checkpoint restored into a
    /// different description is not a resumed run.
    pub config_digest: String,
    pub step: usize,
    pub optimizer: OptimizerConfig,
    /// Steps the optimizer has taken, which is not the same as `step` once
    /// a run resumes: bias correction depends on it.
    pub optimizer_steps: u64,
    /// The seed the RNG was started from. The key itself is a tensor, and
    /// lives in random.safetensors.
    pub seed: u64,
    pub parameters: Vec<ParameterEntry>,
    pub build: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct ParameterEntry {
    pub path: String,
    pub shape: Vec<i32>,
    pub trained: bool,
}

/// The state a checkpoint holds, on its way in or out.
pub struct State<'a> {
    pub config_digest: &'a str,
    pub step: usize,
    pub optimizer: &'a OptimizerConfig,
    pub optimizer_steps: u64,
    /// Every parameter, by qualified path, in declaration order.
    pub parameters: Vec<(String, &'a Array)>,
    /// Which of them are differentiated, as positions into `parameters`.
    pub argnums: &'a [i32],
    /// The optimizer's slots, parallel to `argnums`.
    pub slots: (&'a [Array], &'a [Array]),
    /// The RNG key, so a resumed run draws what a continuous one would.
    pub rng: &'a Array,
    pub seed: u64,
}

/// Writes a checkpoint, atomically. Returns where it landed.
pub fn write(dir: impl AsRef<Path>, state: State<'_>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    let staging = dir.with_extension("writing");
    if staging.exists() {
        std::fs::remove_dir_all(&staging).ok();
    }
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("creating {}", staging.display()))?;

    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        config_digest: state.config_digest.to_string(),
        step: state.step,
        optimizer: state.optimizer.clone(),
        optimizer_steps: state.optimizer_steps,
        seed: state.seed,
        parameters: state
            .parameters
            .iter()
            .enumerate()
            .map(|(i, (path, array))| ParameterEntry {
                path: path.clone(),
                shape: array.shape().to_vec(),
                trained: state.argnums.contains(&(i as i32)),
            })
            .collect(),
        build: crate::build_info(),
    };

    Array::save_safetensors(
        state.parameters.iter().map(|(path, array)| (path, *array)),
        None,
        staging.join("parameters.safetensors"),
    )
    .context("writing parameters")?;

    let (m, v) = state.slots;
    if !m.is_empty() {
        let named: Vec<(String, &Array)> = state
            .argnums
            .iter()
            .enumerate()
            .flat_map(|(slot, &i)| {
                let path = &state.parameters[i as usize].0;
                [(format!("m/{path}"), &m[slot]), (format!("v/{path}"), &v[slot])]
            })
            .collect();
        Array::save_safetensors(named.iter().map(|(k, a)| (k, *a)), None,
                                staging.join("optimizer.safetensors"))
            .context("writing optimizer state")?;
    }

    Array::save_safetensors(
        [("key", state.rng)],
        None,
        staging.join("random.safetensors"),
    )
    .context("writing the RNG state")?;

    std::fs::write(
        staging.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .context("writing the manifest")?;

    // Everything is on disk and reads back before anything claims the
    // name: the manifest parses, and every tensor loads.
    read(&staging).context("the checkpoint just written does not read back")?;
    sync_dir(&staging)?;

    // A rename over an existing directory is not atomic on this platform:
    // the old one has to go first, and a stop in between would leave
    // neither. So the previous checkpoint is moved aside, the new one
    // takes the name in one rename, and only then is the old one dropped.
    let displaced = if dir.exists() {
        let aside = dir.with_extension("replaced");
        if aside.exists() {
            std::fs::remove_dir_all(&aside).ok();
        }
        std::fs::rename(dir, &aside)
            .with_context(|| format!("moving {} aside", dir.display()))?;
        Some(aside)
    } else {
        None
    };
    std::fs::rename(&staging, dir)
        .with_context(|| format!("renaming {} into place", staging.display()))?;
    if let Some(aside) = displaced {
        std::fs::remove_dir_all(aside).ok();
    }
    if let Some(parent) = dir.parent() {
        sync_dir(parent).ok();
    }
    Ok(dir.to_path_buf())
}

/// Asks the filesystem to make a directory's entries durable, so that a
/// checkpoint that exists after a crash is one that reads back.
fn sync_dir(dir: &Path) -> Result<()> {
    let handle = std::fs::File::open(dir).with_context(|| format!("opening {}", dir.display()))?;
    handle
        .sync_all()
        .with_context(|| format!("syncing {}", dir.display()))
}

pub fn read_manifest(dir: impl AsRef<Path>) -> Result<Manifest> {
    let path = dir.as_ref().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let manifest: Manifest = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        manifest.schema_version == SCHEMA_VERSION,
        "checkpoint schema {} is not {}",
        manifest.schema_version,
        SCHEMA_VERSION
    );
    Ok(manifest)
}

/// What a checkpoint holds, read back.
pub struct Loaded {
    pub manifest: Manifest,
    pub parameters: HashMap<String, Array>,
    /// (m, v) by parameter path. Empty when the optimizer has no slots.
    pub slots: HashMap<String, (Array, Array)>,
    pub rng: Option<Array>,
}

pub fn read(dir: impl AsRef<Path>) -> Result<Loaded> {
    let dir = dir.as_ref();
    let manifest = read_manifest(dir)?;
    let parameters = Array::load_safetensors(dir.join("parameters.safetensors"))
        .context("reading parameters")?;

    let optimizer_path = dir.join("optimizer.safetensors");
    let mut slots = HashMap::new();
    if optimizer_path.exists() {
        let arrays =
            Array::load_safetensors(&optimizer_path).context("reading optimizer state")?;
        for (key, array) in &arrays {
            let Some(path) = key.strip_prefix("m/") else {
                continue;
            };
            let v = arrays
                .get(&format!("v/{path}"))
                .with_context(|| format!("optimizer state has m/{path} but no v/{path}"))?;
            slots.insert(path.to_string(), (array.clone(), v.clone()));
        }
    }
    let rng = Array::load_safetensors(dir.join("random.safetensors"))
        .context("reading the RNG state")?
        .remove("key");
    Ok(Loaded {
        manifest,
        parameters,
        slots,
        rng,
    })
}
