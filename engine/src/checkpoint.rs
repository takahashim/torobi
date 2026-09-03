//! Writing and reading a run's state.
//!
//! The format is the one docs/plan.md section 11.2 specifies:
//!
//! ```text
//! checkpoint/000042/
//! ├── manifest.json
//! ├── graph.json
//! ├── parameters.safetensors
//! ├── random.safetensors
//! └── optimizer.safetensors     (absent for an optimizer with no slots)
//! ```
//!
//! Three rules give it its value. It is written to a temporary directory
//! and renamed into place, so an interrupted write leaves no
//! half-checkpoint. It is read back against the manifest (the config's
//! digest, every parameter's path, shape and dtype, the optimizer's kind)
//! so a mismatch is refused rather than silently absorbed. And it carries
//! the description itself, not only its digest: a digest names a
//! GraphConfig, it does not reconstruct one, and a checkpoint nobody can
//! read without the run that wrote it is a run checkpoint in name only.
//!
//! What the engine does not know is carried opaquely. Epoch, batch
//! position, sampler state and dataset identity belong to whoever owns the
//! data, which is never this crate (docs/plan.md section 5A.2); they
//! travel in `run`, which is written verbatim and handed back verbatim.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use mlx_rs::Array;
use serde::{Deserialize, Serialize};

use crate::optimizer::Config as OptimizerConfig;
use crate::tensor::dtype_spelling;

pub const SCHEMA_VERSION: u32 = 2;

/// The file the GraphConfig is written to, beside the manifest.
pub const GRAPH_FILE: &str = "graph.json";

/// What a checkpoint claims about itself. Everything here is checked on the
/// way back in.
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    /// The GraphConfig this state belongs to. A checkpoint restored into a
    /// different description is not a resumed run.
    pub config_digest: String,
    /// Which meaning of the ops the description was written against.
    pub semantics_version: u32,
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
    /// The machine this was written on.
    pub platform: serde_json::Value,
    /// Whatever the caller wanted recorded: epoch, batch position, sampler
    /// state, dataset identity, the Ruby side's own versions. Not
    /// interpreted here, and handed back unchanged on the way out.
    #[serde(default)]
    pub run: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
pub struct ParameterEntry {
    pub path: String,
    pub shape: Vec<i32>,
    /// Checked on the way back in: a parameter restored at another
    /// precision is not the parameter that was saved.
    pub dtype: String,
    pub trained: bool,
}

/// The state a checkpoint holds, on its way in or out.
pub struct State<'a> {
    pub config_digest: &'a str,
    /// The GraphConfig itself, as the caller handed it over. Its digest
    /// must be `config_digest`, which the read-back check confirms.
    pub graph_json: &'a str,
    pub semantics_version: u32,
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
    /// The caller's own record, written verbatim.
    pub run: serde_json::Value,
}

impl Manifest {
    /// What a checkpoint claims, read off the state it is being written
    /// from.
    fn of(state: &State<'_>) -> Result<Self> {
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            config_digest: state.config_digest.to_string(),
            semantics_version: state.semantics_version,
            step: state.step,
            optimizer: state.optimizer.clone(),
            optimizer_steps: state.optimizer_steps,
            seed: state.seed,
            parameters: state
                .parameters
                .iter()
                .enumerate()
                .map(|(i, (path, array))| {
                    let dtype = dtype_spelling(array.dtype()).with_context(|| {
                        format!(
                            "parameter {path:?} holds {:?}, which the graph vocabulary \
                             cannot name; a checkpoint that recorded it could not be \
                             read back into any graph",
                            array.dtype()
                        )
                    })?;
                    Ok(ParameterEntry {
                        path: path.clone(),
                        shape: array.shape().to_vec(),
                        dtype: dtype.to_string(),
                        trained: state.argnums.contains(&(i as i32)),
                    })
                })
                .collect::<Result<_>>()?,
            build: crate::build_info(),
            platform: platform(),
            run: state.run.clone(),
        })
    }
}

/// Writes a checkpoint, atomically. Returns where it landed.
///
/// Three steps, and the order is the point: lay the whole thing out
/// somewhere it does not yet count, read it back, and only then let it
/// take the name.
pub fn write(dir: impl AsRef<Path>, state: State<'_>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    let staging = dir.with_extension("writing");
    if staging.exists() {
        std::fs::remove_dir_all(&staging).ok();
    }
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("creating {}", staging.display()))?;

    lay_out(&staging, &state)?;
    // Nothing claims the name before it reads back: the manifest parses,
    // every tensor loads, and each one is what the manifest says.
    read(&staging).context("the checkpoint just written does not read back")?;
    publish(&staging, dir)?;
    Ok(dir.to_path_buf())
}

/// Writes the five files into a directory that does not count yet.
fn lay_out(staging: &Path, state: &State<'_>) -> Result<()> {
    std::fs::write(staging.join(GRAPH_FILE), state.graph_json).context("writing the graph")?;

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
        Array::save_safetensors(
            named.iter().map(|(k, a)| (k, *a)),
            None,
            staging.join("optimizer.safetensors"),
        )
        .context("writing optimizer state")?;
    }

    Array::save_safetensors([("key", state.rng)], None, staging.join("random.safetensors"))
        .context("writing the RNG state")?;

    std::fs::write(
        staging.join("manifest.json"),
        serde_json::to_vec_pretty(&Manifest::of(state)?)?,
    )
    .context("writing the manifest")
}

/// Lets a finished directory take the name, and makes that durable.
///
/// A rename over an existing directory is not atomic on this platform: the
/// old one has to go first, and a stop in between would leave neither. So
/// the previous checkpoint is moved aside, the new one takes the name in
/// one rename, and only then is the old one dropped.
fn publish(staging: &Path, dir: &Path) -> Result<()> {
    sync_dir(staging)?;

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
    std::fs::rename(staging, dir)
        .with_context(|| format!("renaming {} into place", staging.display()))?;
    if let Some(aside) = displaced {
        std::fs::remove_dir_all(aside).ok();
    }
    if let Some(parent) = dir.parent() {
        sync_dir(parent).ok();
    }
    Ok(())
}

/// Asks the filesystem to make a directory's entries durable, so that a
/// checkpoint that exists after a crash is one that reads back.
fn sync_dir(dir: &Path) -> Result<()> {
    let handle = std::fs::File::open(dir).with_context(|| format!("opening {}", dir.display()))?;
    handle
        .sync_all()
        .with_context(|| format!("syncing {}", dir.display()))
}

/// The machine, for a checkpoint carried between them.
fn platform() -> serde_json::Value {
    serde_json::json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
    })
}

/// What a checkpoint says about itself, as JSON: for a caller deciding
/// which one to resume from, without opening it into a session.
pub fn read_manifest_json(dir: &str) -> Result<String> {
    Ok(serde_json::to_string(&read_manifest(dir)?)?)
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
    // Read to be checked against the manifest, not to be handed on: what
    // the description says is the graph's business, and a caller that
    // wants it reads the file (Torobi::Checkpoint.graph_json).
    let graph_json = std::fs::read_to_string(dir.join(GRAPH_FILE))
        .with_context(|| format!("reading {}", dir.join(GRAPH_FILE).display()))?;
    {
        use sha2::{Digest, Sha256};
        let digest = format!("{:x}", Sha256::digest(graph_json.as_bytes()));
        anyhow::ensure!(
            digest == manifest.config_digest,
            "{GRAPH_FILE} is not the description this manifest claims \
             (digest {}, manifest says {})",
            &digest[..12],
            &manifest.config_digest[..12.min(manifest.config_digest.len())]
        );
    }
    let parameters = Array::load_safetensors(dir.join("parameters.safetensors"))
        .context("reading parameters")?;
    for entry in &manifest.parameters {
        let array = parameters
            .get(&entry.path)
            .with_context(|| format!("the manifest lists {:?}, the file has none", entry.path))?;
        anyhow::ensure!(
            array.shape() == entry.shape,
            "parameter {:?}: the file holds {:?}, the manifest says {:?}",
            entry.path,
            array.shape(),
            entry.shape
        );
        let dtype = dtype_spelling(array.dtype()).unwrap_or("(unnameable)");
        anyhow::ensure!(
            dtype == entry.dtype,
            "parameter {:?}: the file holds {dtype}, the manifest says {}",
            entry.path,
            entry.dtype
        );
    }

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
