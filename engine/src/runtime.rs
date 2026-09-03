//! The one MLX this process has.
//!
//! MLX's constraints are the engine's, not its callers'. There is one
//! device, one default stream, one command queue and one allocator per
//! process; two threads submitting to them at once ends the process, and a
//! forked child inherits none of them. Those facts belong here, where MLX
//! is actually owned, rather than in whatever happens to be calling
//! (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md).
//!
//! They used to live in the Ruby extension, which meant every binding had
//! to decide per call whether it was about to touch MLX. `seed=` looked
//! like a setter and built an RNG key; `fetch` looked like a read and
//! evaluated an array and copied it off the device. Both were wrong, and
//! the engine's own tests and command-line tool went through no gate at
//! all.
//!
//! The rule this replaces that with: **every public route to MLX passes
//! through [`execute`], and callers do not classify.**

use std::sync::{Mutex, MutexGuard, OnceLock};

/// What the runtime refuses, and why. Typed rather than a message: the
/// engine does not know Ruby's error classes, and the layer that does
/// should not read strings to find out which failure this was.
#[derive(Debug)]
pub enum RuntimeError {
    /// A fork. The device did not come along, and neither did whatever
    /// held the gate at the time.
    ForeignProcess,
    /// MLX could not run here even once: something it needs is not in
    /// place, and asking it anyway would end the process rather than
    /// return. The string says what is missing and where it was expected.
    Unavailable(String),
    /// A panic escaped an operation while it held the gate. MLX is
    /// process-global, so what that left behind is not knowable, and no
    /// further work is attempted in this process.
    Poisoned,
    /// The operation itself failed, and the runtime is fine.
    Engine(anyhow::Error),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuntimeError::ForeignProcess => write!(
                f,
                "this process is not the one that loaded the engine (a fork). \
                 A Metal device does not survive fork: open a session in the \
                 child, or run the work in a process of its own."
            ),
            RuntimeError::Unavailable(what) => write!(f, "{what}"),
            RuntimeError::Poisoned => write!(
                f,
                "the engine panicked while it held MLX, so nothing more is \
                 attempted in this process. Start a new one, from a \
                 checkpoint if you have it."
            ),
            RuntimeError::Engine(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<anyhow::Error> for RuntimeError {
    fn from(error: anyhow::Error) -> Self {
        RuntimeError::Engine(error)
    }
}

/// One per process, and no way to make a second: two runtimes would be two
/// mutexes, which is no exclusion at all.
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub(crate) fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(Runtime::new)
}

/// Records the process the engine belongs to.
///
/// Called once when a binding loads, so that a fork after loading is
/// refused even if nothing has touched MLX yet. Without it the runtime
/// would take its pid from whoever first ran an operation, which would
/// quietly bless a child that forked early. Builds nothing.
pub fn initialize() {
    let _ = runtime();
}

pub(crate) struct Runtime {
    origin_pid: u32,
    gate: Mutex<()>,
}

impl Runtime {
    fn new() -> Self {
        Self {
            origin_pid: std::process::id(),
            gate: Mutex::new(()),
        }
    }

    /// Runs one MLX operation, alone.
    ///
    /// Waiting, not refusing: two sessions in a process take turns, which
    /// is what one queue means. Refusing a second call on the *same*
    /// session is the caller's business, and has to happen before this,
    /// or the two would silently interleave.
    ///
    /// A panic is not caught here. It unwinds through the guard, which
    /// poisons the gate and closes this process to further MLX work; the
    /// binding above catches it and says which session it belonged to.
    /// Catching it here would leave the gate unpoisoned and the rule in
    /// `RuntimeError::Poisoned` unenforced.
    pub(crate) fn execute<R>(
        &self,
        operation: impl FnOnce() -> anyhow::Result<R>,
    ) -> Result<R, RuntimeError> {
        let _guard = self.enter()?;
        operation().map_err(RuntimeError::Engine)
    }

    /// The pid first, and before the gate. A mutex another thread held
    /// when the fork happened is locked forever in the child, so a child
    /// that waited would wait for good. Then whether MLX can run here at
    /// all, because the failure that answers otherwise is an abort, not an
    /// error.
    fn enter(&self) -> Result<MutexGuard<'_, ()>, RuntimeError> {
        self.here()?;
        available()?;
        self.gate.lock().map_err(|_| RuntimeError::Poisoned)
    }

    fn here(&self) -> Result<(), RuntimeError> {
        if self.origin_pid == std::process::id() {
            Ok(())
        } else {
            Err(RuntimeError::ForeignProcess)
        }
    }

    /// Frees what an operation cannot: the resources of something being
    /// dropped.
    ///
    /// Two things separate this from [`Runtime::execute`]. A poisoned gate
    /// still lets a release through, because refusing would leak device
    /// memory to no purpose. And in a forked child the value is leaked on
    /// purpose: the device those handles name did not come along, and
    /// freeing them would talk to a device that is not there.
    pub(crate) fn release<T>(&self, value: T) -> Result<(), RuntimeError> {
        if self.here().is_err() {
            std::mem::forget(value);
            return Err(RuntimeError::ForeignProcess);
        }
        let _guard = match self.gate.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        drop(value);
        Ok(())
    }

    /// Whether this process may still reach MLX at all. For a caller
    /// deciding whether to keep going or start over.
    pub(crate) fn usable(&self) -> bool {
        self.here().is_ok() && !self.gate.is_poisoned()
    }
}

/// Whether MLX can still be reached from this process.
pub fn usable() -> bool {
    runtime().usable()
}

/// Refuses, once and for all, if MLX could not start here.
///
/// The failure this prevents is not an error. MLX finds its Metal kernels
/// by asking `dladdr` where its own code lives and looking for
/// `mlx.metallib` next to it; if the file is not there, device
/// initialization throws a C++ exception that nothing in Rust can catch,
/// and the process exits with nothing for any caller to rescue
/// (docs/vendoring.md). So the same question is asked here first, the same
/// way, before MLX ever is.
///
/// Asked once per process: the file does not move while we run, and the
/// answer is the same for every session.
fn available() -> Result<(), RuntimeError> {
    static ANSWER: OnceLock<Option<String>> = OnceLock::new();
    match ANSWER.get_or_init(missing_metallib) {
        None => Ok(()),
        Some(what) => Err(RuntimeError::Unavailable(what.clone())),
    }
}

/// What is missing, if MLX would fail to find its kernels here.
#[cfg(target_os = "macos")]
fn missing_metallib() -> Option<String> {
    // An anchor in this object's data segment: dladdr on it names the
    // binary or bundle this code is linked into, which is exactly where
    // MLX will look (its code is statically linked into the same object).
    static ANCHOR: u8 = 0;
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let found =
        unsafe { libc::dladdr(std::ptr::addr_of!(ANCHOR) as *const libc::c_void, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        // Cannot tell where we are; refusing on ignorance would block a
        // working setup, so let MLX try.
        return None;
    }
    let object = unsafe { std::ffi::CStr::from_ptr(info.dli_fname) };
    let object = std::path::PathBuf::from(object.to_string_lossy().into_owned());
    let expected = object.parent()?.join("mlx.metallib");
    if expected.exists() {
        return None;
    }
    // Someone pointing MLX elsewhere knows more than this check does;
    // defer to them rather than refuse a setup that might work.
    if std::env::var_os("MLX_METAL_PATH").is_some() {
        return None;
    }
    Some(format!(
        "MLX's Metal kernels are not where it will look for them: expected \
         {} beside {}. Without that file MLX aborts the process rather than \
         raising, so this refuses first. The gem's install step puts it \
         there; in a checkout, run `rake metallib`.",
        expected.display(),
        object.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
    ))
}

#[cfg(not(target_os = "macos"))]
fn missing_metallib() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_process_owns_the_runtime() {
        assert!(usable(), "the tests run where the engine was loaded");
        assert!(runtime().here().is_ok());
    }

    #[test]
    fn the_metallib_is_looked_for_beside_this_object() {
        // The check has to agree with MLX, which asks dladdr where its own
        // code is. Ours is in the same object, so if it finds nothing to
        // complain about, MLX will find the file.
        assert!(
            missing_metallib().is_none(),
            "the test binary should have mlx.metallib beside it \
             (engine/build.rs links it into deps/): {:?}",
            missing_metallib()
        );
    }

    #[test]
    fn a_refusal_says_what_is_missing_and_where() {
        // The message is the whole value of refusing rather than aborting,
        // so its shape is worth pinning.
        let refusal = RuntimeError::Unavailable("no kernels at /x/mlx.metallib".into());
        let said = refusal.to_string();
        assert!(said.contains("/x/mlx.metallib"), "{said}");
    }

    #[test]
    fn the_layers_are_told_apart_rather_than_flattened() {
        // The binding maps these to different Ruby classes, so they must
        // not collapse into one another on the way up.
        let engine = RuntimeError::Engine(anyhow::anyhow!("a step failed"));
        assert!(matches!(engine, RuntimeError::Engine(_)));
        assert!(!matches!(RuntimeError::ForeignProcess, RuntimeError::Engine(_)));
        assert!(!matches!(RuntimeError::Poisoned, RuntimeError::ForeignProcess));
    }
}
