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
    /// that waited would wait for good.
    fn enter(&self) -> Result<MutexGuard<'_, ()>, RuntimeError> {
        self.here()?;
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
