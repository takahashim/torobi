//! The narrow waist between Ruby and the engine.
//!
//! Ruby holds one object, a session. It can start one, run steps on it,
//! read numbers off it, turn its knobs, and copy a parameter or a gradient
//! out by name. No tensor handles cross this line, no Ruby block ever runs
//! inside a computation, and the only unsafe code is `gvl`.
//!
//! # Every entry point goes through one of three doors
//!
//! A panic in Rust does not abort here: magnus catches it leaving a
//! `method!` and raises a Ruby `fatal`. But a `fatal` is not rescuable, so
//! a caller cannot tell one from a crash or decide anything about it.
//! Catching first is what makes it an error.
//!
//! | door | for | what it adds |
//! | --- | --- | --- |
//! | [`Session::with_engine`] | anything that uses a session's engine | the session's state machine (busy, closed, poisoned), the GVL, the engine's runtime |
//! | [`global`] | the process-wide MLX calls (`Torobi::Memory`) | the GVL and the engine's runtime |
//! | [`plainly`] | what touches no MLX (`build_info`, `checkpoint_manifest`) | the catch, and nothing else |
//!
//! Two methods are outside them on purpose: `closed?` and `poisoned?` take
//! this crate's own lock and match on a value. They reach no engine, run
//! no MLX, and have nothing in them that can panic, so a door would add a
//! closure and no safety.

mod gvl;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use magnus::exception::ExceptionClass;
use magnus::value::ReprValue;
use magnus::{function, method, prelude::*, Error, RArray, RHash, RString, Ruby, Value};
use torobi_engine::plan::Weights;
use torobi_engine::tensor::{Batch, Tensor};
use torobi_engine::{RuntimeError, Session as EngineSession};

use gvl::{OnRefusal, Outcome};

/// The engine's session, owned by one Ruby object.
///
/// What is left here is the Ruby runtime's half: converting values,
/// releasing the GVL, and saying that one Ruby session serves one thread.
/// Serializing MLX, refusing a forked child and freeing device memory are
/// the engine's, which owns them
/// (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md section 3).
///
/// A `Mutex` rather than a `RefCell`, and `try_lock` rather than `lock`.
/// The reason is the GVL: a step runs with it released, so a second Ruby
/// thread can reach this object while the first is inside MLX. A RefCell
/// panicked there, and a panic that escapes becomes a Ruby `fatal` that no
/// `rescue` catches. Blocking would be worse still: the second thread
/// would hold the GVL while waiting, and the first needs the GVL back to
/// return. So a session in use answers "busy" and stays alive.
///
/// The engine is taken *out* of the slot for the duration of a step, and
/// put back before the GVL is reacquired. Holding the guard across that
/// boundary is what an interrupt used to destroy: CRuby raises on the way
/// out of a blocking region, the Rust frame is skipped, and the guard is
/// never dropped, leaving the session permanently busy. Nothing may be
/// left to do after the boundary (notes/SESSION_CONCURRENCY_SPEC.md 0, 3.1).
#[magnus::wrap(class = "Torobi::Native::Session", free_immediately, size)]
struct Session {
    /// Whether a call that runs MLX is in progress for this session, which
    /// includes waiting for the engine's gate.
    ///
    /// Not the same as `Slot::Running`, which says the engine is out of
    /// the slot. A second thread must be told the session is busy rather
    /// than queued behind the engine's gate and served afterwards: one
    /// session is one conversation (the spec, section 1), and two threads
    /// taking turns at one would interleave a run without saying so.
    in_flight: AtomicBool,
    slot: Mutex<Slot>,
    /// The numbers a watcher reads, kept apart from the engine so that
    /// reading them never waits on a step. Its lock is held for a copy and
    /// never across the GVL boundary.
    snapshot: Mutex<Snapshot>,
    /// What the run is made of, which is settled when it opens.
    names: Names,
}

/// The names a run answers to: its parameters, the batch fields it reads,
/// and the nodes a tap may ask for.
///
/// Taken once, because the plan settles all three at open and nothing
/// moves them (docs/plan.md section 5A.1). Held here rather than asked of
/// the engine so that reading them is not a claim: a watcher wanting to
/// know what a run is made of would otherwise be told the session is
/// busy, which is true of the engine and not of the answer.
///
/// What is trained is not here. Freezing moves it, so it is state, and
/// state is the engine's to report.
struct Names {
    parameters: Vec<String>,
    inputs: Vec<String>,
    nodes: Vec<String>,
}

impl Names {
    /// Taken from an open session; the caller has one in hand.
    ///
    /// These cannot refuse: a session that has just opened is not closed,
    /// and none of the three reaches MLX. A default would be worse than a
    /// panic here, because an empty parameter list is a sentence about the
    /// run rather than a missing answer.
    fn of(engine: &EngineSession) -> Self {
        Self {
            parameters: engine
                .parameter_paths()
                .expect("an open session answers parameter_paths"),
            inputs: engine
                .input_names()
                .expect("an open session answers input_names"),
            nodes: engine
                .node_names()
                .expect("an open session answers node_names"),
        }
    }
}

/// What the last completed step left behind.
///
/// A session serves one thread, but watching it is not serving it: a
/// progress bar or a metrics thread wants the numbers, not the engine.
/// Answering "busy" to those was the shape the spec objected to, and it is
/// what this exists to avoid (notes/SESSION_CONCURRENCY_SPEC.md 3, 4).
///
/// Never a partial step. It is published after the engine has committed,
/// so a reader sees a state the run actually passed through.
#[derive(Clone, Copy)]
struct Snapshot {
    step: usize,
    loss: f32,
    lr: f32,
    seed: u64,
}

impl Snapshot {
    /// Taken from an open session; the caller has one in hand.
    ///
    /// None of the four can refuse (they read state and reach no MLX), and
    /// a default would be a lie a watcher could not tell from an answer:
    /// step 0 is where a run starts.
    fn of(engine: &EngineSession) -> Self {
        Self {
            step: engine.step().expect("an open session answers step"),
            loss: engine.loss().expect("an open session answers loss"),
            lr: engine.lr().expect("an open session answers lr"),
            seed: engine.seed().expect("an open session answers seed"),
        }
    }
}

/// Holds a session's claim for as long as one MLX call needs it.
struct Claim<'a>(&'a AtomicBool);

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// What a session is, from the outside.
enum Slot {
    /// Usable, and nobody is inside it.
    Ready(Box<EngineSession>),
    /// A step is running. The engine belongs to that call's Rust frame,
    /// not to this slot.
    Running,
    /// A panic escaped the engine. The state it left behind is not known
    /// to be consistent, so nothing more is attempted with it.
    Poisoned(String),
    /// Closed by its owner. Its device memory is gone.
    Closed,
}

/// Why a slot could not hand its engine over.
///
/// Data rather than a `magnus::Error`, because it is decided with the GVL
/// released, where no Ruby object may be touched.
enum Refusal {
    Busy,
    Poisoned(String),
    Closed,
}

impl Refusal {
    fn of(slot: &Slot) -> Self {
        match slot {
            Slot::Ready(_) => unreachable!("a ready slot refuses nothing"),
            Slot::Running => Refusal::Busy,
            Slot::Poisoned(what) => Refusal::Poisoned(what.clone()),
            Slot::Closed => Refusal::Closed,
        }
    }

    fn into_error(self, ruby: &Ruby) -> Error {
        match self {
            Refusal::Busy => busy(ruby),
            Refusal::Poisoned(what) => Error::new(
                error_class(ruby, "SessionPoisoned"),
                format!(
                    "this session is poisoned: the engine panicked ({what}), so its \
                     state cannot be trusted. Open a new one, from a checkpoint if \
                     you have it."
                ),
            ),
            Refusal::Closed => Error::new(
                error_class(ruby, "SessionClosed"),
                "this session is closed",
            ),
        }
    }
}

/// What one call produced, decided where the GVL may be released.
enum Ran<T> {
    Finished(Result<T, RuntimeError>),
    Refused(Refusal),
    Panicked(String),
}

impl<T> Ran<T> {
    /// Turned into Ruby's terms, which happens only once the GVL is back.
    fn into_result(self, ruby: &Ruby) -> Result<T, Error> {
        match self {
            Ran::Finished(result) => result.map_err(|e| from_engine(ruby, e)),
            Ran::Refused(refusal) => Err(refusal.into_error(ruby)),
            Ran::Panicked(what) => Err(poisoned(ruby, &what)),
        }
    }
}

/// An engine refusal, in Ruby's terms.
///
/// The engine says which layer refused; this decides which class carries
/// it. Nothing reads a message to find out (the plan, section 9).
fn from_engine(ruby: &Ruby, error: RuntimeError) -> Error {
    let class = match error {
        RuntimeError::ForeignProcess | RuntimeError::Unavailable(_) => "EngineUnavailable",
        RuntimeError::Poisoned => "RuntimePoisoned",
        RuntimeError::Engine(inner) => return to_error(ruby, inner),
    };
    Error::new(error_class(ruby, class), error.to_string())
}

const BUSY: &str = "this session is busy: it serves one thread at a time (a step is \
                    running). Use one session per thread, or a pool.";

fn to_error(ruby: &Ruby, error: anyhow::Error) -> Error {
    Error::new(step_error_class(ruby), format!("{error:#}"))
}

/// One of Torobi's error classes by name. They are defined on the Ruby side
/// (errors.rb), so that the pure-Ruby half has the same hierarchy without
/// this extension; looked up here rather than duplicated.
fn error_class(ruby: &Ruby, name: &str) -> ExceptionClass {
    ruby.class_object()
        .const_get::<_, magnus::RModule>("Torobi")
        .and_then(|m| m.const_get::<_, ExceptionClass>(name))
        .unwrap_or_else(|_| ruby.exception_runtime_error())
}

/// The class a failed step raises.
fn step_error_class(ruby: &Ruby) -> ExceptionClass {
    error_class(ruby, "StepError")
}

fn poisoned(ruby: &Ruby, what: &str) -> Error {
    Error::new(
        error_class(ruby, "SessionPoisoned"),
        format!("the engine panicked ({what}); this session is now poisoned"),
    )
}

fn busy(ruby: &Ruby) -> Error {
    Error::new(error_class(ruby, "Busy"), BUSY)
}

impl Session {
    /// Opens with the parameters given as values.
    ///
    /// `optimizer_json` names the update rule, e.g.
    /// {"kind":"adamw","lr":0.001}. Data, so a journal can record it.
    fn open(
        ruby: &Ruby,
        graph_json: String,
        weights_json: String,
        optimizer_json: String,
        seed: u64,
    ) -> Result<Self, Error> {
        Self::opened(
            ruby,
            &graph_json,
            Weights::Inline(&weights_json),
            &optimizer_json,
            seed,
        )
    }

    /// Opens with the parameters read from a safetensors file.
    ///
    /// A second entry point rather than one that inspects its argument:
    /// where parameters come from is a choice the caller makes, and a
    /// choice made by what type a value happens to be is a choice nobody
    /// wrote down.
    fn open_from_file(
        ruby: &Ruby,
        graph_json: String,
        path: String,
        optimizer_json: String,
        seed: u64,
    ) -> Result<Self, Error> {
        Self::opened(
            ruby,
            &graph_json,
            Weights::File(std::path::Path::new(&path)),
            &optimizer_json,
            seed,
        )
    }

    /// Opens with one published checkpoint per model, named by model.
    ///
    /// `files_json` is {"student": "path", ...}. A model published on its
    /// own does not know what this run will call it, so its tensors carry
    /// no model prefix; naming a file per model is what lets it be
    /// imported without renaming anything.
    /// `fresh_json` is the patterns naming parameters that are expected to
    /// be in no file and are built from their declarations instead.
    fn open_pretrained(
        ruby: &Ruby,
        graph_json: String,
        files_json: String,
        fresh_json: String,
        optimizer_json: String,
        seed: u64,
    ) -> Result<Self, Error> {
        let files: std::collections::BTreeMap<String, std::path::PathBuf> =
            serde_json::from_str(&files_json).map_err(|e| {
                Error::new(ruby.exception_arg_error(), format!("bad file list: {e}"))
            })?;
        let fresh: Vec<String> = serde_json::from_str(&fresh_json).map_err(|e| {
            Error::new(ruby.exception_arg_error(), format!("bad fresh list: {e}"))
        })?;
        Self::opened(
            ruby,
            &graph_json,
            Weights::Pretrained {
                files: &files,
                fresh: &fresh,
            },
            &optimizer_json,
            seed,
        )
    }

    fn opened(
        ruby: &Ruby,
        graph_json: &str,
        weights: Weights<'_>,
        optimizer_json: &str,
        seed: u64,
    ) -> Result<Self, Error> {
        let optimizer = serde_json::from_str(optimizer_json).map_err(|e| {
            Error::new(ruby.exception_arg_error(), format!("bad optimizer: {e}"))
        })?;
        // Opening builds every parameter, an RNG key and the optimizer's
        // slots. That it is MLX work, and therefore waits its turn, is the
        // engine's to know.
        let opened = match gvl::released(
            || EngineSession::open_with(graph_json, weights, optimizer, seed),
            OnRefusal::AskRuby,
        )? {
            Outcome::Done(opened) => opened,
            Outcome::Panicked(what) => return Err(poisoned(ruby, &what)),
            Outcome::Interrupted => unreachable!("released() falls back rather than refusing"),
        };
        opened
            .map(|session| Self {
                in_flight: AtomicBool::new(false),
                snapshot: Mutex::new(Snapshot::of(&session)),
                names: Names::of(&session),
                slot: Mutex::new(Slot::Ready(Box::new(session))),
            })
            .map_err(|e| from_engine(ruby, e))
    }

    /// Runs `work` on the engine, or says why it cannot.
    ///
    /// This is where the Ruby side's rules are enforced: a session already
    /// serving a thread answers "busy" rather than panicking, and a panic
    /// inside the engine poisons this session rather than escaping. What
    /// it does *not* decide is whether `work` touches MLX. That belongs to
    /// the engine, which serializes it and refuses a forked child before
    /// any of it runs (notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md section 3).
    ///
    /// The shape still matters (notes/SESSION_CONCURRENCY_SPEC.md 3.1, 5).
    /// The engine is taken from the slot, used, and put back entirely
    /// inside the GVL-released region. Before that region nothing is held;
    /// after it there is nothing to do but read a value. An interrupt
    /// raised at either edge therefore costs nothing, which is what a
    /// `MutexGuard` held across the boundary used to cost.
    fn with_engine<T>(
        &self,
        ruby: &Ruby,
        work: impl FnOnce(&mut EngineSession) -> Result<T, RuntimeError>,
    ) -> Result<T, Error> {
        // Claimed before the engine is asked, so a second thread on this
        // session is told rather than queued behind the engine's gate.
        let _claim = self.claim().ok_or_else(|| busy(ruby))?;

        match gvl::released(|| self.run(work), OnRefusal::AskRuby)? {
            Outcome::Done(ran) => ran.into_result(ruby),
            Outcome::Panicked(what) => Err(poisoned(ruby, &what)),
            Outcome::Interrupted => unreachable!("released() falls back rather than refusing"),
        }
    }

    /// Claims this session for one MLX call, or answers that someone else
    /// has it. Released when the returned guard is dropped, which happens
    /// on every path out: nothing here raises by longjmp.
    fn claim(&self) -> Option<Claim<'_>> {
        self.in_flight
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Claim(&self.in_flight))
    }

    /// Take the engine, use it, put it back. Touches no Ruby, so this is
    /// the whole of what runs with the GVL released.
    fn run<T>(
        &self,
        work: impl FnOnce(&mut EngineSession) -> Result<T, RuntimeError>,
    ) -> Ran<T> {
        let mut engine = match self.take() {
            Ok(engine) => engine,
            Err(refusal) => return Ran::Refused(refusal),
        };
        match catch_unwind(AssertUnwindSafe(|| work(&mut engine))) {
            Ok(result) => {
                // Published before the engine goes back, so a reader never
                // sees a Ready session whose numbers are a step stale.
                self.publish(&engine);
                self.put_back(Ok(engine));
                Ran::Finished(result)
            }
            Err(panic) => {
                let what = gvl::describe(panic);
                self.put_back(Err(what.clone()));
                Ran::Panicked(what)
            }
        }
    }

    /// Takes the engine out of the slot, leaving `Running` behind.
    fn take(&self) -> Result<Box<EngineSession>, Refusal> {
        self.take_leaving(Slot::Running)
    }

    /// Takes the engine out for good, leaving `Closed` behind.
    ///
    /// A session already closed answers `None` rather than refusing, so a
    /// second `close` is a no-op; anything else (running, poisoned) is a
    /// refusal like any other. Here rather than in `close` so that every
    /// transition of `Slot` is in this one pair of methods: a variant
    /// added later is a match this compiler checks, not a fourth place to
    /// remember.
    fn take_to_close(&self) -> Result<Option<Box<EngineSession>>, Refusal> {
        match self.take_leaving(Slot::Closed) {
            Ok(engine) => Ok(Some(engine)),
            Err(Refusal::Closed) => Ok(None),
            Err(refusal) => Err(refusal),
        }
    }

    /// The engine, if the slot is ready to give it up, leaving `next`
    /// behind. A slot that is not ready keeps what it holds and says why.
    fn take_leaving(&self, next: Slot) -> Result<Box<EngineSession>, Refusal> {
        let Ok(mut guard) = self.slot.try_lock() else {
            return Err(Refusal::Busy);
        };
        match std::mem::replace(&mut *guard, next) {
            Slot::Ready(engine) => Ok(engine),
            other => {
                let refusal = Refusal::of(&other);
                *guard = other;
                Err(refusal)
            }
        }
    }

    /// Copies the engine's numbers out for watchers to read.
    fn publish(&self, engine: &EngineSession) {
        let mut guard = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Snapshot::of(engine);
    }

    /// The last published numbers. Never refuses: a closed or poisoned
    /// session still says where it got to.
    fn watch(&self) -> Snapshot {
        *self.snapshot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Puts the engine back, or records why there is none to put back.
    ///
    /// Blocking rather than `try_lock`, and safe to block: the only other
    /// holders are Ruby threads whose critical section is pure Rust and
    /// never waits for the GVL.
    fn put_back(&self, engine: Result<Box<EngineSession>, String>) {
        let mut guard = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        *guard = match engine {
            Ok(engine) => Slot::Ready(engine),
            Err(panic) => Slot::Poisoned(panic),
        };
    }

    /// A reading. It goes the same way as everything else: whether it
    /// costs MLX is the engine's to know, not this crate's.
    fn read<T>(
        &self,
        ruby: &Ruby,
        work: impl FnOnce(&EngineSession) -> Result<T, RuntimeError>,
    ) -> Result<T, Error> {
        self.with_engine(ruby, |engine| work(engine))
    }

    /// One step on one batch, with the GVL released.
    ///
    /// The batch arrives as {name => [dtype, shape, packed_bytes]}: dtype
    /// and shape readable, payload as native-endian 4-byte values.
    /// Measurement chose this over JSON (docs/plan.md section 5A.2.1).
    fn run_step(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<f32, Error> {
        let batch = read_batch(ruby, batch)?;
        rb_self.with_engine(ruby, |engine| engine.run_step(&batch))
    }

    /// One batch's gradients added to what is waiting, with no step: the
    /// parameters do not move until `apply`. What a batch too large to
    /// hold is trained as several of.
    fn accumulate(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<f32, Error> {
        let batch = read_batch(ruby, batch)?;
        rb_self.with_engine(ruby, |engine| engine.accumulate(&batch))
    }

    /// The step the accumulated gradients ask for.
    fn apply(ruby: &Ruby, rb_self: &Self) -> Result<f32, Error> {
        rb_self.with_engine(ruby, |engine| engine.apply())
    }

    /// How many parts are waiting for a step.
    fn accumulated(ruby: &Ruby, rb_self: &Self) -> Result<usize, Error> {
        rb_self.read(ruby, |engine| engine.accumulated())
    }

    /// Throws away what is waiting, and says how many parts went.
    fn discard(ruby: &Ruby, rb_self: &Self) -> Result<usize, Error> {
        rb_self.with_engine(ruby, |engine| engine.discard())
    }

    /// The loss for one batch without taking a step: no gradients, no
    /// randomness, nothing moved. What a validation set is read with.
    fn evaluate(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<f32, Error> {
        let batch = read_batch(ruby, batch)?;
        rb_self.with_engine(ruby, |engine| engine.evaluate(&batch))
    }

    /// Writes the run's state and returns where it landed. `run` is the
    /// caller's own record (epoch, batch position, sampler state) as JSON;
    /// the engine writes it verbatim and never reads it.
    fn save(ruby: &Ruby, rb_self: &Self, dir: String, run: String) -> Result<String, Error> {
        rb_self.with_engine(ruby, |engine| engine.save(&dir, &run))
    }

    /// Restores state written by `save`, refusing what does not belong.
    /// Returns the caller's record as JSON.
    fn restore(ruby: &Ruby, rb_self: &Self, dir: String) -> Result<String, Error> {
        rb_self.with_engine(ruby, |engine| engine.restore(&dir))
    }

    /// Releases the engine and its device memory. Idempotent, and the
    /// session refuses everything afterwards rather than pretending.
    ///
    /// The freeing itself is the engine's: it takes its own gate, and in a
    /// forked child it leaks rather than talking to a device that did not
    /// come along. What is decided here is only that `close` runs from
    /// `ensure`, where raising would replace the exception being handled,
    /// so a refusal comes back as `false` rather than as an error.
    fn close(ruby: &Ruby, rb_self: &Self) -> Result<bool, Error> {
        // A running session is not closed out from under its own step: the
        // caller closes it at a step boundary (the spec, section 8), and
        // the slot answers "busy" for one that is.
        let Some(mut engine) = rb_self
            .take_to_close()
            .map_err(|refusal| refusal.into_error(ruby))?
        else {
            return Ok(false);
        };
        match gvl::released(move || engine.close(), OnRefusal::PressOn)? {
            // A forked child is told nothing: it is leaving, and its
            // `ensure` should not be made to raise.
            Outcome::Done(closed) => Ok(closed.unwrap_or(false)),
            Outcome::Panicked(what) => Err(poisoned(ruby, &what)),
            Outcome::Interrupted => unreachable!("released() falls back rather than refusing"),
        }
    }

    fn closed(&self) -> bool {
        match self.slot.try_lock() {
            Ok(guard) => matches!(*guard, Slot::Closed),
            // In use, therefore not closed.
            Err(_) => false,
        }
    }

    fn poisoned(&self) -> bool {
        match self.slot.try_lock() {
            Ok(guard) => matches!(*guard, Slot::Poisoned(_)),
            Err(_) => false,
        }
    }

    fn step(&self) -> usize {
        self.watch().step
    }

    fn loss(&self) -> f32 {
        self.watch().loss
    }

    fn lr(&self) -> f32 {
        self.watch().lr
    }

    fn set_lr(ruby: &Ruby, rb_self: &Self, lr: f32) -> Result<f32, Error> {
        rb_self.with_engine(ruby, |engine| {
            engine.set_lr(lr)?;
            Ok(lr)
        })
    }

    fn seed(&self) -> u64 {
        self.watch().seed
    }

    fn set_seed(ruby: &Ruby, rb_self: &Self, seed: u64) -> Result<u64, Error> {
        rb_self.with_engine(ruby, |engine| {
            engine.set_seed(seed)?;
            Ok(seed)
        })
    }

    /// Freezes or unfreezes what matches `pattern`; returns what moved.
    fn set_frozen(
        ruby: &Ruby,
        rb_self: &Self,
        pattern: String,
        frozen: bool,
    ) -> Result<RArray, Error> {
        let moved = rb_self.with_engine(ruby, |engine| {
            engine.set_frozen(&pattern, frozen)
        })?;
        Ok(ruby.ary_from_vec(moved))
    }

    fn trainable(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let paths = rb_self.read(ruby, |engine| engine.trainable())?;
        Ok(ruby.ary_from_vec(paths))
    }

    /// Writes one parameter from a copy: [dtype, shape, packed].
    fn put(ruby: &Ruby, rb_self: &Self, path: String, tensor: RArray) -> Result<(), Error> {
        let tensor = read_tensor(ruby, &path, tensor)?;
        rb_self.with_engine(ruby, |engine| engine.put(&path, &tensor))
    }

    /// Watches a named value. Read-only, and in force from the next step.
    fn tap(ruby: &Ruby, rb_self: &Self, name: String, stat: String) -> Result<(), Error> {
        rb_self.with_engine(ruby, |engine| engine.tap(&name, &stat))
    }

    fn untap(ruby: &Ruby, rb_self: &Self, name: String) -> Result<bool, Error> {
        rb_self.with_engine(ruby, |engine| engine.untap(&name))
    }

    fn taps(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let names = rb_self.read(ruby, |engine| engine.taps())?;
        Ok(ruby.ary_from_vec(names))
    }

    /// Every name a tap may ask for. Answered from what was taken at
    /// open, so a watcher can ask while a step is in flight.
    fn node_names(ruby: &Ruby, rb_self: &Self) -> RArray {
        ruby.ary_from_vec(rb_self.names.nodes.clone())
    }

    /// What the last step's taps saw: [name, dtype, shape, bytes] each.
    fn tapped(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let seen = rb_self.with_engine(ruby, |engine| engine.tapped())?;
        tensors_to_ruby(ruby, seen)
    }

    fn parameter_paths(ruby: &Ruby, rb_self: &Self) -> RArray {
        ruby.ary_from_vec(rb_self.names.parameters.clone())
    }

    fn input_names(ruby: &Ruby, rb_self: &Self) -> RArray {
        ruby.ary_from_vec(rb_self.names.inputs.clone())
    }

    fn fetch(ruby: &Ruby, rb_self: &Self, path: String) -> Result<(String, RArray, RString), Error> {
        let tensor = rb_self.with_engine(ruby, |engine| engine.fetch(&path))?;
        tensor_to_ruby(ruby, tensor)
    }

    /// Gradients with respect to named batch fields: [name, dtype, shape,
    /// bytes] each. The parameters do not move.
    fn field_gradients(
        ruby: &Ruby,
        rb_self: &Self,
        batch: RHash,
        of: Vec<String>,
    ) -> Result<RArray, Error> {
        let batch = read_batch(ruby, batch)?;
        let grads = rb_self.with_engine(ruby, |engine| engine.field_gradients(&batch, &of))?;
        tensors_to_ruby(ruby, grads)
    }

    fn gradients(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<RArray, Error> {
        let batch = read_batch(ruby, batch)?;
        let grads = rb_self.with_engine(ruby, |engine| engine.gradients(&batch))?;
        tensors_to_ruby(ruby, grads)
    }
}

/// Reads {name => [dtype, shape, packed]} into the engine's batch. The
/// bytes are copied out of the Ruby string here, while the GVL is still
/// held; nothing borrowed from Ruby survives into the computation.
///
/// The dtype travels because a graph may declare an i32 input, which is
/// what an embedding reads (docs/plan.md section 5A.2).
fn read_batch(ruby: &Ruby, batch: RHash) -> Result<Batch, Error> {
    let mut read = Batch::new();
    batch.foreach(|name: String, triple: RArray| {
        let tensor = read_tensor(ruby, &name, triple)?;
        read.insert(name, tensor);
        Ok(magnus::r_hash::ForEach::Continue)
    })?;
    Ok(read)
}

/// One [dtype, shape, packed] as the engine's tensor, `name` being what to
/// call it if it is wrong.
///
/// Separate from [`read_batch`] because a batch is not the only thing made
/// of these: `put` writes one parameter, and reading it should not mean
/// building a Ruby Hash to hand to a reader of batches.
fn read_tensor(ruby: &Ruby, name: &str, triple: RArray) -> Result<Tensor, Error> {
    let bad = |what: String| Error::new(ruby.exception_arg_error(), what);
    let dtype: String = triple
        .entry::<String>(0)
        .map_err(|e| bad(format!("input {name:?}: bad dtype ({e})")))?;
    let shape: Vec<i32> = triple
        .entry::<Vec<i32>>(1)
        .map_err(|e| bad(format!("input {name:?}: bad shape ({e})")))?;
    let data: RString = triple
        .entry::<RString>(2)
        .map_err(|e| bad(format!("input {name:?}: data must be a packed String ({e})")))?;
    // Safety: the bytes are read while the GVL is held and copied before it
    // is released. Borrowing them across that would be unsound rather than
    // merely fast: another thread can trigger a compacting GC, and Ruby is
    // free to move the string's buffer.
    let bytes = unsafe { data.as_slice() };
    Tensor::from_bytes(&dtype, shape, bytes, name).map_err(|e| bad(format!("{e:#}")))
}

/// A tensor on its way out: dtype, shape, and the bytes.
///
/// The same three the boundary carries inward, and bytes rather than
/// numbers for the same reason. The Ruby side wraps them in a
/// `Torobi::TensorData`, and turning them into numbers is `to_a`, where a
/// caller can see the cost.
fn tensor_to_ruby(ruby: &Ruby, tensor: Tensor) -> Result<(String, RArray, RString), Error> {
    let shape = ruby.ary_from_vec(tensor.shape.clone());
    let (dtype, bytes) = tensor.as_bytes().map_err(|e| to_error(ruby, e))?;
    Ok((dtype.to_string(), shape, ruby.str_from_slice(bytes)))
}

/// Named tensors on their way out: [name, dtype, shape, bytes] each.
///
/// Three things leave this way (what the taps saw, gradients by parameter,
/// gradients by field), and they leave the same way on purpose: a caller
/// reads all three into `Torobi::TensorData` with one piece of code.
fn tensors_to_ruby(ruby: &Ruby, tensors: Vec<(String, Tensor)>) -> Result<RArray, Error> {
    let out = ruby.ary_new_capa(tensors.len());
    for (name, tensor) in tensors {
        let (dtype, shape, bytes) = tensor_to_ruby(ruby, tensor)?;
        out.push((name, dtype, shape, bytes))?;
    }
    Ok(out)
}

/// Runs one of the engine's process-global calls with the GVL released.
/// The engine serializes them against every running session, because they
/// reach the allocator those sessions share.
fn global<R>(ruby: &Ruby, work: impl FnOnce() -> Result<R, RuntimeError>) -> Result<R, Error> {
    match gvl::released(work, OnRefusal::AskRuby)? {
        Outcome::Done(result) => result.map_err(|e| from_engine(ruby, e)),
        Outcome::Panicked(what) => Err(poisoned(ruby, &what)),
        Outcome::Interrupted => unreachable!("released() falls back rather than refusing"),
    }
}

fn memory(ruby: &Ruby) -> Result<Value, Error> {
    // The reading is MLX's and goes through the gate; turning it into a
    // Ruby Hash is not, and happens after.
    let report = global(ruby, torobi_engine::memory::Memory::report)?;
    plainly(ruby, || json_to_ruby(ruby, report))
}

/// Frees what the allocator holds but is not using. Returns [before, after].
fn clear_cache(ruby: &Ruby) -> Result<(usize, usize), Error> {
    global(ruby, torobi_engine::memory::Memory::clear_cache)
}

/// Caps what this process may allocate on the device, in bytes; 0 lifts the
/// cap. Returns the cap now in force.
fn set_memory_limit(ruby: &Ruby, bytes: usize) -> Result<usize, Error> {
    global(ruby, || torobi_engine::memory::Memory::set_limit(bytes))
}

fn reset_peak_memory(ruby: &Ruby) -> Result<(), Error> {
    global(ruby, torobi_engine::memory::Memory::reset_peak)
}

fn json_to_ruby(ruby: &Ruby, value: serde_json::Value) -> Result<Value, Error> {
    ruby.class_object()
        .const_get::<_, magnus::RModule>("JSON")
        .and_then(|json| json.funcall("parse", (value.to_string(),)))
}

/// Runs work that touches no MLX, turning a panic into an error a caller
/// can rescue.
///
/// magnus already catches a panic on its way out of a `method!`, so the
/// process does not abort. What it makes of it is a Ruby `fatal`, which no
/// `rescue` holds and which ends the run anyway. Catching first is what
/// turns that into a `Torobi::SessionPoisoned` a caller can see, decide
/// about, and log. burn-rb's `boundary` is the same move for the same
/// reason.
///
/// See the module comment for which door each entry point uses.
fn plainly<T>(ruby: &Ruby, work: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(panic) => Err(poisoned(ruby, &gvl::describe(panic))),
    }
}

/// What the engine was built from, as a Ruby Hash. A journal records it,
/// so a run can say which build produced it.
fn build_info(ruby: &Ruby) -> Result<Value, Error> {
    plainly(ruby, || json_to_ruby(ruby, torobi_engine::build_info()))
}

/// What a checkpoint says about itself, without opening it into a session.
/// For a caller deciding which one to resume from, and for anyone asking
/// what a directory on disk actually holds.
fn checkpoint_manifest(ruby: &Ruby, dir: String) -> Result<Value, Error> {
    plainly(ruby, || {
        let json =
            torobi_engine::checkpoint::read_manifest_json(&dir).map_err(|e| to_error(ruby, e))?;
        ruby.class_object()
            .const_get::<_, magnus::RModule>("JSON")
            .and_then(|module| module.funcall("parse", (json,)))
    })
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    // Recorded here, before anything can fork: everything native checks it.
    // The engine records the process it belongs to, so a fork after this
    // is refused even before anything has touched MLX.
    torobi_engine::initialize();
    let torobi = ruby.define_module("Torobi")?;
    let native = torobi.define_module("Native")?;
    native.define_singleton_method("build_info", function!(build_info, 0))?;
    native.define_singleton_method("checkpoint_manifest", function!(checkpoint_manifest, 1))?;
    native.define_singleton_method("memory", function!(memory, 0))?;
    native.define_singleton_method("clear_cache", function!(clear_cache, 0))?;
    native.define_singleton_method("memory_limit=", function!(set_memory_limit, 1))?;
    native.define_singleton_method("reset_peak_memory", function!(reset_peak_memory, 0))?;
    let class = native.define_class("Session", ruby.class_object())?;
    class.define_singleton_method("open", function!(Session::open, 4))?;
    class.define_singleton_method("open_from_file", function!(Session::open_from_file, 4))?;
    class.define_singleton_method("open_pretrained", function!(Session::open_pretrained, 5))?;
    class.define_method("run_step", method!(Session::run_step, 1))?;
    class.define_method("accumulate", method!(Session::accumulate, 1))?;
    class.define_method("apply", method!(Session::apply, 0))?;
    class.define_method("accumulated", method!(Session::accumulated, 0))?;
    class.define_method("discard", method!(Session::discard, 0))?;
    class.define_method("evaluate", method!(Session::evaluate, 1))?;
    class.define_method("save", method!(Session::save, 2))?;
    class.define_method("restore", method!(Session::restore, 1))?;
    class.define_method("close", method!(Session::close, 0))?;
    class.define_method("closed?", method!(Session::closed, 0))?;
    class.define_method("poisoned?", method!(Session::poisoned, 0))?;
    class.define_method("step", method!(Session::step, 0))?;
    class.define_method("loss", method!(Session::loss, 0))?;
    class.define_method("lr", method!(Session::lr, 0))?;
    class.define_method("lr=", method!(Session::set_lr, 1))?;
    class.define_method("seed", method!(Session::seed, 0))?;
    class.define_method("seed=", method!(Session::set_seed, 1))?;
    class.define_method("set_frozen", method!(Session::set_frozen, 2))?;
    class.define_method("trainable", method!(Session::trainable, 0))?;
    class.define_method("put", method!(Session::put, 2))?;
    class.define_method("tap_node", method!(Session::tap, 2))?;
    class.define_method("untap", method!(Session::untap, 1))?;
    class.define_method("taps", method!(Session::taps, 0))?;
    class.define_method("node_names", method!(Session::node_names, 0))?;
    class.define_method("tapped", method!(Session::tapped, 0))?;
    class.define_method("parameter_paths", method!(Session::parameter_paths, 0))?;
    class.define_method("input_names", method!(Session::input_names, 0))?;
    class.define_method("fetch", method!(Session::fetch, 1))?;
    class.define_method("gradients", method!(Session::gradients, 1))?;
    class.define_method("field_gradients", method!(Session::field_gradients, 2))?;
    Ok(())
}
