//! The narrow waist between Ruby and the engine.
//!
//! Ruby holds one object, a session. It can start one, run steps on it,
//! read numbers off it, turn its knobs, and copy a parameter or a gradient
//! out by name. No tensor handles cross this line, no Ruby block ever runs
//! inside a computation, and the only unsafe code is `gvl`.

mod gvl;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use magnus::exception::ExceptionClass;
use magnus::value::ReprValue;
use magnus::{function, method, prelude::*, Error, RArray, RHash, RString, Ruby, Value};
use torobi_engine::tensor::{unpack, Batch, PackedBatch, PackedTensor, Tensor, Values};
use torobi_engine::Session as EngineSession;

/// One MLX, one command queue.
///
/// A session's mutex keeps one thread inside one session; it says nothing
/// about two sessions. Two threads running two sessions submit to the same
/// MLX default stream at once, and Metal ends the process for it
/// ("commit command buffer with uncommitted encoder", reproduced). So
/// everything that runs MLX takes this first, whichever session it belongs
/// to, and two sessions in a process take turns rather than race
/// (notes/SESSION_CONCURRENCY_SPEC.md section 9).
///
/// Held only with the GVL released, so waiting on it never stops other
/// Ruby threads.
static MLX: Mutex<()> = Mutex::new(());

/// The process this extension was loaded in.
///
/// A Metal device does not survive fork. Ruby's preflight refuses to open
/// a session in a child, but it cannot speak for a session the parent
/// already opened, nor for anyone calling `Torobi::Native` directly. This
/// is the check that can (the spec, sections 3 and 9).
static ORIGIN_PID: OnceLock<u32> = OnceLock::new();

/// Whether this is the process that loaded the extension.
fn native_process() -> bool {
    ORIGIN_PID.get().is_none_or(|origin| *origin == std::process::id())
}

fn foreign_process(ruby: &Ruby) -> Error {
    Error::new(
        error_class(ruby, "EngineUnavailable"),
        "this process is not the one that loaded Torobi (a fork). A Metal \
         device does not survive fork: open a session in the child, or run \
         the work in a process of its own.",
    )
}

/// What a refused entry into the blocking region is allowed to do.
enum OnRefusal {
    /// Hand control back to Ruby, which raises whatever it was holding,
    /// then try again. What ordinary work does: the caller asked for an
    /// interrupt and should get it.
    AskRuby,
    /// Never raise. Take the gate with the GVL held instead.
    ///
    /// `close` needs this. It runs from `ensure`, where raising would
    /// replace the exception being handled, and where the unwind would
    /// drop the engine outside the gate, freeing device memory beside
    /// another session's step.
    PressOn,
}

/// Runs MLX work with the GVL released and the gate held.
///
/// `work` waits in an Option rather than being captured by value: the
/// blocking region can refuse to start, dropping the closure with whatever
/// it holds, and for `close` that would be the engine itself. Returns the
/// message of a panic inside `work` rather than resuming it.
fn in_mlx<W, R>(work: W, refusal: OnRefusal) -> Result<Ran2<R>, Error>
where
    W: FnOnce() -> R,
{
    let attempts = match refusal {
        OnRefusal::AskRuby => ENTRY_ATTEMPTS,
        OnRefusal::PressOn => 1,
    };
    let mut pending = Some(work);
    for _ in 0..attempts {
        let outcome = gvl::without_gvl(|| {
            let work = pending.take().expect("the region is entered at most once");
            let _gate = MLX.lock().unwrap_or_else(|e| e.into_inner());
            work()
        });
        match outcome {
            gvl::Outcome::Done(value) => return Ok(Ran2::Done(value)),
            gvl::Outcome::Panicked(what) => return Ok(Ran2::Panicked(what)),
            gvl::Outcome::Interrupted => {
                // The region was never entered, so nothing ran and nothing
                // was taken. The work is still here, in `pending`.
                if matches!(refusal, OnRefusal::AskRuby) {
                    // Let Ruby act on whatever it was flagging. Through
                    // `protect`, because `rb_thread_check_ints` raises by
                    // longjmp, and a longjmp through this frame would skip
                    // every Rust destructor between here and Ruby: the
                    // work, and the batch it borrows. `protect` turns that
                    // into a value, so the return is an ordinary one and
                    // everything is dropped on the way out.
                    magnus::rb_sys::protect(|| {
                        unsafe { rb_sys::rb_thread_check_ints() };
                        rb_sys::special_consts::Qnil as rb_sys::VALUE
                    })?;
                    // Nothing was raised, so the flag was the scheduler's
                    // timer, which that call has now cleared. Try again.
                }
            }
        }
    }

    // Ruby's flag stayed set through every round. Do the work with the GVL
    // held rather than refuse it: it stalls the other Ruby threads for one
    // call, but nothing is lost and this terminates (see the spec, 9.1,
    // which names this and the GC's cleanup as the two exceptions to
    // taking the gate with the GVL released).
    let work = pending.take().expect("no attempt entered the region");
    let _gate = MLX.lock().unwrap_or_else(|e| e.into_inner());
    match catch_unwind(AssertUnwindSafe(work)) {
        Ok(value) => Ok(Ran2::Done(value)),
        Err(panic) => Ok(Ran2::Panicked(gvl::describe(panic))),
    }
}

/// The engine's session, owned by one Ruby object.
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
    /// The process this session was opened in. Checked before every use,
    /// because a session that crossed a fork must not reach MLX.
    origin_pid: u32,
    /// Whether a call that runs MLX is in progress for this session, which
    /// includes waiting for the gate.
    ///
    /// Not the same as `Slot::Running`, which says the engine is out of
    /// the slot. A second thread must be told the session is busy rather
    /// than queued behind the gate and served afterwards: one session is
    /// one conversation (the spec, section 1), and two threads taking
    /// turns at one would interleave a run without saying so.
    in_flight: AtomicBool,
    slot: Mutex<Slot>,
    /// The numbers a watcher reads, kept apart from the engine so that
    /// reading them never waits on a step. Its lock is held for a copy and
    /// never across the GVL boundary.
    snapshot: Mutex<Snapshot>,
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
    fn of(engine: &EngineSession) -> Self {
        Self {
            step: engine.step(),
            loss: engine.loss(),
            lr: engine.lr(),
            seed: engine.seed(),
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

/// The GC frees device memory as surely as `close` does.
///
/// Without this, forgetting to close (or an exception between `open` and
/// the block that would have closed it) let Ruby's sweep drop the engine
/// wherever it happened to run, beside another session's step. The gate is
/// taken with the GVL held here, which the spec names as its second
/// exception (9.1): a sweep cannot hand the GVL back, and the thread it
/// waits for does not want it.
impl Drop for Session {
    fn drop(&mut self) {
        let engine = match self.slot.get_mut() {
            Ok(slot) => match std::mem::replace(slot, Slot::Closed) {
                Slot::Ready(engine) => engine,
                _ => return,
            },
            Err(_) => return,
        };
        if self.origin_pid != std::process::id() {
            // The device those arrays belong to did not survive the fork.
            std::mem::forget(engine);
            return;
        }
        let _gate = MLX.lock().unwrap_or_else(|e| e.into_inner());
        drop(engine);
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

/// What a GVL-released call produced. `Panicked` is separate from the
/// engine's own errors because it is the session that is lost, not a step.
enum Ran2<R> {
    Done(R),
    Panicked(String),
}

/// What one call produced, decided where the GVL may be released.
enum Ran<T> {
    Finished(anyhow::Result<T>),
    Refused(Refusal),
    Panicked(String),
}

impl<T> Ran<T> {
    /// Turned into Ruby's terms, which happens only once the GVL is back.
    fn into_result(self, ruby: &Ruby) -> Result<T, Error> {
        match self {
            Ran::Finished(result) => result.map_err(|e| to_error(ruby, e)),
            Ran::Refused(refusal) => Err(refusal.into_error(ruby)),
            Ran::Panicked(what) => Err(poisoned(ruby, &what)),
        }
    }
}

/// How many times a call retries after the blocking region refuses to
/// start. `RB_NOGVL_INTR_FAIL` refuses whenever Ruby's interrupt flag is
/// set, and that flag is also the scheduler's timer, so most refusals mean
/// nothing at all. `rb_thread_check_ints` clears a timer and raises a real
/// interrupt; a handful of rounds is more than enough, and the bound is
/// there so this can never spin.
const ENTRY_ATTEMPTS: u32 = 8;

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
    /// `optimizer_json` names the update rule, e.g.
    /// {"kind":"adamw","lr":0.001}. Data, so a journal can record it.
    fn open(
        ruby: &Ruby,
        graph_json: String,
        weights_json: String,
        optimizer_json: String,
    ) -> Result<Self, Error> {
        if !native_process() {
            return Err(foreign_process(ruby));
        }
        let optimizer = serde_json::from_str(&optimizer_json).map_err(|e| {
            Error::new(ruby.exception_arg_error(), format!("bad optimizer: {e}"))
        })?;
        // Opening builds every parameter, the RNG key and the optimizer's
        // slots: MLX work, and it used to run beside another session's
        // step.
        let opened = match in_mlx(
            || EngineSession::open_with(&graph_json, &weights_json, optimizer),
            OnRefusal::AskRuby,
        )? {
            Ran2::Done(opened) => opened,
            Ran2::Panicked(what) => return Err(poisoned(ruby, &what)),
        };
        opened
            .map(|session| Self {
                origin_pid: std::process::id(),
                in_flight: AtomicBool::new(false),
                snapshot: Mutex::new(Snapshot::of(&session)),
                slot: Mutex::new(Slot::Ready(Box::new(session))),
            })
            .map_err(|e| to_error(ruby, e))
    }

    /// Runs `work` on the engine, or says why it cannot.
    ///
    /// This is the one place the state machine is enforced: a busy session
    /// answers rather than panics, a poisoned one refuses, and a panic
    /// inside the engine poisons rather than escaping.
    ///
    /// The shape matters as much as the rules
    /// (notes/SESSION_CONCURRENCY_SPEC.md 3.1, 5). The engine is taken from
    /// the slot, used, and put back entirely inside the GVL-released
    /// region. Before that region nothing is held; after it there is
    /// nothing to do but read a value. An interrupt raised at either edge
    /// therefore costs nothing, which is what a `MutexGuard` held across
    /// the boundary used to cost: it was never dropped, and the session
    /// stayed busy for good.
    fn with_engine<T>(
        &self,
        ruby: &Ruby,
        work: impl FnOnce(&mut EngineSession) -> anyhow::Result<T>,
    ) -> Result<T, Error> {
        // Before anything else, and before any lock: a session that has
        // crossed a fork must not reach MLX, and must not wait on a gate
        // that was held by a thread the fork did not bring along.
        if !self.here() {
            return Err(foreign_process(ruby));
        }

        // Claimed before the gate, so a second thread on this session is
        // told rather than queued behind it.
        let _claim = self.claim().ok_or_else(|| busy(ruby))?;

        match in_mlx(|| self.run(work), OnRefusal::AskRuby)? {
            Ran2::Done(ran) => ran.into_result(ruby),
            Ran2::Panicked(what) => Err(poisoned(ruby, &what)),
        }
    }

    /// The exception: work that provably touches no MLX, run with the GVL
    /// held and without the gate.
    ///
    /// Everything else goes through [`Session::with_engine`], because
    /// deciding per call site which one runs MLX is a decision that gets
    /// made wrongly. `seed=` looked like a setter and built an RNG key;
    /// `fetch` looked like a read and evaluated an array and copied it off
    /// the device. Both were bypassing the gate.
    ///
    /// What may use this: reading names the plan already holds (`taps`,
    /// `node_names`, `trainable`, `parameter_paths`, `input_names`),
    /// changing the tap set, and copying out the host-side tensors a
    /// previous pass already brought back (`tapped`). Nothing that builds,
    /// evaluates or frees an `Array`.
    fn on_cpu<T>(
        &self,
        ruby: &Ruby,
        work: impl FnOnce(&mut EngineSession) -> anyhow::Result<T>,
    ) -> Result<T, Error> {
        if !self.here() {
            return Err(foreign_process(ruby));
        }
        self.run(work).into_result(ruby)
    }

    /// Whether this session belongs to the running process.
    fn here(&self) -> bool {
        self.origin_pid == std::process::id() && native_process()
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
    /// the whole of what runs with the GVL released and the gate held.
    fn run<T>(&self, work: impl FnOnce(&mut EngineSession) -> anyhow::Result<T>) -> Ran<T> {
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
        let Ok(mut guard) = self.slot.try_lock() else {
            return Err(Refusal::Busy);
        };
        match std::mem::replace(&mut *guard, Slot::Running) {
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

    /// A name the plan already holds. See [`Session::on_cpu`].
    fn read<T>(&self, ruby: &Ruby, work: impl FnOnce(&EngineSession) -> T) -> Result<T, Error> {
        self.on_cpu(ruby, |engine| Ok(work(engine)))
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
    /// Freeing device memory is MLX work like any other, so it goes
    /// through the gate rather than happening wherever the guard is
    /// dropped. In a forked child it is not done at all: the device those
    /// arrays belong to did not come along, and the child is leaving.
    fn close(ruby: &Ruby, rb_self: &Self) -> Result<bool, Error> {
        // The pid first, before any lock: a mutex another thread held when
        // the fork happened is locked forever in the child, and `close`
        // runs from `ensure`, where refusing would be noise. A child marks
        // the session closed if it can and leaves the device alone.
        if !rb_self.here() {
            if let Ok(mut guard) = rb_self.slot.try_lock() {
                if let Slot::Ready(engine) = std::mem::replace(&mut *guard, Slot::Closed) {
                    // The device those arrays belong to did not come along.
                    std::mem::forget(engine);
                }
            }
            return Ok(false);
        }

        let engine = {
            let mut guard = rb_self.slot.try_lock().map_err(|_| busy(ruby))?;
            // A running session is not closed out from under its own step:
            // the caller closes it at a step boundary (the spec, section 8).
            if matches!(*guard, Slot::Running) {
                return Err(busy(ruby));
            }
            match std::mem::replace(&mut *guard, Slot::Closed) {
                Slot::Ready(engine) => Some(engine),
                _ => None,
            }
        };
        let was_live = engine.is_some();
        if let Some(engine) = engine {
            let _ = in_mlx(move || drop(engine), OnRefusal::PressOn)?;
        }
        Ok(was_live)
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
        rb_self.on_cpu(ruby, |engine| {
            engine.set_lr(lr);
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
        let one = ruby.hash_new();
        one.aset(path.clone(), tensor)?;
        let mut batch = read_batch(ruby, one)?;
        let tensor = batch
            .remove(&path)
            .ok_or_else(|| Error::new(ruby.exception_arg_error(), "no tensor given"))?;
        rb_self.with_engine(ruby, |engine| engine.put(&path, &tensor))
    }

    /// Watches a named value. Read-only, and in force from the next step.
    fn tap(ruby: &Ruby, rb_self: &Self, name: String, stat: String) -> Result<(), Error> {
        rb_self.on_cpu(ruby, |engine| engine.tap(&name, &stat))
    }

    fn untap(ruby: &Ruby, rb_self: &Self, name: String) -> Result<bool, Error> {
        rb_self.on_cpu(ruby, |engine| Ok(engine.untap(&name)))
    }

    fn taps(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let names = rb_self.read(ruby, |engine| engine.taps())?;
        Ok(ruby.ary_from_vec(names))
    }

    fn node_names(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let names = rb_self.read(ruby, |engine| engine.node_names())?;
        Ok(ruby.ary_from_vec(names))
    }

    /// What the last step's taps saw: [name, shape, data] each.
    fn tapped(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let seen = rb_self.on_cpu(ruby, |engine| Ok(engine.tapped()))?;
        let out = ruby.ary_new_capa(seen.len());
        for (name, tensor) in seen {
            let (shape, data) = tensor_to_ruby(ruby, tensor);
            out.push((name, shape, data))?;
        }
        Ok(out)
    }

    fn parameter_paths(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let paths = rb_self.read(ruby, |engine| engine.parameter_paths())?;
        Ok(ruby.ary_from_vec(paths))
    }

    fn input_names(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let names = rb_self.read(ruby, |engine| engine.input_names())?;
        Ok(ruby.ary_from_vec(names))
    }

    fn fetch(ruby: &Ruby, rb_self: &Self, path: String) -> Result<(RArray, RArray), Error> {
        let tensor = rb_self.with_engine(ruby, |engine| engine.fetch(&path))?;
        Ok(tensor_to_ruby(ruby, tensor))
    }

    fn gradients(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<RArray, Error> {
        let batch = read_batch(ruby, batch)?;
        let grads = rb_self.with_engine(ruby, |engine| engine.gradients(&batch))?;
        let out = ruby.ary_new_capa(grads.len());
        for (path, tensor) in grads {
            let (shape, data) = tensor_to_ruby(ruby, tensor);
            out.push((path, shape, data))?;
        }
        Ok(out)
    }
}

/// Reads {name => [dtype, shape, packed]} into the engine's batch. The
/// bytes are copied out of the Ruby string here, while the GVL is still
/// held; nothing borrowed from Ruby survives into the computation.
///
/// The dtype travels because a graph may declare an i32 input, which is
/// what an embedding reads (docs/plan.md section 5A.2).
fn read_batch(ruby: &Ruby, batch: RHash) -> Result<Batch, Error> {
    let bad = |what: String| Error::new(ruby.exception_arg_error(), what);
    let mut packed = PackedBatch::new();
    batch.foreach(|name: String, triple: RArray| {
        let dtype: String = triple
            .entry::<String>(0)
            .map_err(|e| bad(format!("input {name:?}: bad dtype ({e})")))?;
        let shape: Vec<i32> = triple
            .entry::<Vec<i32>>(1)
            .map_err(|e| bad(format!("input {name:?}: bad shape ({e})")))?;
        let data: RString = triple
            .entry::<RString>(2)
            .map_err(|e| bad(format!("input {name:?}: data must be a packed String ({e})")))?;
        // Safety: the bytes are copied immediately, under the GVL, and the
        // string is not modified in between.
        let bytes = unsafe { data.as_slice() }.to_vec();
        packed.insert(name, PackedTensor { dtype, shape, bytes });
        Ok(magnus::r_hash::ForEach::Continue)
    })?;
    unpack(&packed).map_err(|e| bad(format!("{e:#}")))
}

/// A tensor leaves as its shape and its flat data, both plain Ruby arrays.
fn tensor_to_ruby(ruby: &Ruby, tensor: Tensor) -> (RArray, RArray) {
    let data = match tensor.values {
        Values::F32(values) => ruby.ary_from_vec(values),
        Values::I32(values) => ruby.ary_from_vec(values),
    };
    (ruby.ary_from_vec(tensor.shape), data)
}

/// What the device is holding, as a Ruby Hash: active, cache, peak and the
/// limit, in bytes. Process-wide, because MLX's allocator is.
/// Runs one of MLX's process-global calls under the same gate a step
/// takes. These reach the allocator every running session is also using,
/// so they are not free to happen alongside one.
fn global_mlx<R>(ruby: &Ruby, work: impl FnOnce() -> anyhow::Result<R>) -> Result<R, Error> {
    if !native_process() {
        return Err(foreign_process(ruby));
    }
    match in_mlx(work, OnRefusal::AskRuby)? {
        Ran2::Done(result) => result.map_err(|e| to_error(ruby, e)),
        Ran2::Panicked(what) => Err(poisoned(ruby, &what)),
    }
}

fn memory(ruby: &Ruby) -> Result<Value, Error> {
    let report = global_mlx(ruby, torobi_engine::memory::report)?;
    json_to_ruby(ruby, report)
}

/// Frees what the allocator holds but is not using. Returns [before, after].
fn clear_cache(ruby: &Ruby) -> Result<(usize, usize), Error> {
    global_mlx(ruby, torobi_engine::memory::clear_cache)
}

/// Caps what this process may allocate on the device, in bytes; 0 lifts the
/// cap. Returns the cap now in force.
fn set_memory_limit(ruby: &Ruby, bytes: usize) -> Result<usize, Error> {
    global_mlx(ruby, || torobi_engine::memory::set_limit(bytes))
}

fn reset_peak_memory(ruby: &Ruby) -> Result<(), Error> {
    global_mlx(ruby, torobi_engine::memory::reset_peak)
}

fn json_to_ruby(ruby: &Ruby, value: serde_json::Value) -> Result<Value, Error> {
    ruby.class_object()
        .const_get::<_, magnus::RModule>("JSON")
        .and_then(|json| json.funcall("parse", (value.to_string(),)))
}

/// What the engine was built from, as a Ruby Hash. A journal records it,
/// so a run can say which build produced it.
fn build_info(ruby: &Ruby) -> Result<Value, Error> {
    json_to_ruby(ruby, torobi_engine::build_info())
}

/// What a checkpoint says about itself, without opening it into a session.
/// For a caller deciding which one to resume from, and for anyone asking
/// what a directory on disk actually holds.
fn checkpoint_manifest(ruby: &Ruby, dir: String) -> Result<Value, Error> {
    let json = torobi_engine::Session::read_manifest(&dir).map_err(|e| to_error(ruby, e))?;
    ruby.class_object()
        .const_get::<_, magnus::RModule>("JSON")
        .and_then(|module| module.funcall("parse", (json,)))
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    // Recorded here, before anything can fork: everything native checks it.
    let _ = ORIGIN_PID.set(std::process::id());
    let torobi = ruby.define_module("Torobi")?;
    let native = torobi.define_module("Native")?;
    native.define_singleton_method("build_info", function!(build_info, 0))?;
    native.define_singleton_method("checkpoint_manifest", function!(checkpoint_manifest, 1))?;
    native.define_singleton_method("memory", function!(memory, 0))?;
    native.define_singleton_method("clear_cache", function!(clear_cache, 0))?;
    native.define_singleton_method("memory_limit=", function!(set_memory_limit, 1))?;
    native.define_singleton_method("reset_peak_memory", function!(reset_peak_memory, 0))?;
    let class = native.define_class("Session", ruby.class_object())?;
    class.define_singleton_method("open", function!(Session::open, 3))?;
    class.define_method("run_step", method!(Session::run_step, 1))?;
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
    Ok(())
}
