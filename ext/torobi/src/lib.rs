//! The narrow waist between Ruby and the engine.
//!
//! Ruby holds one object, a session. It can start one, run steps on it,
//! read numbers off it, turn its knobs, and copy a parameter or a gradient
//! out by name. No tensor handles cross this line, no Ruby block ever runs
//! inside a computation, and the only unsafe code is `gvl`.

mod gvl;

use std::sync::Mutex;

use magnus::exception::ExceptionClass;
use magnus::value::ReprValue;
use magnus::{function, method, prelude::*, Error, RArray, RHash, RString, Ruby, Value};
use torobi_engine::tensor::{unpack, Batch, PackedBatch, PackedTensor, Tensor, Values};
use torobi_engine::Session as EngineSession;

/// The engine's session, owned by one Ruby object.
///
/// A `Mutex` rather than a `RefCell`, and `try_lock` rather than `lock`.
/// The reason is the GVL: a step runs with it released, so a second Ruby
/// thread can reach this object while the first is inside MLX. A RefCell
/// panicked there, and a panic that escapes becomes a Ruby `fatal` that no
/// `rescue` catches. Blocking would be worse still: the second thread
/// would hold the GVL while waiting, and the first needs the GVL back to
/// return. So a session in use answers "busy" and stays alive.
#[magnus::wrap(class = "Torobi::Native::Session", free_immediately, size)]
struct Session(Mutex<State>);

/// What a session is, from the outside.
enum State {
    /// Usable. One conversation at a time, which the mutex enforces.
    Live(Box<EngineSession>),
    /// A panic escaped the engine. The state it left behind is not known
    /// to be consistent, so nothing more is attempted with it.
    Poisoned(String),
    /// Closed by its owner. Its device memory is gone.
    Closed,
}

impl State {
    fn engine(&mut self) -> Result<&mut EngineSession, String> {
        match self {
            State::Live(session) => Ok(session),
            State::Poisoned(what) => Err(format!(
                "this session is poisoned: the engine panicked ({what}), so its \
                 state cannot be trusted. Open a new one, from a checkpoint if \
                 you have it."
            )),
            State::Closed => Err("this session is closed".to_string()),
        }
    }
}

fn to_error(ruby: &Ruby, error: anyhow::Error) -> Error {
    Error::new(step_error_class(ruby), format!("{error:#}"))
}

/// The class a failed step raises. Defined on the Ruby side (errors.rb), so
/// that the pure-Ruby half has the same hierarchy without this extension;
/// looked up here rather than duplicated.
fn step_error_class(ruby: &Ruby) -> ExceptionClass {
    ruby.class_object()
        .const_get::<_, magnus::RModule>("Torobi")
        .and_then(|m| m.const_get::<_, ExceptionClass>("StepError"))
        .unwrap_or_else(|_| ruby.exception_runtime_error())
}

fn busy(ruby: &Ruby) -> Error {
    Error::new(
        step_error_class(ruby),
        "this session is busy: it serves one thread at a time (a step is \
         running). Use one session per thread, or a pool.",
    )
}

fn message(ruby: &Ruby, what: String) -> Error {
    Error::new(step_error_class(ruby), what)
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
        let optimizer = serde_json::from_str(&optimizer_json).map_err(|e| {
            Error::new(ruby.exception_arg_error(), format!("bad optimizer: {e}"))
        })?;
        EngineSession::open_with(&graph_json, &weights_json, optimizer)
            .map(|session| Self(Mutex::new(State::Live(Box::new(session)))))
            .map_err(|e| to_error(ruby, e))
    }

    /// Runs `work` on the engine, or says why it cannot.
    ///
    /// This is the one place the state machine is enforced: a busy session
    /// answers rather than panics, a poisoned one refuses, and a panic
    /// inside the engine poisons rather than escaping.
    fn with_engine<T>(
        &self,
        ruby: &Ruby,
        release_gvl: bool,
        work: impl FnOnce(&mut EngineSession) -> anyhow::Result<T>,
    ) -> Result<T, Error> {
        let mut guard = self.0.try_lock().map_err(|_| busy(ruby))?;
        let engine = guard.engine().map_err(|what| message(ruby, what))?;

        let outcome = if release_gvl {
            // Nothing inside touches the Ruby VM: the engine sees only its
            // own data.
            gvl::without_gvl(|| work(engine))
        } else {
            Ok(work(engine))
        };

        match outcome {
            Ok(result) => result.map_err(|e| to_error(ruby, e)),
            Err(panic) => {
                *guard = State::Poisoned(panic.clone());
                Err(message(
                    ruby,
                    format!("the engine panicked ({panic}); this session is now poisoned"),
                ))
            }
        }
    }

    /// A reading, which needs no GVL release and no engine mutation.
    fn read<T>(&self, ruby: &Ruby, work: impl FnOnce(&EngineSession) -> T) -> Result<T, Error> {
        self.with_engine(ruby, false, |engine| Ok(work(engine)))
    }

    /// One step on one batch, with the GVL released.
    ///
    /// The batch arrives as {name => [dtype, shape, packed_bytes]}: dtype
    /// and shape readable, payload as native-endian 4-byte values.
    /// Measurement chose this over JSON (docs/plan.md section 5A.2.1).
    fn run_step(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<f32, Error> {
        let batch = read_batch(ruby, batch)?;
        rb_self.with_engine(ruby, true, |engine| engine.run_step(&batch))
    }

    /// A span: one step per batch, all of them read before the GVL is
    /// released, so the engine never asks anyone for data mid-span.
    fn run_steps(ruby: &Ruby, rb_self: &Self, batches: RArray) -> Result<f32, Error> {
        let batches: Vec<Batch> = batches
            .into_iter()
            .map(|value| {
                read_batch(
                    ruby,
                    RHash::from_value(value).ok_or_else(|| {
                        Error::new(ruby.exception_arg_error(), "each batch must be a Hash")
                    })?,
                )
            })
            .collect::<Result<_, Error>>()?;
        rb_self.with_engine(ruby, true, |engine| engine.run_steps(&batches))
    }

    /// Writes the run's state and returns where it landed.
    fn save(ruby: &Ruby, rb_self: &Self, dir: String) -> Result<String, Error> {
        rb_self.with_engine(ruby, true, |engine| engine.save(&dir))
    }

    /// Restores state written by `save`, refusing what does not belong.
    fn restore(ruby: &Ruby, rb_self: &Self, dir: String) -> Result<(), Error> {
        rb_self.with_engine(ruby, true, |engine| engine.restore(&dir))
    }

    /// Releases the engine and its device memory. Idempotent, and the
    /// session refuses everything afterwards rather than pretending.
    fn close(ruby: &Ruby, rb_self: &Self) -> Result<bool, Error> {
        let mut guard = rb_self.0.try_lock().map_err(|_| busy(ruby))?;
        let was_live = matches!(*guard, State::Live(_));
        *guard = State::Closed;
        Ok(was_live)
    }

    fn closed(&self) -> bool {
        match self.0.try_lock() {
            Ok(guard) => matches!(*guard, State::Closed),
            // In use, therefore not closed.
            Err(_) => false,
        }
    }

    fn poisoned(&self) -> bool {
        match self.0.try_lock() {
            Ok(guard) => matches!(*guard, State::Poisoned(_)),
            Err(_) => false,
        }
    }

    fn step(ruby: &Ruby, rb_self: &Self) -> Result<usize, Error> {
        rb_self.read(ruby, |engine| engine.step())
    }

    fn loss(ruby: &Ruby, rb_self: &Self) -> Result<f32, Error> {
        rb_self.read(ruby, |engine| engine.loss())
    }

    fn lr(ruby: &Ruby, rb_self: &Self) -> Result<f32, Error> {
        rb_self.read(ruby, |engine| engine.lr())
    }

    fn set_lr(ruby: &Ruby, rb_self: &Self, lr: f32) -> Result<f32, Error> {
        rb_self.with_engine(ruby, false, |engine| {
            engine.set_lr(lr);
            Ok(lr)
        })
    }

    fn seed(ruby: &Ruby, rb_self: &Self) -> Result<u64, Error> {
        rb_self.read(ruby, |engine| engine.seed())
    }

    fn set_seed(ruby: &Ruby, rb_self: &Self, seed: u64) -> Result<u64, Error> {
        rb_self.with_engine(ruby, false, |engine| {
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
        let moved = rb_self.with_engine(ruby, true, |engine| {
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
        rb_self.with_engine(ruby, true, |engine| engine.put(&path, &tensor))
    }

    /// Watches a named value. Read-only, and in force from the next step.
    fn tap(ruby: &Ruby, rb_self: &Self, name: String, stat: String) -> Result<(), Error> {
        rb_self.with_engine(ruby, false, |engine| engine.tap(&name, &stat))
    }

    fn untap(ruby: &Ruby, rb_self: &Self, name: String) -> Result<bool, Error> {
        rb_self.with_engine(ruby, false, |engine| Ok(engine.untap(&name)))
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
        let seen = rb_self.with_engine(ruby, false, |engine| engine.tapped())?;
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
        let tensor = rb_self.with_engine(ruby, false, |engine| engine.fetch(&path))?;
        Ok(tensor_to_ruby(ruby, tensor))
    }

    fn gradients(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<RArray, Error> {
        let batch = read_batch(ruby, batch)?;
        let grads = rb_self.with_engine(ruby, true, |engine| engine.gradients(&batch))?;
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
fn memory(ruby: &Ruby) -> Result<Value, Error> {
    let report = torobi_engine::memory::report().map_err(|e| to_error(ruby, e))?;
    json_to_ruby(ruby, report)
}

/// Frees what the allocator holds but is not using. Returns [before, after].
fn clear_cache(ruby: &Ruby) -> Result<(usize, usize), Error> {
    torobi_engine::memory::clear_cache().map_err(|e| to_error(ruby, e))
}

/// Caps what this process may allocate on the device, in bytes; 0 lifts the
/// cap. Returns the cap now in force.
fn set_memory_limit(ruby: &Ruby, bytes: usize) -> Result<usize, Error> {
    torobi_engine::memory::set_limit(bytes).map_err(|e| to_error(ruby, e))
}

fn reset_peak_memory(ruby: &Ruby) -> Result<(), Error> {
    torobi_engine::memory::reset_peak().map_err(|e| to_error(ruby, e))
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

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let torobi = ruby.define_module("Torobi")?;
    let native = torobi.define_module("Native")?;
    native.define_singleton_method("build_info", function!(build_info, 0))?;
    native.define_singleton_method("memory", function!(memory, 0))?;
    native.define_singleton_method("clear_cache", function!(clear_cache, 0))?;
    native.define_singleton_method("memory_limit=", function!(set_memory_limit, 1))?;
    native.define_singleton_method("reset_peak_memory", function!(reset_peak_memory, 0))?;
    let class = native.define_class("Session", ruby.class_object())?;
    class.define_singleton_method("open", function!(Session::open, 3))?;
    class.define_method("run_step", method!(Session::run_step, 1))?;
    class.define_method("run_steps", method!(Session::run_steps, 1))?;
    class.define_method("save", method!(Session::save, 1))?;
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
