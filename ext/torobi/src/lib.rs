//! The narrow waist between Ruby and the engine.
//!
//! Ruby holds one object, a session. It can start one, run steps on it,
//! read numbers off it, turn its knobs, and copy a parameter or a gradient
//! out by name. No tensor handles cross this line, no Ruby block ever runs
//! inside a computation, and the only unsafe code is `gvl`.

mod gvl;

use std::cell::RefCell;

use magnus::exception::ExceptionClass;
use magnus::value::ReprValue;
use magnus::{function, method, prelude::*, Error, RArray, RHash, RString, Ruby, Value};
use torobi_engine::session::{unpack, Batch, PackedBatch, PackedTensor, Tensor};
use torobi_engine::Session as EngineSession;

/// The engine's session, owned by one Ruby object.
///
/// `RefCell` rather than a lock: a session is a single conversation, as
/// documented on the Ruby side, and a second thread reaching in while a
/// step runs is a caller error we would rather name than hide.
#[magnus::wrap(class = "Torobi::Native::Session", free_immediately, size)]
struct Session(RefCell<EngineSession>);

/// The class a failed step raises. Defined on the Ruby side (errors.rb), so
/// that the pure-Ruby half has the same hierarchy without this extension;
/// looked up here rather than duplicated.
fn step_error_class(ruby: &Ruby) -> ExceptionClass {
    ruby.class_object()
        .const_get::<_, magnus::RModule>("Torobi")
        .and_then(|m| m.const_get::<_, ExceptionClass>("StepError"))
        .unwrap_or_else(|_| ruby.exception_runtime_error())
}

fn to_error(ruby: &Ruby, error: anyhow::Error) -> Error {
    Error::new(step_error_class(ruby), format!("{error:#}"))
}

impl Session {
    fn open(ruby: &Ruby, graph_json: String, weights_json: String) -> Result<Self, Error> {
        EngineSession::open(&graph_json, &weights_json)
            .map(|session| Self(RefCell::new(session)))
            .map_err(|e| to_error(ruby, e))
    }

    fn borrow_mut(&self, ruby: &Ruby) -> Result<std::cell::RefMut<'_, EngineSession>, Error> {
        self.0.try_borrow_mut().map_err(|_| {
            Error::new(
                step_error_class(ruby),
                "this session is already running a step; one session serves one thread",
            )
        })
    }

    /// One step on one batch, with the GVL released.
    ///
    /// The batch arrives as {name => [shape, packed_bytes]}: shapes as small
    /// integer arrays, payloads as native-endian f32 strings. Measurement
    /// chose this over JSON (docs/plan.md section 5A.2.1).
    fn run_step(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<f32, Error> {
        let batch = read_batch(ruby, batch)?;
        let mut session = rb_self.borrow_mut(ruby)?;
        // No Ruby API is touched inside: the engine only sees its own data.
        gvl::without_gvl(|| session.run_step(&batch)).map_err(|e| to_error(ruby, e))
    }

    /// A span: one step per batch, all of them read before the GVL is
    /// released, so the engine never asks anyone for data mid-span.
    fn run_steps(ruby: &Ruby, rb_self: &Self, batches: RArray) -> Result<f32, Error> {
        let batches: Vec<Batch> = batches
            .into_iter()
            .map(|value| read_batch(ruby, RHash::from_value(value).ok_or_else(|| {
                Error::new(ruby.exception_arg_error(), "each batch must be a Hash")
            })?))
            .collect::<Result<_, Error>>()?;
        let mut session = rb_self.borrow_mut(ruby)?;
        gvl::without_gvl(|| session.run_steps(&batches)).map_err(|e| to_error(ruby, e))
    }

    fn step(&self) -> usize {
        self.0.borrow().step()
    }

    fn loss(&self) -> f32 {
        self.0.borrow().loss()
    }

    fn lr(&self) -> f32 {
        self.0.borrow().lr()
    }

    fn set_lr(ruby: &Ruby, rb_self: &Self, lr: f32) -> Result<f32, Error> {
        rb_self.borrow_mut(ruby)?.set_lr(lr);
        Ok(lr)
    }

    fn parameter_paths(ruby: &Ruby, rb_self: &Self) -> RArray {
        ruby.ary_from_vec(rb_self.0.borrow().parameter_paths())
    }

    fn input_names(ruby: &Ruby, rb_self: &Self) -> RArray {
        ruby.ary_from_vec(rb_self.0.borrow().input_names())
    }

    fn fetch(ruby: &Ruby, rb_self: &Self, path: String) -> Result<(RArray, RArray), Error> {
        let tensor = rb_self
            .0
            .borrow()
            .fetch(&path)
            .map_err(|e| to_error(ruby, e))?;
        Ok(tensor_to_ruby(ruby, tensor))
    }

    fn gradients(ruby: &Ruby, rb_self: &Self, batch: RHash) -> Result<RArray, Error> {
        let batch = read_batch(ruby, batch)?;
        let grads = {
            let session = rb_self.0.borrow();
            gvl::without_gvl(|| session.gradients(&batch)).map_err(|e| to_error(ruby, e))?
        };
        let out = ruby.ary_new_capa(grads.len());
        for (path, tensor) in grads {
            let (shape, data) = tensor_to_ruby(ruby, tensor);
            out.push((path, shape, data))?;
        }
        Ok(out)
    }
}

/// Reads {name => [shape, packed]} into the engine's batch. The bytes are
/// copied out of the Ruby string here, while the GVL is still held; nothing
/// borrowed from Ruby survives into the computation.
fn read_batch(ruby: &Ruby, batch: RHash) -> Result<Batch, Error> {
    let bad = |what: String| Error::new(ruby.exception_arg_error(), what);
    let mut packed = PackedBatch::new();
    batch.foreach(|name: String, pair: RArray| {
        let shape: Vec<i32> = pair
            .entry::<Vec<i32>>(0)
            .map_err(|e| bad(format!("input {name:?}: bad shape ({e})")))?;
        let data: RString = pair
            .entry::<RString>(1)
            .map_err(|e| bad(format!("input {name:?}: data must be a packed String ({e})")))?;
        // Safety: the bytes are copied immediately, under the GVL, and the
        // string is not modified in between.
        let bytes = unsafe { data.as_slice() }.to_vec();
        packed.insert(name, PackedTensor { shape, bytes });
        Ok(magnus::r_hash::ForEach::Continue)
    })?;
    unpack(&packed).map_err(|e| bad(format!("{e:#}")))
}

/// A tensor leaves as two plain Ruby arrays: its shape and its flat data.
fn tensor_to_ruby(ruby: &Ruby, tensor: Tensor) -> (RArray, RArray) {
    (
        ruby.ary_from_vec(tensor.shape),
        ruby.ary_from_vec(tensor.data),
    )
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let torobi = ruby.define_module("Torobi")?;
    let native = torobi.define_module("Native")?;
    let class = native.define_class("Session", ruby.class_object())?;
    class.define_singleton_method("open", function!(Session::open, 2))?;
    class.define_method("run_step", method!(Session::run_step, 1))?;
    class.define_method("run_steps", method!(Session::run_steps, 1))?;
    class.define_method("step", method!(Session::step, 0))?;
    class.define_method("loss", method!(Session::loss, 0))?;
    class.define_method("lr", method!(Session::lr, 0))?;
    class.define_method("lr=", method!(Session::set_lr, 1))?;
    class.define_method("parameter_paths", method!(Session::parameter_paths, 0))?;
    class.define_method("input_names", method!(Session::input_names, 0))?;
    class.define_method("fetch", method!(Session::fetch, 1))?;
    class.define_method("gradients", method!(Session::gradients, 1))?;
    Ok(())
}
