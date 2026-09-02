//! The narrow waist between Ruby and the engine.
//!
//! Ruby holds one object, a session. It can start one, run steps on it,
//! read numbers off it, turn its knobs, and copy a parameter or a gradient
//! out by name. No tensor handles cross this line, no Ruby block ever runs
//! inside a computation, and the only unsafe code is `gvl`.

mod gvl;

use std::cell::RefCell;

use magnus::{function, method, prelude::*, Error, RArray, Ruby};
use torobi_engine::session::Tensor;
use torobi_engine::Session as EngineSession;

/// The engine's session, owned by one Ruby object.
///
/// `RefCell` rather than a lock: a session is a single conversation, as
/// documented on the Ruby side, and a second thread reaching in while a
/// step runs is a caller error we would rather name than hide.
#[magnus::wrap(class = "Torobi::Native::Session", free_immediately, size)]
struct Session(RefCell<EngineSession>);

fn to_error(ruby: &Ruby, error: anyhow::Error) -> Error {
    Error::new(ruby.exception_runtime_error(), format!("{error:#}"))
}

impl Session {
    fn open(ruby: &Ruby, graph_json: String, bindings_json: String) -> Result<Self, Error> {
        EngineSession::open(&graph_json, &bindings_json)
            .map(|session| Self(RefCell::new(session)))
            .map_err(|e| to_error(ruby, e))
    }

    fn borrow_mut(&self, ruby: &Ruby) -> Result<std::cell::RefMut<'_, EngineSession>, Error> {
        self.0.try_borrow_mut().map_err(|_| {
            Error::new(
                ruby.exception_runtime_error(),
                "this session is already running a step; one session serves one thread",
            )
        })
    }

    /// Runs `n` steps with the GVL released.
    fn run_steps(ruby: &Ruby, rb_self: &Self, n: usize) -> Result<f32, Error> {
        let mut session = rb_self.borrow_mut(ruby)?;
        // No Ruby API is touched inside: the engine only sees its own data.
        gvl::without_gvl(|| session.run_steps(n)).map_err(|e| to_error(ruby, e))
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

    fn fetch(ruby: &Ruby, rb_self: &Self, path: String) -> Result<(RArray, RArray), Error> {
        let tensor = rb_self
            .0
            .borrow()
            .fetch(&path)
            .map_err(|e| to_error(ruby, e))?;
        Ok(tensor_to_ruby(ruby, tensor))
    }

    fn gradients(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let grads = {
            let session = rb_self.0.borrow();
            gvl::without_gvl(|| session.gradients()).map_err(|e| to_error(ruby, e))?
        };
        let out = ruby.ary_new_capa(grads.len());
        for (path, tensor) in grads {
            let (shape, data) = tensor_to_ruby(ruby, tensor);
            out.push((path, shape, data))?;
        }
        Ok(out)
    }
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
    class.define_method("run_steps", method!(Session::run_steps, 1))?;
    class.define_method("step", method!(Session::step, 0))?;
    class.define_method("loss", method!(Session::loss, 0))?;
    class.define_method("lr", method!(Session::lr, 0))?;
    class.define_method("lr=", method!(Session::set_lr, 1))?;
    class.define_method("parameter_paths", method!(Session::parameter_paths, 0))?;
    class.define_method("fetch", method!(Session::fetch, 1))?;
    class.define_method("gradients", method!(Session::gradients, 0))?;
    Ok(())
}
