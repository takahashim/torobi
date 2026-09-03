//! Releasing the GVL around the engine's long calls.
//!
//! This is the one piece of unsafe machinery in the binding, and it stays
//! small on purpose: the closure it runs must not touch the Ruby API, and
//! nothing else in this crate needs the GVL released.
//!
//! Two things make it more than a wrapper (notes/SESSION_CONCURRENCY_SPEC.md
//! section 5).
//!
//! A panic is caught and returned rather than resumed. magnus turns a panic
//! that escapes into a Ruby `fatal`, which no `rescue` catches and which
//! ends the process.
//!
//! And `rb_nogvl` with `RB_NOGVL_INTR_FAIL` is used rather than
//! `rb_thread_call_without_gvl`. The latter checks for pending interrupts
//! on the way out of the blocking region and raises there, which is a
//! longjmp through this frame: anything the caller still had to do (drop a
//! guard, record a step) would never happen. With the flag, the region
//! returns normally and Ruby delivers the interrupt at its next checkpoint,
//! by which time the caller has finished and the Ruby side has recorded the
//! step the engine actually took.

use std::any::Any;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// What came back from a GVL-released call.
pub enum Outcome<R> {
    Done(R),
    /// The closure panicked. Its message, as far as it can be recovered.
    Panicked(String),
    /// An interrupt was already pending, so the closure never ran. Nothing
    /// happened: no step was taken, and no state moved.
    Interrupted,
}

struct Payload<F, R> {
    call: Option<F>,
    outcome: Option<Result<R, Box<dyn Any + Send>>>,
}

unsafe extern "C" fn trampoline<F, R>(data: *mut c_void) -> *mut c_void
where
    F: FnOnce() -> R,
{
    let payload = &mut *(data as *mut Payload<F, R>);
    let call = payload.call.take().expect("the closure runs exactly once");
    payload.outcome = Some(catch_unwind(AssertUnwindSafe(call)));
    std::ptr::null_mut()
}

/// Runs `call` with the GVL released, so other Ruby threads proceed.
///
/// `call` must leave nothing for its caller to finish: whatever has to
/// happen for the session to stay consistent has to happen inside it.
/// Only reading a plain value out of the result is safe afterwards.
///
/// # Safety contract (upheld by the callers in this crate, not by the type
/// system): `call` must not touch the Ruby VM in any way.
pub fn without_gvl<F, R>(call: F) -> Outcome<R>
where
    F: FnOnce() -> R,
{
    let mut payload = Payload {
        call: Some(call),
        outcome: None,
    };
    unsafe {
        rb_sys::rb_nogvl(
            Some(trampoline::<F, R>),
            &mut payload as *mut Payload<F, R> as *mut c_void,
            // No unblocking function: a step is not interruptible from
            // outside. Ruby drives the loop one step at a time, so an
            // interrupt lands between steps, which is the same semantics a
            // UBF would buy here (the spec, section 7).
            None,
            std::ptr::null_mut(),
            rb_sys::RB_NOGVL_INTR_FAIL as std::ffi::c_int,
        );
    }
    match payload.outcome {
        Some(Ok(value)) => Outcome::Done(value),
        Some(Err(panic)) => Outcome::Panicked(describe(panic)),
        // RB_NOGVL_INTR_FAIL: an interrupt was already pending, so the
        // region was never entered.
        None => Outcome::Interrupted,
    }
}

/// What a panic said, as far as it can be recovered.
pub fn describe(panic: Box<dyn Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_string())
}
