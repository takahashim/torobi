//! Releasing the GVL around the engine's long calls.
//!
//! This is the one piece of unsafe machinery in the binding, and it stays
//! small on purpose: the closure it runs must not touch the Ruby API, and
//! nothing else in this crate needs the GVL released. A panic inside is
//! caught and resumed once the GVL is back, so it never unwinds through C.

use std::any::Any;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

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
/// # Safety contract (upheld by the callers in this crate, not by the type
/// system): `call` must not touch the Ruby VM in any way.
pub fn without_gvl<F, R>(call: F) -> R
where
    F: FnOnce() -> R,
{
    let mut payload = Payload {
        call: Some(call),
        outcome: None,
    };
    unsafe {
        rb_sys::rb_thread_call_without_gvl(
            Some(trampoline::<F, R>),
            &mut payload as *mut Payload<F, R> as *mut c_void,
            // No unblocking function: a step is not interruptible from
            // outside, and pretending otherwise would be worse than waiting
            // for it (a step is milliseconds to a second).
            None,
            std::ptr::null_mut(),
        );
    }
    match payload.outcome.expect("the trampoline stored an outcome") {
        Ok(value) => value,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}
