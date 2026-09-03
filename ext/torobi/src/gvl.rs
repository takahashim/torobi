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
/// Returns `Err` with the panic's message if `call` panicked. Resuming the
/// unwind would be worse than returning it: magnus turns a panic that
/// escapes into a Ruby `fatal`, which no `rescue` catches and which ends
/// the process. The caller turns this into a StepError and poisons the
/// session instead (docs/plan.md section 4.1).
///
/// # Safety contract (upheld by the callers in this crate, not by the type
/// system): `call` must not touch the Ruby VM in any way.
pub fn without_gvl<F, R>(call: F) -> Result<R, String>
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
        Ok(value) => Ok(value),
        Err(panic) => Err(describe(panic)),
    }
}

/// What a panic said, as far as it can be recovered.
fn describe(panic: Box<dyn Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_string())
}
