//! What MLX says when it refuses, and how that becomes a Rust error.
//!
//! mlx-c reports a failure twice: once through the error handler, with a
//! message, and once through the status the call returns (or, for a
//! constructor, the empty handle it returns). The handler runs first, so
//! its message waits here until the caller sees the failure and asks for
//! it ([`check`], [`failure`]).
//!
//! **Our handler must be in place before mlx-c is first asked anything.**
//! mlx-c's own prints and calls `exit`, which no caller can rescue. The
//! rule that makes that hold without a call at every entrance: every mlx-c
//! call either makes a handle or is handed one, and the handles that can
//! fail as they are made are born in only a few places, which install
//! first ([`install`]): `Array::try_from_op` and `Array::made` for arrays,
//! `Stream::current` and `Stream::cpu` for streams, and `Closure::new` for
//! closures. `memory` installs too, for the allocator calls that touch no
//! handle at all. The empty containers (`mlx_array_new`, the vector and
//! map `_new` calls) are left out: making one allocates and nothing else.
//! The runtime also installs when it is created, since every route in
//! production passes through it first.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe, Location};
use std::sync::Once;

use super::sys;

/// A failure MLX reported, or one the engine raised in the same terms.
///
/// Printed as `"<what>" at <location>`, the form mlx-rs's `Exception`
/// took, so the engine's messages did not change when the binding under
/// them did.
#[derive(Debug, Clone, PartialEq)]
pub struct Exception {
    what: String,
    location: &'static Location<'static>,
}

pub type Result<T> = std::result::Result<T, Exception>;

impl Exception {
    #[track_caller]
    pub fn custom(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            location: Location::caller(),
        }
    }

    pub fn what(&self) -> &str {
        &self.what
    }
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} at {}", self.what, self.location)
    }
}

impl std::error::Error for Exception {}

impl From<&str> for Exception {
    #[track_caller]
    fn from(what: &str) -> Self {
        Self::custom(what)
    }
}

thread_local! {
    /// What MLX last said on this thread, waiting for the failing status
    /// that goes with it. MLX reports on the thread that failed, so one
    /// slot per thread is one per failure in flight.
    static SAID: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Called by mlx-c, from C++, before it returns a failing status.
///
/// Nothing may unwind out of here, so the whole body is caught, and the
/// slot is borrowed with `try_` so a re-entrant report is dropped rather
/// than a panic.
extern "C" fn on_error(message: *const c_char, _data: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let text = describe(message);
        let _ = SAID.try_with(|slot| {
            if let Ok(mut said) = slot.try_borrow_mut() {
                *said = Some(text);
            }
        });
    }));
}

fn describe(message: *const c_char) -> String {
    if message.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
}

static INSTALLED: Once = Once::new();

/// Puts our handler in place of mlx-c's default, once per process.
///
/// Once rather than per call: mlx-c keeps the handler in a plain global,
/// so writing it from two threads at once would be a race.
pub(crate) fn install() {
    INSTALLED.call_once(|| unsafe {
        sys::mlx_set_error_handler(Some(on_error), std::ptr::null_mut(), None);
    });
}

/// A failing status as an error, in MLX's words when it left any.
#[track_caller]
pub(crate) fn failure(operation: &str) -> Exception {
    let location = Location::caller();
    let what = SAID.with(|slot| slot.borrow_mut().take()).unwrap_or_else(|| format!("{operation} failed"));
    Exception { what, location }
}

/// `Ok` for mlx-c's success, MLX's own message otherwise.
#[track_caller]
pub(crate) fn check(status: c_int, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(failure(operation))
    }
}

/// Drops whatever MLX last said on this thread, for a caller that already
/// knows a better reason for the failure (a closure's own error).
pub(crate) fn forget_said() {
    SAID.with(|slot| slot.borrow_mut().take());
}

/// MLX's report, as mlx-c would deliver it, for tests that stage a failure
/// MLX cannot be made to produce on this machine.
#[cfg(test)]
pub(crate) fn report(message: &str) {
    let message = std::ffi::CString::new(message).unwrap();
    on_error(message.as_ptr(), std::ptr::null_mut());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlxc::Array;
    use crate::runtime;

    /// The marker that tells a copy of this test binary it is the child.
    const CHILD: &str = "TOROBI_MLXC_CHILD";

    /// An MLX failure is an error, not the end of the process.
    ///
    /// Tested in a copy of this binary that runs one test and nothing else:
    /// if the handler were not in place, mlx-c's default would print and
    /// exit, and that exit should fail this test rather than end the whole
    /// run with nothing reported.
    #[test]
    fn an_mlx_failure_is_an_error_not_an_exit() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "mlxc::error::tests::child_an_mlx_failure_comes_back",
                "--ignored",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .expect("the test binary should run again");
        let said = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "the child did not finish: {:?}\n{said}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(said.contains("1 passed"), "the child ran nothing: {said}");
    }

    /// The child's half. Ignored so it runs only when asked for by name.
    #[test]
    #[ignore = "run by an_mlx_failure_is_an_error_not_an_exit"]
    fn child_an_mlx_failure_comes_back() {
        if std::env::var_os(CHILD).is_none() {
            return;
        }
        let said = runtime::runtime()
            .execute(|| {
                let two = Array::from_slice(&[1.0f32, 2.0], &[2])?;
                let three = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
                Ok(two.add(&three).err().map(|error| error.what().to_string()))
            })
            .expect("the runtime should let this through");
        let said = said.expect("shapes 2 and 3 do not broadcast");
        assert!(said.contains("broadcast"), "{said}");
    }

    #[test]
    fn an_exception_prints_the_way_mlx_rs_printed_one() {
        let said = Exception::custom("no binding for input \"x\"").to_string();
        assert!(said.starts_with(r#""no binding for input \"x\"" at "#), "{said}");
    }
}
