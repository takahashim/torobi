//! What MLX says when it refuses, and how that becomes a Rust error.
//!
//! Two routes, and they are kept apart. MLX's own failures arrive through
//! the error handler mlx-c calls before it returns a non-zero status; a
//! closure's failures (an error it returned, or a panic) are parked on the
//! Rust side of the trampoline, because the boundary can only say "1".
//!
//! **The handler is ours, registered once.** mlx-c's default prints and
//! calls `exit`, which is not a thing a caller can rescue. Nothing here
//! relies on whichever handler mlx-rs installs: until mlx-rs leaves the
//! build both may be linked, and a process that never touches mlx-rs must
//! still get errors rather than an exit (the child-process test below).

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe, Location};
use std::sync::Once;

use super::sys;

/// A failure MLX reported, or one the engine raised in the same terms.
///
/// Shaped like mlx-rs's `Exception`, down to how it prints, so the engine's
/// messages do not change with the binding underneath them.
#[derive(Debug, PartialEq)]
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
    /// The last thing MLX said on this thread, waiting for the status that
    /// goes with it. MLX reports on the thread that failed, so one slot per
    /// thread is one slot per failure.
    static SAID: RefCell<Option<String>> = const { RefCell::new(None) };

    /// What MLX said when a constructor could not make an array. Kept apart
    /// from `SAID`, because the failure that surfaces is a later op meeting
    /// the empty handle, and that op's own report would replace it.
    static UNMADE: RefCell<Option<String>> = const { RefCell::new(None) };

    /// What a closure failed with, on its way back through C.
    static PARKED: RefCell<Option<Parked>> = const { RefCell::new(None) };
}

/// Why a closure reported failure to mlx-c.
pub(crate) enum Parked {
    Error(Exception),
    Panic(Box<dyn Any + Send>),
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
            if let Ok(mut slot) = slot.try_borrow_mut() {
                *slot = Some(text);
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

/// Puts our handler in place of mlx-c's default, once per process.
///
/// Called before anything that can fail. `Once` rather than per call:
/// mlx-c keeps the handler in a plain global, so writing it from two
/// threads at once would be a race, and once is all it takes.
pub(crate) fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        sys::mlx_set_error_handler(Some(on_error), std::ptr::null_mut(), None);
    });
}

/// A failing status as an error, in MLX's words when it left any.
///
/// When what failed was an op meeting an empty array, and a constructor
/// failed to make one earlier, the constructor's report is the cause and
/// leads the message.
#[track_caller]
pub(crate) fn failure(operation: &str) -> Exception {
    let location = Location::caller();
    let said = SAID.with(|slot| slot.borrow_mut().take());
    let what = match said {
        Some(said) if said.contains("non-empty mlx_array") => {
            match UNMADE.with(|slot| slot.borrow_mut().take()) {
                Some(cause) => format!("{cause} (an array could not be made; then: {said})"),
                None => said,
            }
        }
        Some(said) => said,
        None => format!("{operation} failed"),
    };
    Exception { what, location }
}

/// Why an array a caller holds is empty: what MLX said when it could not
/// make it, if that was on this thread and not already reported.
#[track_caller]
pub(crate) fn never_made() -> Exception {
    let cause = UNMADE.with(|slot| slot.borrow_mut().take());
    Exception::custom(match cause {
        Some(cause) => format!("{cause} (an array could not be made)"),
        None => "this array was never made: mlx-c returned an empty handle".to_string(),
    })
}

/// A constructor handed back an empty array: keep what MLX said about it.
pub(crate) fn unmade() {
    let said = SAID.with(|slot| slot.borrow_mut().take());
    UNMADE.with(|slot| *slot.borrow_mut() = said);
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

pub(crate) fn park(parked: Parked) {
    PARKED.with(|slot| *slot.borrow_mut() = Some(parked));
}

/// The status of a call that ran a closure, with the closure's own failure
/// taking precedence over what MLX made of it.
///
/// When a closure fails, mlx-c turns the `1` it returned into a C++
/// exception and reports *that* ("mlx_closure returned a non-zero value"),
/// which says nothing. The parked error is the cause, so it wins, and MLX's
/// message is dropped rather than left to be blamed on a later call.
///
/// A panic resumes here, on the Rust side, with its own payload: the
/// engine's runtime relies on a panic unwinding through its gate to close
/// the process to further MLX work (`crate::runtime`), and turning it into
/// an error on the way through C would quietly undo that.
#[track_caller]
pub(crate) fn settle(status: c_int, operation: &str) -> Result<()> {
    match PARKED.with(|slot| slot.borrow_mut().take()) {
        Some(Parked::Panic(payload)) => {
            forget_said();
            resume_unwind(payload)
        }
        Some(Parked::Error(error)) if status != 0 => {
            forget_said();
            Err(error)
        }
        _ => check(status, operation),
    }
}

fn forget_said() {
    SAID.with(|slot| slot.borrow_mut().take());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlxc::handle::Stream;
    use crate::mlxc::Array;
    use crate::runtime;

    /// The marker that tells a copy of this test binary it is the child.
    const CHILD: &str = "TOROBI_MLXC_CHILD";

    /// An MLX failure is an error, in a process where mlx-rs never ran.
    ///
    /// mlx-rs registers its own handler the first time it is used, and
    /// every other test in this binary may use it first. So the claim is
    /// tested where that cannot happen: a copy of this binary that runs
    /// one test and nothing else. If the shim leaned on mlx-rs's handler,
    /// the child would meet mlx-c's default, which prints and exits.
    #[test]
    fn an_mlx_failure_is_an_error_where_mlx_rs_never_ran() {
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
    #[ignore = "run by an_mlx_failure_is_an_error_where_mlx_rs_never_ran"]
    fn child_an_mlx_failure_comes_back() {
        if std::env::var_os(CHILD).is_none() {
            return;
        }
        let said = runtime::runtime()
            .execute(|| {
                let two = Array::from_slice(&[1.0f32, 2.0], &[2]);
                let three = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
                let stream = Stream::default_device();
                let added = Array::try_from_op(|res| unsafe {
                    sys::mlx_add(res, two.as_raw(), three.as_raw(), stream.as_raw())
                });
                Ok(added.err().map(|error| error.what().to_string()))
            })
            .expect("the runtime should let this through");
        let said = said.expect("shapes 2 and 3 do not broadcast");
        assert!(said.contains("broadcast"), "{said}");
    }

    /// What a constructor leaves behind when it cannot make an array: an
    /// empty handle, which the next op refuses. That is an error rather
    /// than an exit, and on a Mac it is the only way to reach it, since
    /// making an array there cannot fail.
    #[test]
    fn an_op_on_an_array_that_was_never_made_is_an_error() {
        let said = runtime::runtime()
            .execute(|| {
                let empty = Array::empty();
                let stream = Stream::default_device();
                let added = Array::try_from_op(|res| unsafe {
                    sys::mlx_add(res, empty.as_raw(), empty.as_raw(), stream.as_raw())
                });
                Ok(added.err().map(|error| error.what().to_string()))
            })
            .expect("the runtime should let this through");
        let said = said.expect("an empty handle is not an array");
        assert!(said.contains("non-empty"), "{said}");
    }

    /// Where it can fail, on CUDA with no usable device, MLX's report about
    /// the constructor is the cause, and leads what the later op says.
    #[test]
    fn a_constructor_s_failure_leads_the_error_it_causes() {
        on_error(c"cudaMallocManaged(&data, size) failed".as_ptr(), std::ptr::null_mut());
        unmade();
        on_error(c"expected a non-empty mlx_array".as_ptr(), std::ptr::null_mut());
        let said = failure("mlx_add").what().to_string();
        assert!(said.starts_with("cudaMallocManaged"), "{said}");
        assert!(said.contains("non-empty"), "{said}");

        // And it is used once, not blamed on a later, unrelated failure.
        on_error(c"expected a non-empty mlx_array".as_ptr(), std::ptr::null_mut());
        assert!(!failure("mlx_add").what().contains("cudaMallocManaged"));
    }

    #[test]
    fn an_exception_prints_the_way_mlx_rs_printed_one() {
        let said = Exception::custom("no binding for input \"x\"").to_string();
        assert!(said.starts_with(r#""no binding for input \"x\"" at "#), "{said}");
    }
}
