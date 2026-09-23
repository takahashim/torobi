//! Differentiation, which is where a Rust closure crosses into C++.
//!
//! This is the riskiest part of the binding: every other call either links
//! or does not, and this one can compile and then misbehave. The closure
//! travels to mlx-c as a payload with a destructor, is called back through
//! [`trampoline`], and reports failure with the only thing the boundary
//! carries, an `int`. What actually went wrong waits on the Rust side
//! (`error::park`) and is picked up by `error::settle`.

use std::ffi::{c_int, c_void};
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::error::{self, check, failure, install, Exception, Parked, Result};
use super::handle::Vector;
use super::{sys, Array};

const SUCCESS: c_int = 0;
const FAILURE: c_int = 1;

/// A Rust closure, owned by mlx-c as an `mlx_closure`.
///
/// `'a` is what the closure borrows. The closure (and every copy MLX takes
/// of it) is freed when this is dropped, which is why nothing that holds
/// one may outlive `'a`.
struct Closure<'a> {
    raw: sys::mlx_closure,
    _borrows: PhantomData<&'a ()>,
}

type Function<'a> = dyn FnMut(&[Array]) -> Result<Vec<Array>> + 'a;

impl<'a> Closure<'a> {
    fn new<F>(f: F) -> Result<Self>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
    {
        install();
        let payload = Box::into_raw(Box::new(f)) as *mut c_void;
        // From here the payload is mlx-c's: it wraps it with `release` as
        // the deleter before anything else can fail, and runs that deleter
        // when the last copy of the closure goes, or on its own failure.
        let raw = unsafe {
            sys::mlx_closure_new_func_payload(Some(trampoline::<F>), payload, Some(release::<F>))
        };
        if raw.ctx.is_null() {
            return Err(failure("mlx_closure_new_func_payload"));
        }
        Ok(Self {
            raw,
            _borrows: PhantomData,
        })
    }
}

impl Drop for Closure<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlx_closure_free(self.raw) };
    }
}

/// What mlx-c calls, and what calls the Rust closure behind `payload`.
///
/// Nothing may unwind out of here: this frame is called from C++, where a
/// Rust panic is undefined behaviour rather than an error. A panic is
/// caught and parked with its payload, and resumed once control is back
/// in Rust (`error::settle`).
///
/// `inputs` belongs to mlx-c, which frees it after this returns; `out` is
/// an empty vector mlx-c made, and what is written into it is copied, so
/// the vector built here is dropped afterwards like any other.
extern "C" fn trampoline<F>(
    out: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int
where
    F: FnMut(&[Array]) -> Result<Vec<Array>>,
{
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let closure = unsafe { &mut *(payload as *mut F) };
        let produced = unsafe { Vector::read(inputs) }
            .and_then(|given| closure(&given))
            .and_then(|produced| Vector::of(&produced))
            .and_then(|vector| {
                check(unsafe { sys::mlx_vector_array_set(out, vector.0) }, "mlx_vector_array_set")
            });
        match produced {
            Ok(()) => SUCCESS,
            Err(error) => {
                error::park(Parked::Error(error));
                FAILURE
            }
        }
    }));
    caught.unwrap_or_else(|payload| {
        error::park(Parked::Panic(payload));
        FAILURE
    })
}

extern "C" fn release<F>(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    // A closure whose captures panic on drop would otherwise unwind into
    // C++. Dropping the panic is the lesser harm.
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(payload as *mut F));
    }));
}

/// An `mlx_closure_value_and_grad`, freed when it goes.
struct ValueAndGrad(sys::mlx_closure_value_and_grad);

impl Drop for ValueAndGrad {
    fn drop(&mut self) {
        unsafe { sys::mlx_closure_value_and_grad_free(self.0) };
    }
}

/// The closures `value_and_grad_with_argnums` accepts: one that can fail
/// and one that cannot, as mlx-rs takes both.
pub trait IntoValueAndGrad<'a, Err> {
    #[doc(hidden)]
    fn into_function(self) -> Box<Function<'a>>;
}

impl<'a, F> IntoValueAndGrad<'a, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn into_function(mut self) -> Box<Function<'a>> {
        Box::new(move |inputs| Ok(self(inputs)))
    }
}

impl<'a, F> IntoValueAndGrad<'a, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>> + 'a,
{
    fn into_function(self) -> Box<Function<'a>> {
        Box::new(self)
    }
}

/// `f` and its gradient with respect to the arguments `argnums` names, as
/// a function of the arguments: mlx-rs's `value_and_grad_with_argnums`.
///
/// Every handle a call makes is owned as soon as it exists, so a failure
/// at any point (MLX refusing the closure, the closure failing, a vector
/// that cannot be read back) frees them on the way out.
pub fn value_and_grad_with_argnums<'a, F, Err>(
    f: F,
    argnums: &'a [i32],
) -> impl FnMut(&[Array]) -> Result<(Vec<Array>, Vec<Array>)> + 'a
where
    F: IntoValueAndGrad<'a, Err> + 'a,
{
    let closure = Closure::new(f.into_function()).map_err(|e| e.what().to_string());
    move |arrays: &[Array]| {
        let closure = closure.as_ref().map_err(|what| Exception::custom(what.clone()))?;
        let mut valued = ValueAndGrad(unsafe { sys::mlx_closure_value_and_grad_new() });
        check(
            unsafe {
                sys::mlx_value_and_grad(&mut valued.0, closure.raw, argnums.as_ptr(), argnums.len())
            },
            "mlx_value_and_grad",
        )?;

        let given = Vector::of(arrays)?;
        let mut values = Vector::new();
        let mut grads = Vector::new();
        let status = unsafe {
            sys::mlx_closure_value_and_grad_apply(&mut values.0, &mut grads.0, valued.0, given.0)
        };
        error::settle(status, "mlx_closure_value_and_grad_apply")?;
        Ok((values.arrays()?, grads.arrays()?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::mlxc::handle::Stream;
    use crate::runtime;

    fn run<R>(f: impl FnOnce() -> Result<R>) -> R {
        runtime::runtime()
            .execute(|| Ok(f()?))
            .expect("the runtime should let this through")
    }

    fn square(x: &Array) -> Result<Array> {
        let stream = Stream::default_device();
        Array::try_from_op(|res| unsafe {
            sys::mlx_multiply(res, x.as_raw(), x.as_raw(), stream.as_raw())
        })
    }

    /// Counts the drops of whatever it is captured by, so a test can see
    /// that the payload mlx-c was handed went exactly once.
    #[derive(Clone)]
    struct Dropped(Arc<AtomicUsize>);

    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A counter, and a token whose clones each count one drop.
    fn counter() -> (Arc<AtomicUsize>, Dropped) {
        let count = Arc::new(AtomicUsize::new(0));
        (count.clone(), Dropped(count))
    }

    /// d/dx of x^2 at x = 3 is 6, and the value is 9.
    ///
    /// Trivial arithmetic on purpose: what is under test is the closure
    /// crossing into C++ and the arrays coming back, not the maths.
    #[test]
    fn a_closure_crosses_into_mlx_and_its_gradient_comes_back() {
        let answer = run(|| {
            let f = |inputs: &[Array]| -> Result<Vec<Array>> { Ok(vec![square(&inputs[0])?]) };
            let (values, grads) = value_and_grad_with_argnums(f, &[0])(&[Array::from_f32(3.0)])?;
            Ok((values[0].item_cast::<f32>(), grads[0].item_cast::<f32>()))
        });
        assert_eq!(answer, (9.0, 6.0));
    }

    /// The infallible form is the same function.
    #[test]
    fn a_closure_that_cannot_fail_is_taken_as_well() {
        let answer = run(|| {
            let f = |inputs: &[Array]| -> Vec<Array> { vec![square(&inputs[0]).unwrap()] };
            let (_, grads) = value_and_grad_with_argnums(f, &[0])(&[Array::from_f32(2.0)])?;
            Ok(grads[0].item_cast::<f32>())
        });
        assert_eq!(answer, 4.0);
    }

    /// A closure that fails reports its own error rather than MLX's
    /// "mlx_closure returned a non-zero value", which is the whole reason
    /// the parking slot exists.
    #[test]
    fn a_closure_that_fails_says_what_it_was() {
        let said = run(|| {
            let refuses = |_: &[Array]| -> Result<Vec<Array>> {
                Err(Exception::custom("the objective gave up"))
            };
            Ok(value_and_grad_with_argnums(refuses, &[0])(&[Array::from_f32(1.0)])
                .unwrap_err()
                .to_string())
        });
        assert!(said.contains("the objective gave up"), "{said}");
        assert!(!said.contains("non-zero"), "{said}");
    }

    /// A panic does not cross C++, and comes out the other side as the
    /// same panic, which is what lets the runtime's gate be poisoned by it.
    #[test]
    fn a_closure_that_panics_resumes_its_panic_on_the_rust_side() {
        let caught = std::panic::catch_unwind(|| {
            let panics = |_: &[Array]| -> Result<Vec<Array>> { panic!("inside the closure") };
            let _ = value_and_grad_with_argnums(panics, &[0])(&[Array::from_f32(1.0)]);
        });
        let payload = caught.expect_err("the panic should come back");
        let said = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert_eq!(said, "inside the closure");

        // And nothing is left parked to be blamed on the next call.
        let next = run(|| {
            let f = |inputs: &[Array]| -> Result<Vec<Array>> { Ok(vec![square(&inputs[0])?]) };
            let (values, _) = value_and_grad_with_argnums(f, &[0])(&[Array::from_f32(2.0)])?;
            Ok(values[0].item_cast::<f32>())
        });
        assert_eq!(next, 4.0);
    }

    /// The payload mlx-c was handed is dropped exactly once, however the
    /// call ended: a double drop would be a double free, and none would be
    /// whatever the closure captured, leaked.
    #[test]
    fn the_closure_is_released_once_whatever_happened() {
        type Outcome = fn(&[Array]) -> Result<Vec<Array>>;
        let outcomes: [(&str, Outcome); 3] = [
            ("succeeded", |inputs| Ok(vec![square(&inputs[0])?])),
            ("failed", |_| Err(Exception::custom("no"))),
            // Not a scalar, so MLX refuses to differentiate it after the
            // closure has run and produced arrays.
            ("MLX refused", |_| Ok(vec![Array::from_slice(&[1.0f32, 2.0], &[2])])),
        ];
        for (what, outcome) in outcomes {
            let (count, token) = counter();
            run(|| {
                let f = move |inputs: &[Array]| {
                    let _held = &token;
                    outcome(inputs)
                };
                let mut vg = value_and_grad_with_argnums(f, &[0]);
                let _ = vg(&[Array::from_f32(3.0)]);
                let _ = vg(&[Array::from_f32(4.0)]);
                drop(vg);
                Ok(())
            });
            assert_eq!(count.load(Ordering::SeqCst), 1, "released after it {what}");
        }

        let (count, token) = counter();
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let f = move |_: &[Array]| -> Result<Vec<Array>> {
                let _held = &token;
                panic!("mid-trace")
            };
            let _ = value_and_grad_with_argnums(f, &[0])(&[Array::from_f32(1.0)]);
        }));
        assert_eq!(count.load(Ordering::SeqCst), 1, "released after it panicked");
    }

    /// What MLX is still holding after a failed call is what it held
    /// before: nothing the call made, or was handed, is left referenced.
    ///
    /// The input is large and evaluated, so a single leaked reference to
    /// it (in the vector handed to MLX, in the closure's inputs, in the
    /// values that never came back) keeps megabytes active.
    #[test]
    fn a_failed_call_leaves_nothing_behind() {
        let active = || {
            let mut bytes = 0usize;
            check(unsafe { sys::mlx_get_active_memory(&mut bytes) }, "active memory").unwrap();
            bytes
        };
        let (before, after) = run(|| {
            let before = active();
            for _ in 0..3 {
                let input = Array::from_slice(&vec![1.0f32; 1 << 20], &[1 << 20]);
                input.eval()?;
                let not_a_scalar = |inputs: &[Array]| -> Result<Vec<Array>> {
                    Ok(vec![square(&inputs[0])?])
                };
                let refused = value_and_grad_with_argnums(not_a_scalar, &[0])(&[input]);
                assert!(refused.is_err(), "a vector has no gradient");
            }
            Ok((before, active()))
        });
        assert_eq!(after, before, "{} bytes are still held", after.saturating_sub(before));
    }
}
