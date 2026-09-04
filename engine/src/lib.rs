//! Torobi's execution engine.
//!
//! A [`Session`] owns one graph, its parameters and its bound inputs, and
//! runs training steps against MLX. It is the whole of what the Ruby side
//! can reach: no tensors cross the boundary (only copies, by name), no
//! callbacks come back in, and the model never leaves this crate.

// What escapes this crate, and nothing else.
//
// **The gate is the whole design, so the compiler should hold it.** Every
// public route to MLX goes through `runtime::execute` (`crate::runtime`),
// and that was true of `Session` while `state`, `executor`, `interp` and
// `op` were public beside it: a caller could build a `TrainState` and
// differentiate with it, off the gate, and the failure that follows is
// two threads on one command queue, which ends the process rather than
// returning. Enforced by convention is enforced until somebody is in a
// hurry.
//
// What is public is what the Ruby extension and the command-line tool
// actually use: a session, the values that cross to it, where its
// parameters come from, the allocator's numbers, and a checkpoint's
// manifest.
pub mod checkpoint;
pub(crate) mod executor;
#[cfg(test)]
mod fixtures;
pub(crate) mod graph;
pub mod init;
pub(crate) mod interp;
pub mod memory;
pub(crate) mod op;
pub(crate) mod optimizer;
pub(crate) mod plan;
pub mod runtime;
pub mod session;
pub(crate) mod state;
pub mod tensor;

pub use optimizer::Config as Optimizer;
pub use plan::Weights;
pub use runtime::{initialize, RuntimeError};
pub use session::Session;

/// What this engine was built from. The vendoring ledger (docs/vendoring.md)
/// says every artifact must be able to report its revisions; a journal
/// records this so a run can say which build produced it.
pub fn build_info() -> serde_json::Value {
    serde_json::json!({
        "torobi_engine": env!("CARGO_PKG_VERSION"),
        "mlx_rs": mlx_rs_revision(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "target": std::env::consts::ARCH,
    })
}

/// The mlx-rs revision this was built against, read from the dependency
/// pin at build time (see build.rs and docs/vendoring.md).
fn mlx_rs_revision() -> &'static str {
    option_env!("TOROBI_MLX_RS_REV").unwrap_or("unknown, see docs/vendoring.md")
}
