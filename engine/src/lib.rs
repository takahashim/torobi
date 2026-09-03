//! Torobi's execution engine.
//!
//! A [`Session`] owns one graph, its parameters and its bound inputs, and
//! runs training steps against MLX. It is the whole of what the Ruby side
//! can reach: no tensors cross the boundary (only copies, by name), no
//! callbacks come back in, and the model never leaves this crate.

pub mod checkpoint;
pub mod executor;
#[cfg(test)]
mod fixtures;
pub mod graph;
pub mod init;
pub mod interp;
pub mod memory;
pub mod op;
pub mod optimizer;
pub mod plan;
pub mod runtime;
pub mod session;
pub mod state;
pub mod tensor;

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
