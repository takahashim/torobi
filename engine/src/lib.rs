//! Torobi's execution engine.
//!
//! A [`Session`] owns one graph, its parameters and its bound inputs, and
//! runs training steps against MLX. It is the whole of what the Ruby side
//! can reach: no tensors cross the boundary (only copies, by name), no
//! callbacks come back in, and the model never leaves this crate.

pub mod graph;
pub mod interp;
pub mod optimizer;
pub mod session;

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

/// mlx-rs is a path dependency for now (the ledger explains why), so the
/// version is what its manifest declares.
fn mlx_rs_revision() -> &'static str {
    option_env!("TOROBI_MLX_RS_REV").unwrap_or("path dependency, see docs/vendoring.md")
}
