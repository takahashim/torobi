//! Torobi's execution engine.
//!
//! A [`Session`] owns one graph, its parameters and its bound inputs, and
//! runs training steps against MLX. It is the whole of what the Ruby side
//! can reach: no tensors cross the boundary (only copies, by name), no
//! callbacks come back in, and the model never leaves this crate.

pub mod graph;
pub mod interp;
pub mod session;

pub use session::Session;
