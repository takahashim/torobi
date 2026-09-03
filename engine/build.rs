//! Records what this engine is built against, so a run can say so.
//!
//! The vendoring ledger asks that every artifact report its revisions
//! (docs/vendoring.md). The mlx-rs pin lives in Cargo.toml as a git rev;
//! read it here rather than repeating it in the source, where the two
//! would drift.

use std::fs;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    let manifest = fs::read_to_string("Cargo.toml").expect("reading Cargo.toml");
    let rev = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("mlx-rs"))
        .and_then(|line| line.split("rev = \"").nth(1))
        .and_then(|rest| rest.split('"').next())
        .unwrap_or("unpinned");
    println!("cargo:rustc-env=TOROBI_MLX_RS_REV={rev}");
}
