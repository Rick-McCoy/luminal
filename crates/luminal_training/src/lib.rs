//! This crate is deprecated. Please use `luminal::training` instead.
//!
//! This module re-exports the `training` module from the core `luminal` crate
//! for backwards compatibility.

#[deprecated(
    since = "0.3.0",
    note = "Use luminal::training directly instead of luminal_training"
)]
pub use luminal::training::*;
