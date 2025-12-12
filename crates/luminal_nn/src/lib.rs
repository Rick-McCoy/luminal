//! This crate is deprecated. Please use `luminal::nn` instead.
//!
//! This module re-exports the `nn` module from the core `luminal` crate
//! for backwards compatibility.

#[deprecated(since = "0.3.0", note = "Use luminal::nn directly instead of luminal_nn")]
pub use luminal::nn::*;
