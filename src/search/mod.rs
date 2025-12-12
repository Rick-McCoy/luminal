//! Search-based compilation infrastructure.
//!
//! This module provides the search-based optimization system for Luminal,
//! which uses egglog (equality saturation) to explore equivalent computation
//! graphs and find optimal kernel implementations.
//!
//! # Feature Flags
//!
//! The search infrastructure is gated behind the `search` feature flag.
//! Additionally, you need either `metal` or `cuda` feature for actual execution:
//!
//! ```toml
//! [dependencies]
//! luminal = { version = "0.2", features = ["search", "metal"] }
//! ```
//!
//! # Architecture
//!
//! - **types.rs**: Core types (`Kernel`, `GraphTerm`, `GPUArch`, etc.)
//! - **backend.rs**: `SearchBackend` trait for codegen/runtime
//! - **translate.rs**: Convert Luminal graphs to search IR
//! - **codegen.rs**: Generate GPU kernels from search IR
//! - **extract.rs**: egglog-based optimization and extraction
//! - **run.rs**: Execute kernels on GPU
//! - **operators.rs**: Custom kernel operators
//!
//! # Usage
//!
//! Users interact with search through the `UnifiedCompiler` provided by each backend:
//!
//! ```ignore
//! use luminal::prelude::*;
//! use luminal_metal::UnifiedMetalCompiler;
//!
//! let mut cx = Graph::new();
//! // ... build graph ...
//!
//! // Fast mode (default) - uses hand-written kernels
//! cx.compile(UnifiedMetalCompiler::<f32>::fast(), &mut outputs);
//!
//! // Optimal mode - uses search-based optimization
//! cx.compile(UnifiedMetalCompiler::<f32>::optimal(), &mut outputs);
//! ```

mod backend;
mod types;

#[cfg(feature = "search")]
pub mod codegen;
#[cfg(feature = "search")]
pub mod debug;
#[cfg(feature = "search")]
pub mod egraph_debugger;
#[cfg(feature = "search")]
pub mod extract;
#[cfg(feature = "search")]
pub mod operators;
#[cfg(feature = "search")]
pub mod run;
#[cfg(feature = "search")]
pub mod translate;
#[cfg(feature = "search")]
pub mod utils;

pub use backend::*;
pub use types::*;

#[cfg(feature = "search")]
pub use operators::*;
