//! Core types for the search-based compilation system.
//!
//! These types are backend-agnostic and used throughout the search infrastructure.

use crate::shape::Expression;
use petgraph::prelude::{NodeIndex, StableGraph};
use petgraph::Directed;
use std::collections::HashMap;

/// GPU architecture identifier for codegen.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GPUArch {
    /// NVIDIA CUDA
    CUDA,
    /// Apple Metal with buffer type hints
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    /// Create a new Metal architecture.
    pub fn metal() -> Self {
        GPUArch::Metal(HashMap::new())
    }

    /// Create a new CUDA architecture.
    pub fn cuda() -> Self {
        GPUArch::CUDA
    }

    /// Get Metal buffer type for a variable.
    pub fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    /// Add a Metal buffer type hint.
    pub fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }

    /// Check if this is CUDA.
    pub fn is_cuda(&self) -> bool {
        matches!(self, GPUArch::CUDA)
    }

    /// Check if this is Metal.
    pub fn is_metal(&self) -> bool {
        matches!(self, GPUArch::Metal(_))
    }
}

/// A compiled GPU kernel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    /// The kernel source code.
    pub code: String,
    /// Grid dimensions (x, y, z).
    pub grid: (Expression, Expression, Expression),
    /// Threadblock dimensions (x, y, z).
    pub threadblock: (Expression, Expression, Expression),
    /// Shared memory size in bytes.
    pub smem: Expression,
    /// Output buffer sizes.
    pub outputs: Vec<Expression>,
}

/// Reference to a global memory buffer.
#[derive(Clone, Debug)]
pub enum GMEMBuffer {
    /// Output from a previous kernel.
    PrevKernel { kernel: usize, output: usize },
    /// Input from the computation graph.
    Input { node: NodeIndex },
}

/// IR nodes in the search graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphTerm {
    /// Global memory reference.
    GMEM { label: String },
    /// Loop entry (outer loop).
    LoopIn {
        range: Expression,
        stride: Expression,
    },
    /// Loop exit (inner loop).
    LoopOut {
        range: Expression,
        stride: Expression,
    },
    /// Element-wise addition.
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise maximum.
    Max,
    /// Base-2 exponential.
    Exp2,
    /// Base-2 logarithm.
    Log2,
    /// Reciprocal.
    Recip,
    /// Sine function.
    Sin,
    /// Negation.
    Neg,
    /// Square root.
    Sqrt,
    /// Less-than comparison.
    LessThan,
    /// Modulo operation.
    Mod,
    /// Shared memory marker.
    SMEM,
    /// Load from global to shared memory.
    SMEMLoad,
    /// Read from shared memory.
    SMEMRead,
    /// Custom kernel.
    Custom(Kernel),
    /// Debug output.
    Diff(String),
    /// Graph break point.
    Break,
    /// Tensor core matrix multiply.
    TCMatmul {
        a_k_stride: Expression,
        b_k_stride: Expression,
        a_inner_stride: Expression,
        b_inner_stride: Expression,
        c_inner_stride: Expression,
        k_outer_loops: Expression,
    },
}

/// A subgraph in the search IR.
pub type SubGraph = StableGraph<GraphTerm, (), Directed>;

/// Cross-subgraph tensor indices (source node, target node).
pub type CrossSubGraphTensorIndexes = (NodeIndex, NodeIndex);

/// A meta-graph of subgraphs.
pub type MetaGraph = StableGraph<SubGraph, CrossSubGraphTensorIndexes, Directed>;

/// Kernel graph with edge weights (buffer index, output index).
pub type KernelGraph = StableGraph<Kernel, (usize, usize)>;

/// Initialization data for inputs.
#[derive(Debug, Clone)]
pub enum InitData {
    /// Initialize from an expression (for sizes).
    Expr(Expression),
    /// Initialize from concrete data.
    Data(Vec<f32>),
}

/// Dynamic variable map (variable char -> value).
pub type DynMap = rustc_hash::FxHashMap<char, usize>;

// Backend-specific type aliases

/// Metal device type.
#[cfg(feature = "metal")]
pub type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>;
/// Metal buffer type.
#[cfg(feature = "metal")]
pub type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;
/// Metal function type.
#[cfg(feature = "metal")]
pub type Function =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>>;

/// CUDA device type.
#[cfg(feature = "cuda")]
pub type Device = std::sync::Arc<cudarc::driver::CudaContext>;
/// CUDA buffer type.
#[cfg(feature = "cuda")]
pub type Buffer = cudarc::driver::CudaSlice<f32>;

/// Stub device type when no GPU backend is enabled.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type Device = ();
/// Stub buffer type when no GPU backend is enabled.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type Buffer = Vec<f32>;
