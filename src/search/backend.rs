//! Backend trait for search-based compilation.
//!
//! This trait allows the search infrastructure to generate and evaluate kernels
//! without depending on specific GPU backends.

use super::types::{DynMap, GPUArch, InitData, KernelGraph, SubGraph};
use petgraph::prelude::NodeIndex;
use rustc_hash::FxHashMap;
use std::time::Duration;

/// Result of compiling a kernel graph.
pub struct CompiledKernels<F> {
    /// Compiled kernel functions keyed by kernel code hash.
    pub functions: FxHashMap<String, F>,
}

/// Result of buffer assignment.
pub struct BufferAssignment {
    /// Buffer sizes (in elements).
    pub buffer_sizes: Vec<crate::shape::Expression>,
    /// Mapping from kernel node to buffer indices.
    pub buffer_map: FxHashMap<NodeIndex, Vec<usize>>,
}

/// Trait for search backend implementations.
///
/// Backends (Metal, CUDA) implement this trait to provide:
/// - GPU architecture information
/// - Kernel compilation
/// - Kernel execution and timing
pub trait SearchBackend: Sized {
    /// The compiled kernel function type.
    type CompiledKernel;
    /// The GPU buffer type.
    type Buffer;

    /// Get the GPU architecture for this backend.
    fn arch(&self) -> GPUArch;

    /// Generate kernel code from a search subgraph.
    ///
    /// Returns the kernel graph and input mapping, or None if codegen fails.
    fn codegen(
        &self,
        graph: SubGraph,
        dyn_map: &DynMap,
    ) -> Option<(KernelGraph, FxHashMap<NodeIndex, usize>)>;

    /// Compile kernels from a kernel graph.
    fn compile_kernels(&self, kernels: &KernelGraph) -> FxHashMap<String, Self::CompiledKernel>;

    /// Assign buffers for intermediate storage.
    fn assign_buffers(&self, kernels: &KernelGraph) -> BufferAssignment;

    /// Copy data from host to device.
    fn host_to_device(&self, data: &[f32]) -> Self::Buffer;

    /// Allocate a new buffer on device.
    fn alloc_buffer(&self, size_bytes: usize) -> Self::Buffer;

    /// Run the kernel graph and return execution time.
    ///
    /// # Arguments
    /// - `inputs`: Map from input index to (buffer, is_const) pairs
    /// - `kernels`: The kernel graph to execute
    /// - `dyn_map`: Dynamic variable values
    /// - `compiled`: Compiled kernel functions
    /// - `intermediates`: Pre-allocated intermediate buffers
    /// - `buffer_map`: Buffer assignment from `assign_buffers`
    ///
    /// # Returns
    /// Tuple of (output data as Vec<f32>, execution time)
    fn run_graph(
        &self,
        inputs: &mut FxHashMap<usize, (&Self::Buffer, bool)>,
        kernels: &KernelGraph,
        dyn_map: &DynMap,
        compiled: &FxHashMap<String, Self::CompiledKernel>,
        intermediates: &mut [Self::Buffer],
        buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
    ) -> (Vec<Vec<f32>>, Duration);

    /// Evaluate the cost of a search graph by running it.
    ///
    /// Returns the execution time in microseconds, or None if evaluation fails.
    fn evaluate_cost(
        &self,
        graph: &SubGraph,
        inputs: &[(String, InitData)],
        dyn_map: &DynMap,
        warmup_trials: usize,
        trials: usize,
    ) -> Option<u128> {
        // Default implementation using the other methods
        let (kernels, _gmem_mapping) = self.codegen(graph.clone(), dyn_map)?;

        let compiled = self.compile_kernels(&kernels);
        let assignment = self.assign_buffers(&kernels);

        // Prepare inputs
        let mut input_buffers: Vec<Self::Buffer> = Vec::new();
        for (_label, init) in inputs {
            let data = match init {
                InitData::Data(d) => d.clone(),
                InitData::Expr(e) => {
                    let size = e.exec(dyn_map).unwrap_or(1);
                    vec![0.0f32; size]
                }
            };
            input_buffers.push(self.host_to_device(&data));
        }

        // Create input mapping
        let mut inputs_map: FxHashMap<usize, (&Self::Buffer, bool)> = FxHashMap::default();
        for (i, buf) in input_buffers.iter().enumerate() {
            inputs_map.insert(i, (buf, false));
        }

        // Allocate intermediates
        let mut intermediates: Vec<Self::Buffer> = assignment
            .buffer_sizes
            .iter()
            .map(|size| {
                let bytes = size.exec(dyn_map).unwrap_or(1) * std::mem::size_of::<f32>();
                self.alloc_buffer(bytes)
            })
            .collect();

        // Warmup
        for _ in 0..warmup_trials {
            let _ = self.run_graph(
                &mut inputs_map,
                &kernels,
                dyn_map,
                &compiled,
                &mut intermediates,
                &assignment.buffer_map,
            );
        }

        // Measure
        let mut total_time = Duration::ZERO;
        for _ in 0..trials {
            let (_, time) = self.run_graph(
                &mut inputs_map,
                &kernels,
                dyn_map,
                &compiled,
                &mut intermediates,
                &assignment.buffer_map,
            );
            total_time += time;
        }

        Some((total_time.as_micros() / trials as u128).max(1))
    }
}
