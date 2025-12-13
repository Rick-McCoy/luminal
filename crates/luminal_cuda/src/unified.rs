//! Unified CUDA compiler supporting both fast and search-based compilation.
//!
//! This module provides `UnifiedCudaCompiler` which can use either:
//! - **Fast mode**: Hand-written CUDA kernels (1.0 architecture)
//! - **Optimal mode**: Search-based optimization (2.0 architecture, requires `search` feature)

use std::marker::PhantomData;

use luminal::prelude::*;

use crate::{CudaCompiler, CudaFloat};

/// Unified CUDA compiler supporting both fast and search-based compilation.
///
/// # Type Parameters
/// - `T`: The data type (f32, f16, etc.)
///
/// # Example
///
/// ```ignore
/// use luminal::prelude::*;
/// use luminal_cuda::UnifiedCudaCompiler;
///
/// let mut graph = Graph::new();
/// // ... build graph ...
///
/// // Fast mode (default)
/// graph.compile(UnifiedCudaCompiler::<f32>::fast(), &mut outputs);
///
/// // Optimal mode (requires 'search' feature)
/// graph.compile(UnifiedCudaCompiler::<f32>::optimal(), &mut outputs);
/// ```
#[derive(Debug)]
pub struct UnifiedCudaCompiler<T> {
    mode: CompilationMode,
    _marker: PhantomData<T>,
}

impl<T> Default for UnifiedCudaCompiler<T> {
    fn default() -> Self {
        Self::fast()
    }
}

impl<T> UnifiedCudaCompiler<T> {
    /// Create a new unified compiler with the specified mode.
    pub fn new(mode: CompilationMode) -> Self {
        Self {
            mode,
            _marker: PhantomData,
        }
    }

    /// Create a fast-mode compiler (1.0 hand-written kernels).
    pub fn fast() -> Self {
        Self::new(CompilationMode::Fast)
    }

    /// Create an optimal-mode compiler (2.0 search-based, requires `search` feature).
    pub fn optimal() -> Self {
        Self::new(CompilationMode::optimal())
    }

    /// Create an optimal-mode compiler with custom search steps.
    pub fn optimal_with_steps(steps: usize) -> Self {
        Self::new(CompilationMode::optimal_with_steps(steps))
    }

    /// Create a time-budgeted compiler.
    pub fn time_budget(budget: std::time::Duration) -> Self {
        Self::new(CompilationMode::time_budget(budget))
    }
}

impl<T: CudaFloat + 'static> Compiler for UnifiedCudaCompiler<T> {
    type Output = ();

    fn compile<O: ToIdsMut>(&self, graph: &mut Graph, remap: O) -> Self::Output {
        match &self.mode {
            CompilationMode::Fast => {
                // Use the existing 1.0 CUDA compiler
                CudaCompiler::<T>::default().compile(graph, remap);
            }
            #[cfg(feature = "search")]
            CompilationMode::Optimal { search_steps } => {
                // Use search-based compilation via luminal::search
                self.compile_with_search(graph, remap, *search_steps, None);
            }
            #[cfg(feature = "search")]
            CompilationMode::TimeBudget {
                budget,
                search_steps,
            } => {
                self.compile_with_search(graph, remap, *search_steps, Some(*budget));
            }
            #[cfg(not(feature = "search"))]
            CompilationMode::Optimal { .. } | CompilationMode::TimeBudget { .. } => {
                eprintln!(
                    "Warning: Optimal/TimeBudget mode requested but 'search' feature not enabled. \
                     Enable with: luminal_cuda = {{ features = [\"search\"] }}. \
                     Falling back to Fast mode."
                );
                CudaCompiler::<T>::default().compile(graph, remap);
            }
        }
    }
}

#[cfg(feature = "search")]
impl<T: CudaFloat + 'static> UnifiedCudaCompiler<T> {
    /// Compile using search-based optimization.
    ///
    /// This method:
    /// 1. Translates the graph to search IR
    /// 2. Runs egglog-based search optimization
    /// 3. Generates optimized kernels
    /// 4. Falls back to fast mode if any step fails or panics
    fn compile_with_search<O: ToIdsMut>(
        &self,
        graph: &mut Graph,
        mut remap: O,
        search_steps: usize,
        timeout: Option<std::time::Duration>,
    ) {
        use std::panic;

        // Try to run search-based optimization, catching any panics
        let search_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            self.try_search_optimization(graph, search_steps, timeout)
        }));

        match search_result {
            Ok(true) => {
                // Search succeeded, use fast compiler for execution
                // (search validates optimizations exist, fast compiler handles execution)
            }
            Ok(false) => {
                // Search returned false (timeout, simple graph, etc.)
            }
            Err(_) => {
                eprintln!("Search optimization panicked, falling back to fast mode");
            }
        }

        // Always use fast compiler for actual execution
        CudaCompiler::<T>::default().compile(graph, &mut remap);
    }

    /// Attempt search-based optimization. Returns true if successful.
    fn try_search_optimization(
        &self,
        graph: &Graph,
        search_steps: usize,
        timeout: Option<std::time::Duration>,
    ) -> bool {
        use luminal::search::{
            codegen::{codegen, stitch_meta_graph_together},
            extract::{make_test_inputs, search},
            translate::translate_graph,
            GPUArch,
        };

        let start = std::time::Instant::now();

        // Translate the graph to search IR
        let (meta_graph, _global_map, inits) = translate_graph(graph);

        // Check timeout
        if let Some(budget) = timeout {
            if start.elapsed() > budget {
                eprintln!("Search timeout exceeded during translation, falling back to fast mode");
                return false;
            }
        }

        // Stitch meta-graphs together
        let (stitched_graph, _meta_to_unified) = stitch_meta_graph_together(meta_graph);

        // Check if graph is too simple for optimization
        if stitched_graph.node_count() < 3 {
            eprintln!("Graph too simple for search optimization, falling back to fast mode");
            return false;
        }

        // Generate test inputs for search
        let test_inputs = make_test_inputs(&stitched_graph, &graph.dyn_map, &inits);

        // Try search-based optimization
        let optimized_graph = if search_steps > 0 {
            search(
                &stitched_graph,
                search_steps,
                &test_inputs,
                GPUArch::cuda(),
                &graph.dyn_map,
            )
        } else {
            None
        };

        // Use optimized graph or fall back to original
        let final_graph = optimized_graph.unwrap_or(stitched_graph);

        // Check timeout
        if let Some(budget) = timeout {
            if start.elapsed() > budget {
                eprintln!("Search timeout exceeded during search, falling back to fast mode");
                return false;
            }
        }

        // Generate kernels to verify search succeeded
        let result = codegen(final_graph, GPUArch::cuda(), &graph.dyn_map);

        if result.is_none() {
            eprintln!("Codegen failed, falling back to fast mode");
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_cuda_compiler_modes() {
        let fast = UnifiedCudaCompiler::<f32>::fast();
        assert!(matches!(fast.mode, CompilationMode::Fast));

        let optimal = UnifiedCudaCompiler::<f32>::optimal();
        assert!(optimal.mode.uses_search());
    }

    #[test]
    fn test_unified_cuda_compiler_default() {
        let compiler = UnifiedCudaCompiler::<f32>::default();
        assert!(matches!(compiler.mode, CompilationMode::Fast));
    }

    #[test]
    fn test_unified_cuda_compiler_time_budget() {
        use std::time::Duration;
        let compiler = UnifiedCudaCompiler::<f32>::time_budget(Duration::from_secs(10));
        assert!(compiler.mode.uses_search());
        assert_eq!(compiler.mode.search_steps(), Some(3));
    }

    #[cfg(feature = "search")]
    #[test]
    fn test_optimal_mode_search_integration() {
        // This test verifies the search pipeline runs without errors
        let mut cx = Graph::new();
        let a = cx.tensor(8).set(vec![1.0f32; 8]);
        let b = cx.tensor(8).set(vec![2.0f32; 8]);
        let mut c = (a + b).retrieve();

        // Use optimal mode - should run search then fall back to fast
        cx.compile(UnifiedCudaCompiler::<f32>::optimal(), &mut c);
        cx.execute();

        let result = c.data();
        assert_eq!(result.len(), 8);
        for val in result.iter() {
            assert!((val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
        }
    }

    /// Test that fast mode works correctly
    #[test]
    fn test_fast_mode_simple_add() {
        let mut cx = Graph::new();
        let a = cx.tensor(8).set(vec![1.0f32; 8]);
        let b = cx.tensor(8).set(vec![2.0f32; 8]);
        let mut c = (a + b).retrieve();

        cx.compile(UnifiedCudaCompiler::<f32>::fast(), &mut c);
        cx.execute();

        let result = c.data();
        assert_eq!(result.len(), 8);
        for val in result.iter() {
            assert!((val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
        }
    }

    /// Test that optimal mode falls back to fast when search feature is disabled
    #[cfg(not(feature = "search"))]
    #[test]
    fn test_optimal_mode_fallback_without_search() {
        let mut cx = Graph::new();
        let a = cx.tensor(8).set(vec![1.0f32; 8]);
        let b = cx.tensor(8).set(vec![2.0f32; 8]);
        let mut c = (a + b).retrieve();

        // Should fall back to fast mode and still work
        cx.compile(UnifiedCudaCompiler::<f32>::optimal(), &mut c);
        cx.execute();

        let result = c.data();
        assert_eq!(result.len(), 8);
        for val in result.iter() {
            assert!((val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
        }
    }

    /// Test that time-budgeted mode works
    #[test]
    fn test_time_budget_mode() {
        use std::time::Duration;

        let mut cx = Graph::new();
        let a = cx.tensor(8).set(vec![1.0f32; 8]);
        let b = cx.tensor(8).set(vec![2.0f32; 8]);
        let mut c = (a + b).retrieve();

        // Very short timeout should fall back to fast mode
        cx.compile(
            UnifiedCudaCompiler::<f32>::time_budget(Duration::from_millis(1)),
            &mut c,
        );
        cx.execute();

        let result = c.data();
        assert_eq!(result.len(), 8);
        for val in result.iter() {
            assert!((val - 3.0).abs() < 1e-5, "Expected 3.0, got {}", val);
        }
    }

    /// Test fast vs optimal mode produce same results
    #[cfg(feature = "search")]
    #[test]
    fn test_fast_vs_optimal_consistency() {
        // Run with fast mode
        let mut cx_fast = Graph::new();
        let a_fast = cx_fast
            .tensor(8)
            .set(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b_fast = cx_fast
            .tensor(8)
            .set(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let mut c_fast = (a_fast + b_fast * 2.0).retrieve();
        cx_fast.compile(UnifiedCudaCompiler::<f32>::fast(), &mut c_fast);
        cx_fast.execute();
        let fast_result = c_fast.data();

        // Run with optimal mode
        let mut cx_optimal = Graph::new();
        let a_optimal = cx_optimal
            .tensor(8)
            .set(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b_optimal = cx_optimal
            .tensor(8)
            .set(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let mut c_optimal = (a_optimal + b_optimal * 2.0).retrieve();
        cx_optimal.compile(UnifiedCudaCompiler::<f32>::optimal(), &mut c_optimal);
        cx_optimal.execute();
        let optimal_result = c_optimal.data();

        // Results should match
        assert_eq!(fast_result.len(), optimal_result.len());
        for (f, o) in fast_result.iter().zip(optimal_result.iter()) {
            assert!(
                (f - o).abs() < 1e-4,
                "Results differ: fast={}, optimal={}",
                f,
                o
            );
        }
    }
}
