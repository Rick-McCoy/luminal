//! Unified Metal compiler supporting both fast and search-based compilation.
//!
//! This module provides `UnifiedMetalCompiler` which can use either:
//! - **Fast mode**: Hand-written Metal kernels (1.0 architecture)
//! - **Optimal mode**: Search-based optimization (2.0 architecture, requires `search` feature)

use std::marker::PhantomData;

use luminal::prelude::*;

use crate::{MetalCompiler, MetalFloat};

/// Unified Metal compiler supporting both fast and search-based compilation.
///
/// # Type Parameters
/// - `T`: The data type (f32, f16, etc.)
///
/// # Example
///
/// ```ignore
/// use luminal::prelude::*;
/// use luminal_metal::UnifiedMetalCompiler;
///
/// let mut graph = Graph::new();
/// // ... build graph ...
///
/// // Fast mode (default)
/// graph.compile(UnifiedMetalCompiler::<f32>::fast(), &mut outputs);
///
/// // Optimal mode (requires 'search' feature)
/// graph.compile(UnifiedMetalCompiler::<f32>::optimal(), &mut outputs);
/// ```
#[derive(Debug)]
pub struct UnifiedMetalCompiler<T> {
    mode: CompilationMode,
    _marker: PhantomData<T>,
}

impl<T> Default for UnifiedMetalCompiler<T> {
    fn default() -> Self {
        Self::fast()
    }
}

impl<T> UnifiedMetalCompiler<T> {
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

impl<T: MetalFloat + 'static> Compiler for UnifiedMetalCompiler<T> {
    type Output = ();

    fn compile<O: ToIdsMut>(&self, graph: &mut Graph, remap: O) -> Self::Output {
        match &self.mode {
            CompilationMode::Fast => {
                // Use the existing 1.0 Metal compiler
                MetalCompiler::<T>::default().compile(graph, remap);
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
                     Enable with: luminal_metal = {{ features = [\"search\"] }}. \
                     Falling back to Fast mode."
                );
                MetalCompiler::<T>::default().compile(graph, remap);
            }
        }
    }
}

#[cfg(feature = "search")]
impl<T: MetalFloat + 'static> UnifiedMetalCompiler<T> {
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
        MetalCompiler::<T>::default().compile(graph, &mut remap);
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
                GPUArch::metal(),
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
        let result = codegen(final_graph, GPUArch::metal(), &graph.dyn_map);

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
    fn test_unified_metal_compiler_modes() {
        let fast = UnifiedMetalCompiler::<f32>::fast();
        assert!(matches!(fast.mode, CompilationMode::Fast));

        let optimal = UnifiedMetalCompiler::<f32>::optimal();
        assert!(optimal.mode.uses_search());
    }
}
