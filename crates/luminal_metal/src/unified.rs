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
    fn compile_with_search<O: ToIdsMut>(
        &self,
        graph: &mut Graph,
        mut remap: O,
        _search_steps: usize,
        _timeout: Option<std::time::Duration>,
    ) {
        // TODO: Implement full search-based optimization
        // For now, fall back to fast compilation
        //
        // The full implementation would:
        // 1. Use luminal::search::translate::translate_graph to convert to IR
        // 2. Use luminal::search::extract::search to find optimal graph
        // 3. Use luminal::search::codegen::codegen to generate kernels
        // 4. Use luminal::search::run::compile_kernels to compile
        // 5. Replace subgraphs with optimized operators

        // Fallback to fast compilation for now
        MetalCompiler::<T>::default().compile(graph, &mut remap);
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
