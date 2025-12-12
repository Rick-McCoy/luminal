//! Unified compilation API for Luminal
//!
//! This module provides types for unifying the 1.0 (hand-written kernels) and
//! 2.0 (search-based optimization) compilation systems.

use std::time::Duration;

/// Compilation mode selection for the unified compiler.
///
/// Luminal supports two compilation approaches:
/// - **Fast (1.0)**: Uses hand-written, pre-optimized kernels. Compilation is nearly
///   instant but may not achieve optimal performance for all workloads.
/// - **Optimal (2.0)**: Uses search-based optimization with egglog to find the best
///   kernel implementations. Takes longer to compile but can discover better fusions
///   and optimizations.
///
/// # Examples
///
/// ```ignore
/// use luminal::CompilationMode;
/// use std::time::Duration;
///
/// // Use fast compilation (default)
/// let mode = CompilationMode::Fast;
///
/// // Use search-based optimization (slower compile, faster runtime)
/// let mode = CompilationMode::Optimal { search_steps: 3 };
///
/// // Try optimization with a time budget, fallback to fast
/// let mode = CompilationMode::TimeBudget {
///     budget: Duration::from_secs(30),
///     search_steps: 3,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub enum CompilationMode {
    /// Use hand-written kernels from luminal_metal/luminal_cuda (fast compile).
    ///
    /// This is the default mode and suitable for development and iteration.
    #[default]
    Fast,

    /// Use search-based optimization (slower compile, optimal runtime).
    ///
    /// The `search_steps` parameter controls how many egglog saturation steps to run.
    /// More steps = more optimization opportunities explored, but longer compile time.
    /// Typical values: 2-5.
    Optimal {
        /// Number of egglog saturation steps for search (typical: 2-5)
        search_steps: usize,
    },

    /// Try search-based optimization with a time budget.
    ///
    /// If the search doesn't complete within the budget, falls back to Fast mode.
    /// This is useful for production where you want optimization when possible
    /// but need reliable compile times.
    TimeBudget {
        /// Maximum time to spend on search-based compilation
        budget: Duration,
        /// Number of egglog saturation steps for search
        search_steps: usize,
    },
}

impl CompilationMode {
    /// Create a new Fast compilation mode.
    pub fn fast() -> Self {
        CompilationMode::Fast
    }

    /// Create a new Optimal compilation mode with default search steps.
    pub fn optimal() -> Self {
        CompilationMode::Optimal { search_steps: 3 }
    }

    /// Create a new Optimal compilation mode with custom search steps.
    pub fn optimal_with_steps(search_steps: usize) -> Self {
        CompilationMode::Optimal { search_steps }
    }

    /// Create a time-budgeted compilation mode.
    pub fn time_budget(budget: Duration) -> Self {
        CompilationMode::TimeBudget {
            budget,
            search_steps: 3,
        }
    }

    /// Create a time-budgeted compilation mode with custom search steps.
    pub fn time_budget_with_steps(budget: Duration, search_steps: usize) -> Self {
        CompilationMode::TimeBudget {
            budget,
            search_steps,
        }
    }

    /// Returns true if this mode uses search-based optimization.
    pub fn uses_search(&self) -> bool {
        !matches!(self, CompilationMode::Fast)
    }

    /// Get the number of search steps for this mode, or None for Fast mode.
    pub fn search_steps(&self) -> Option<usize> {
        match self {
            CompilationMode::Fast => None,
            CompilationMode::Optimal { search_steps } => Some(*search_steps),
            CompilationMode::TimeBudget { search_steps, .. } => Some(*search_steps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_mode_default() {
        let mode = CompilationMode::default();
        assert!(matches!(mode, CompilationMode::Fast));
        assert!(!mode.uses_search());
        assert!(mode.search_steps().is_none());
    }

    #[test]
    fn test_compilation_mode_optimal() {
        let mode = CompilationMode::optimal();
        assert!(mode.uses_search());
        assert_eq!(mode.search_steps(), Some(3));

        let mode = CompilationMode::optimal_with_steps(5);
        assert_eq!(mode.search_steps(), Some(5));
    }

    #[test]
    fn test_compilation_mode_time_budget() {
        let mode = CompilationMode::time_budget(Duration::from_secs(10));
        assert!(mode.uses_search());
        assert_eq!(mode.search_steps(), Some(3));

        let mode = CompilationMode::time_budget_with_steps(Duration::from_secs(10), 5);
        assert_eq!(mode.search_steps(), Some(5));
    }
}
