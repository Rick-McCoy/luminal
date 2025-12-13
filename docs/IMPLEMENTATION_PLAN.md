# Luminal Implementation Plan

This document tracks the implementation status of Luminal's search-based compilation system and related infrastructure.

**Last Updated:** 2025-12-13

---

## Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Fix Metal test failures | ✅ Complete |
| 1 | Search module tests | ✅ Complete |
| 2 | CompatKernel/Diff implementations | ✅ Complete |
| 3 | CUDA warp-cooperative matmul | ✅ Complete |
| 4 | Expand search space | ✅ Complete |
| 5 | Unify 1.0 and 2.0 architectures | ✅ Complete |
| 6 | Training infrastructure | ✅ Complete |
| 7 | **Fix CNN search optimization** | 🔴 **Blocked** |
| 8 | Benchmarking suite | Not started |
| 9 | ROCm backend | Not started |
| 10 | Distributed computing | Not started |
| 11 | Python bindings | Not started |

---

## Current State

### Test Summary

| Crate | Tests | Status |
|-------|-------|--------|
| `luminal` (core + nn + training + search) | 222 | ✅ All pass |
| `luminal_metal` | 3 | ✅ All pass |
| `luminal_cuda` | ~168 | ⚠️ Some pre-existing failures |

### Architecture (v0.3.0)

```
luminal/
├── src/
│   ├── nn/           # Neural network modules (Linear, Conv2D, Transformer, etc.)
│   ├── training/     # Autograd, optimizers, loss functions, schedulers
│   └── search/       # Search-based optimization (behind `search` feature)
├── crates/
│   ├── luminal_metal/  # Metal backend + UnifiedMetalCompiler
│   └── luminal_cuda/   # CUDA backend + UnifiedCudaCompiler
```

### What Works

| Component | Status | Location |
|-----------|--------|----------|
| Graph translation | ✅ | `src/search/translate.rs` |
| Kernel codegen | ✅ | `src/search/codegen.rs` |
| GPU execution (run_graph) | ✅ | `src/search/run.rs` |
| Search optimization | ✅ | `src/search/extract.rs` |
| Custom kernel injection | ✅ | `src/search/operators.rs` |
| Metal backend | ✅ | `crates/luminal_metal/` |
| CUDA backend | ✅ | `crates/luminal_cuda/` |
| CNN training (Conv2D + gradients) | ✅ | Fixed via `PoolLastDim` primitive |
| UnifiedMetalCompiler (fast + optimal) | ⚠️ | Falls back for CNN models |
| UnifiedCudaCompiler (fast + optimal) | ⚠️ | Falls back for CNN models |

---

## Critical Issue: CNN Search Optimization Fails

### Problem Summary

When running `mnist_cnn --optimal`, the search optimization **always fails** with "cycle in kernel meta graph" and falls back to fast mode. The fallback is graceful but means no search optimization actually occurs for CNN models.

```bash
$ cargo run -p mnist_cnn --release --features metal,search -- --optimal
[codegen] FAIL: cycle in kernel meta graph
Codegen failed, falling back to fast mode
```

### Root Cause Analysis

The kernel meta-graph (which tracks dependencies between kernels) develops **cycles** when processing graphs containing `PoolLastDim` operations. This happens because:

1. **PoolLastDim is a dimension-expanding operation**: 
   - Input: `(..., N)` → Output: `(..., num_windows, kernel_size)`
   - The loop-nest IR expects operations to maintain or reduce dimensions

2. **Kernel assignment creates circular dependencies**:
   - Operations before pooling form kernel A
   - Operations after pooling form kernel B  
   - The `pool_*` GMEM node (boundary between A and B) gets assigned to both kernels
   - This creates: A → pool_GMEM → B and B → pool_GMEM → A (cycle!)

3. **Location of the failure**:
   ```
   src/search/codegen.rs:123-125
   let Ok(t) = toposort(&kernel_meta_graph, None) else {
       eprintln!("[codegen] FAIL: cycle in kernel meta graph");
       return None;
   };
   ```

### Attempted Fixes (What Didn't Work)

1. **Splitting kernels by loop signature** - Created new cycles when nodes shared dependencies
2. **Duplicate LoopIn/LoopOut detection** - Caught symptoms but not root cause
3. **Level normalization** - Caused different levels to collapse, creating duplicate nodes
4. **Merging compatible kernels** - Made cycle problem worse

### Why This Is Hard

The search IR's loop-nest model fundamentally assumes:
- Each kernel has a single coherent loop structure
- Operations flow from outer loops to inner loops
- GMEM nodes are either inputs (no incoming edges) or outputs (no outgoing internal edges)

PoolLastDim violates these assumptions because:
- It changes the loop structure mid-graph
- The pool_* GMEM acts as both output (of source kernel) and input (of consumer kernel)
- Multiple kernel assignments to the same node create cycles

---

## Known Limitations

1. **Dynamic shape pooling gradients**: `pool_last_dim` uses a primitive operator for concrete shapes (gradients work) but falls back to shape tracker for dynamic shapes (no gradient support). This affects `cumsum_last_dim` and `arange`.

2. **Test coverage gaps**:
   - `src/search/extract.rs` (search function) has no direct unit tests
   - `src/search/utils.rs` has no tests

3. **PoolLastDim search IR integration**: The search IR's loop-nest model cannot represent dimension-expanding operations. See "Critical Issue" section above.

4. **Search codegen limitations**: The search codegen doesn't support:
   - Dimension-expanding operations (PoolLastDim, PoolLastDimBackward)
   - Complex CNN training graphs (falls back to fast mode)
   - Multi-pool patterns create kernel meta-graph cycles

5. **UnifiedCompiler optimal mode behavior**:
   - For simple MLP/Transformer models: Search runs successfully
   - For CNN models with pooling: Falls back to fast mode
   - The fallback is graceful but means no optimization occurs

---

## Phase 7: Fix CNN Search Optimization

### Goal

Make `mnist_cnn --optimal` run search optimization successfully without falling back to fast mode.

### Concrete Plan

#### Option A: Treat PoolLastDim as Opaque Custom Kernels

**Complexity: Medium | Estimated: 2-3 days**

Instead of trying to represent PoolLastDim in the loop-nest IR, treat it as an opaque custom kernel that the search system doesn't try to optimize.

**Steps:**

1. **Modify `translate.rs` to emit `Custom` nodes for PoolLastDim** (instead of GMEM placeholders)
   ```rust
   // In translate_graph, for PoolLastDim:
   s if s.starts_with("PoolLastDim {") => {
       let custom = g.add_node(GraphTerm::Custom(format!(
           "pool_lastdim_{}_{}_{}",
           kernel_size, stride, dilation
       )));
       // ... connect sources and mark as kernel boundary
   }
   ```

2. **Implement `Custom` kernel execution in codegen**
   - `codegen.rs` already has a `Custom` handler (lines 133-140)
   - Add Metal/CUDA implementations for PoolLastDim custom kernels
   - Return the pre-compiled kernel code directly

3. **Ensure kernel boundaries are respected**
   - Custom nodes should force kernel splits (already handled)
   - No kernel should contain both regular loop operations AND a custom kernel

4. **Test with mnist_cnn**
   - Verify no "cycle in kernel meta graph" error
   - Verify search actually produces different kernels than fast mode

#### Option B: Fix Kernel Assignment to Prevent Cycles

**Complexity: High | Estimated: 1-2 weeks**

Fix the kernel assignment algorithm to properly handle GMEM boundary nodes.

**Steps:**

1. **Analyze the cycle creation**
   - Add debug output to `split_kernels_marked_graph` showing exactly which edges create cycles
   - Identify the pattern: which node assignments cause A → B → A dependencies

2. **Modify GMEM kernel assignment in `reassign_disjoint_kernels`**
   - Currently at lines 1975-2050 in `codegen.rs`
   - `pool_*` GMEMs inherit kernels from INCOMING neighbors (line 2038-2039)
   - This creates cycles when both incoming and outgoing neighbors are in the kernel

3. **Implement exclusive kernel assignment for boundary GMEMs**
   ```rust
   // Proposed logic:
   if label.starts_with("pool_") {
       // Boundary GMEM belongs to NEITHER kernel
       // It's a separate "passthrough" that connects kernel outputs to inputs
       kernels.clear();
       kernels.insert(next_kernel_idx);  // Give it its own kernel ID
       next_kernel_idx += 1;
   }
   ```

4. **Handle the meta-graph edge creation**
   - `split_kernels` (line 1541) builds the kernel meta-graph
   - When an edge crosses a boundary GMEM, ensure proper A → boundary → B edges

5. **Validate no cycles**
   - Add assertion: `assert!(toposort(&kernel_meta_graph, None).is_ok())`
   - If cycles detected, print which kernels are involved

#### Option C: Subgraph Isolation

**Complexity: Medium | Estimated: 1 week**

Split the graph at PoolLastDim boundaries and optimize each subgraph independently.

**Steps:**

1. **In `translate_graph`, detect PoolLastDim nodes**
   ```rust
   let pool_nodes: Vec<NodeIndex> = graph.node_indices()
       .filter(|n| is_pool_last_dim(graph, *n))
       .collect();
   ```

2. **Create separate meta-graphs for each segment**
   - Segment 1: Everything before the first PoolLastDim
   - Segment 2: Between PoolLastDim 1 and 2
   - etc.

3. **Run search optimization on each segment independently**
   - Each segment has no dimension-expanding operations
   - Each segment can be optimized in isolation

4. **Stitch results together**
   - Optimized segment 1 → PoolLastDim (fast mode) → Optimized segment 2 → ...

### Recommended Approach

**Start with Option A** (Custom Kernels) because:
- Lowest risk of breaking existing functionality
- PoolLastDim is inherently a "custom" operation not representable in loop-nest IR
- Faster to implement and test
- Other systems (TVM, Triton) also treat pooling as opaque primitives

If Option A succeeds, consider Option C for better optimization of the surrounding operations.

### Success Criteria

1. `cargo run -p mnist_cnn --release --features metal,search -- --optimal` shows:
   - No "falling back to fast mode" message
   - No "cycle in kernel meta graph" error
   - Compilation time may be longer (search is running)
   
2. All existing tests still pass:
   ```bash
   cargo test --features search
   cargo test -p luminal_metal
   ```

3. Search actually produces different (hopefully faster) kernels:
   - Compare iteration time between `--optimal` and default mode
   - `--optimal` should show measurable improvement or at least parity

---

## Completed Phases

### Phase 0: Fix Metal Test Failures ✅

Fixed Metal backend tests that were failing due to stale test data:
- `test_conv2d`: Changed from hardcoded values to CPU comparison pattern
- `test_sum` (fp16): Compare against dfdx fp32 ground truth with 0.05 tolerance
- `test_sum2` (fp16): Changed to `assert_close_precision` with 1e-2 tolerance

### Phase 1: Search Module Tests ✅

The search module has tests in:
- `src/search/translate.rs`: 17 tests for graph translation
- `src/search/codegen.rs`: 6 tests for kernel generation

### Phase 2: CompatKernel/Diff Implementations ✅

Implemented search-related operators:
- `CompatKernel::process` for Metal and CUDA (`src/search/operators.rs`)
- `Diff::process` for Metal and CUDA (debug operator)
- `custom_kernel()` function for injecting hand-written kernels

### Phase 3: CUDA Warp-Cooperative Matmul ✅

Implemented warp-cooperative 8×8 matrix multiply for CUDA:
- Located in `src/search/codegen.rs` (TCMatmul handler)
- Each warp (32 threads) computes one 8×8 tile
- Uses `__syncwarp()` for synchronization

### Phase 4: Expand Search Space ✅

Enhanced egglog rules in `src/search/code.lisp`:
- Added tile sizes: 4, 8, 16, 32
- Increased search budget to 10,000 graphs
- Added early termination for fast kernels (<50µs)

### Phase 5: Unify 1.0 and 2.0 Architectures ✅

**Completed:**
- `CompilationMode` enum in `src/unified.rs`
- `UnifiedMetalCompiler` in `crates/luminal_metal/src/unified.rs`
- `UnifiedCudaCompiler` in `crates/luminal_cuda/src/unified.rs`
- Module reorganization (nn, training merged into core)
- `compile_with_search` methods implemented using the full search pipeline

**Metal UnifiedCompiler implementation** (lines 109-221 in `unified.rs`):
```rust
fn compile_with_search(...) {
    // 1. catch_unwind to handle panics gracefully
    // 2. try_search_optimization:
    //    - translate_graph() → meta_graph
    //    - stitch_meta_graph_together() → stitched_graph  
    //    - make_test_inputs() → test data for search
    //    - search() → optimized_graph (or None)
    //    - codegen() → kernel graph (or None → fallback)
    // 3. Always use MetalCompiler for actual execution
}
```

**Graceful fallback triggers:**
- Graph too simple (< 3 nodes)
- Search timeout exceeded
- Codegen returns None (e.g., kernel meta-graph has cycles)
- Search panics (caught by catch_unwind)

**Bug fixes during Phase 5:**
- `codegen.rs:1158-1162`: Changed `neighbor_levels.pop().unwrap()` to `.pop()` to handle empty loop levels
- `codegen.rs:1172-1175`: Same fix for LoopOut handling
- `extract.rs:398`: Changed `codegen().unwrap()` to graceful `let Some(...) = ... else { return None }`

**Current limitation:**
Search validates optimizations exist but uses fast compiler for execution. This is because:
1. Search uses `objc2_metal` / `cudarc` types
2. Backend compilers use `metal_rs` / different abstractions
3. CNN models fail during codegen (cycle in kernel meta-graph)

### Phase 6: Training Infrastructure ✅

Complete training stack implemented in `src/training/`:

**Autograd** (`autograd.rs`):
- Automatic differentiation via `Autograd` compiler
- CNN gradient support via `PoolLastDim`/`PoolLastDimBackward` primitives

**Optimizers** (`optimizer.rs`):
- `sgd_on_graph`, `sgd_momentum_on_graph`
- `adam_on_graph` (with weight decay support)
- `rmsprop_on_graph`
- `lamb_on_graph` (for large batch training)
- `clip_grad_norm`, `clip_grad_value`

**Loss Functions** (`loss.rs`):
- MSE, RMSE, MAE, Huber, Smooth L1
- Cross-entropy, KL divergence, Binary cross-entropy
- Focal loss, Label smoothing

**Training Utilities**:
- `GradientAccumulator` (`accumulation.rs`)
- `CheckpointManager`, `checkpoint()` (`checkpoint.rs`)
- `GradScaler`, `AMPContext` (`mixed_precision.rs`)
- Learning rate schedulers (`scheduler.rs`)

---

## Remaining Phases

### Phase 8: Benchmarking Suite

Create systematic benchmarks comparing Luminal against PyTorch:

| Benchmark | Sizes | Metrics |
|-----------|-------|---------|
| MatMul | 512-4096 | TFLOPS |
| Attention | seq_len 512-8192 | TFLOPS |
| LLaMA Inference | 7B, 8B | tokens/sec |
| Training Step | 1B params | step/sec |

### Phase 9: ROCm Backend

New crate `luminal_rocm/` with:
- HIP/rocBLAS integration
- Same operator set as Metal/CUDA backends

### Phase 10: Distributed Computing

- NCCL/RCCL communication backends
- Data parallel training
- Tensor parallel (for large models)
- Pipeline parallel

### Phase 11: Python Bindings

PyO3-based bindings exposing:
- `Graph`, `Tensor` classes
- Operator overloading (`+`, `@`, etc.)
- NumPy interop

---

## Debugging Guide

### Environment Variables

| Variable | Effect |
|----------|--------|
| `PRINT_REJECT=1` | Print detailed rejection messages from codegen |
| `DEBUG_GRAPHVIZ=1` | Open Graphviz visualizations (warning: opens many browser tabs) |
| `PRINT_EGGLOG=1` | Print egglog search statistics |
| `RUST_BACKTRACE=1` | Show full stack traces for panics |

### Common Error Messages

**"[codegen] FAIL: cycle in kernel meta graph"**
- Location: `src/search/codegen.rs:123-125`
- Cause: Kernel dependencies form a cycle, preventing topological sort
- Usually caused by PoolLastDim operations creating shared dependencies

**"Codegen failed, falling back to fast mode"**
- Location: `crates/luminal_metal/src/unified.rs:216` (or cuda equivalent)
- Cause: `codegen()` returned `None`
- Check PRINT_REJECT=1 for specific reason

**"Search optimization panicked, falling back to fast mode"**
- Location: `crates/luminal_metal/src/unified.rs:141`
- Cause: Unhandled panic in search pipeline (caught by catch_unwind)
- Run with RUST_BACKTRACE=1 to see panic location

### Key Files for Debugging

| File | Purpose |
|------|---------|
| `src/search/codegen.rs:1100-1200` | Kernel assignment and level calculation |
| `src/search/codegen.rs:1541-1600` | Kernel meta-graph construction |
| `src/search/codegen.rs:1975-2050` | `reassign_disjoint_kernels` (GMEM handling) |
| `src/search/translate.rs:283-330` | PoolLastDim translation |
| `crates/luminal_metal/src/unified.rs:149-221` | Search pipeline orchestration |

---

## File Locations

### Core Library (`src/`)

| Component | File |
|-----------|------|
| Graph, GraphTensor | `graph.rs`, `graph_tensor.rs` |
| Primitive operators | `op.rs` |
| High-level ops | `hl_ops/` |
| Shape tracking | `shape/` |
| NN modules | `nn/` |
| Training | `training/` |
| Search (feature-gated) | `search/` |
| CompilationMode | `unified.rs` |

### Search Module (`src/search/`, requires `search` feature)

| Component | File |
|-----------|------|
| Graph translation | `translate.rs` |
| Kernel codegen | `codegen.rs` |
| Search/optimization | `extract.rs` |
| GPU execution | `run.rs` |
| Custom kernel API | `operators.rs` |
| Egglog rules | `code.lisp` |
| Types | `types.rs` |

### Backend Crates

| Component | File |
|-----------|------|
| Metal compiler | `crates/luminal_metal/src/lib.rs` |
| UnifiedMetalCompiler | `crates/luminal_metal/src/unified.rs` |
| CUDA compiler | `crates/luminal_cuda/src/lib.rs` |
| UnifiedCudaCompiler | `crates/luminal_cuda/src/unified.rs` |

---

## Quick Verification Commands

```bash
# Core library tests
cargo test --lib

# Core with search feature (Metal, macOS only)
cargo test --lib --features search,metal

# Core with search feature (CUDA)
cargo test --lib --features search,cuda

# Metal backend tests (macOS only)
cd crates/luminal_metal && cargo test

# CUDA backend tests
cd crates/luminal_cuda && cargo test -- --test-threads=1

# All workspace tests
cargo test --workspace

# Format and lint
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

---

## Search Demo

A working demo of search-based compilation is at `examples/search_demo/`:

```bash
# CUDA
cargo run -p search_demo --release --features cuda

# Metal
cargo run -p search_demo --release --features metal
```

The demo shows:
1. Full pipeline: graph → IR → codegen → GPU execution
2. Custom kernel injection via `custom_kernel()`
3. Matrix multiply through the search pipeline
4. MLP inference with search-optimized kernels
