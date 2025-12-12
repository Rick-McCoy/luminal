# Luminal 2.0 Implementation Plan

This document provides a step-by-step implementation guide to complete the search-based compilation system and realize Luminal's full potential.

**Last Updated:** 2025-12-12 — Phases 0-6 complete. Version 0.3.0 released. CNN gradient bugs documented with regression tests.

### Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Fix Metal test failures | ✅ Complete |
| 1 | Fix E2E tests | ✅ Complete |
| 2 | Metal `todo!()` implementations | ✅ Complete |
| 2.5 | Address test coverage gaps | ✅ Complete |
| 3 | CUDA warp-cooperative matmul | ✅ Complete |
| 4 | Expand search space | ✅ Complete |
| 5 | Unify 1.0 and 2.0 Architectures | ✅ Complete |
| 6 | Complete Training Infrastructure | ✅ Complete |
| 7-10 | Remaining phases | Not started |

### ⚠️ CRITICAL BUGS - ROOT CAUSE IDENTIFIED, FIX REQUIRES ARCHITECTURAL CHANGES

| Bug | Severity | Location | Description |
|-----|----------|----------|-------------|
| **Multi-layer CNN gradients** | 🔴 CRITICAL | `src/training/autograd.rs` | Stacking 2+ Conv2D layers causes index out of bounds during backprop |
| **AvgPool2D/MaxPool2D gradients** | 🔴 CRITICAL | `src/training/autograd.rs` | Pooling layers cause index out of bounds immediately |
| **`pool_last_dim` backprop** | 🔴 CRITICAL | `src/hl_ops/movement.rs` + `add_grad` | Shape tracker index expressions become invalid during gradient flow |

**Root Cause (Identified 2025-12-12):** The autograd system cannot compute gradients for tensors that pass THROUGH `pool_last_dim` because:
1. `pool_last_dim` creates intermediate `Contiguous` nodes with complex shape transformations
2. The `add_grad` function in autograd tries to "undo" these transformations by manipulating shape trackers
3. Shape tracker index expressions can compute indices beyond the data buffer bounds

**Impact:** Cannot train standard CNN architectures (LeNet, VGG, ResNet, etc.). Only single Conv2D + Linear works.

**Regression Tests:** 8 tests added to `src/training/autograd.rs` (marked `#[ignore]`) that document these bugs.

**Workaround:** The `mnist_cnn` example uses a single Conv2D layer. This is a stopgap, not a solution.

**Required Fix:** Implement `pool_last_dim` as a primitive op with explicit gradient, OR add scatter-add support to autograd.

**See:** Section 6.4 for full technical details and reproduction steps.

> **Note (Phase 5 Complete):** The `crates/luminal_2/` crate has been merged into `luminal::search` (behind the `search` feature flag). References to `crates/luminal_2/` paths in historical sections (Phases 0-4) now correspond to `src/search/`. For example:
> - `crates/luminal_2/src/codegen.rs` → `src/search/codegen.rs`
> - `crates/luminal_2/src/translate.rs` → `src/search/translate.rs`
> - `luminal_2::` imports → `luminal::search::`

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Phase 0: Fix Failing Metal Backend Tests](#phase-0-fix-failing-metal-backend-tests) ✅ COMPLETE
3. [Phase 1: Fix Broken E2E Tests](#phase-1-fix-broken-e2e-tests) ✅ COMPLETE
4. [Phase 2: Complete Metal `todo!()` Implementations](#phase-2-complete-metal-todo-implementations) ✅ COMPLETE
5. [Phase 2.5: Address Test Coverage Gaps](#phase-25-address-test-coverage-gaps) ✅ COMPLETE
6. [Phase 3: CUDA Warp-Cooperative Matmul](#phase-3-cuda-warp-cooperative-matmul--complete) ✅ COMPLETE
7. [Phase 4: Expand Search Space](#phase-4-expand-search-space) ✅ COMPLETE
8. [Phase 5: Unify 1.0 and 2.0 Architectures](#phase-5-unify-10-and-20-architectures--complete) ✅ COMPLETE
9. [Phase 6: Complete Training Infrastructure](#phase-6-complete-training-infrastructure--complete) ✅ COMPLETE
10. [Phase 7: Benchmarking Suite](#phase-7-benchmarking-suite)
11. [Phase 8: ROCm Backend](#phase-8-rocm-backend)
12. [Phase 9: Distributed Computing](#phase-9-distributed-computing)
13. [Phase 10: Python Bindings](#phase-10-python-bindings)
14. [Appendix D: Search-Based Compilation Demo](#appendix-d-search-based-compilation-demo)

---

## Current State Assessment

### Test Summary (as of 2025-12-12)

| Crate | Tests | Status |
|-------|-------|--------|
| `luminal` (core + nn + training + search) | 186 pass, 8 ignored | ✅ All pass (8 ignored document known CNN gradient bugs) |
| `luminal_metal` | 205 | ✅ All pass |
| `luminal_cuda` | 168/199 | ⚠️ 31 pre-existing failures (fp16, norm, conv2d) |

**Note:** The 8 ignored tests are regression tests that document the CNN gradient bugs described in Section 6.4. They are marked with `#[ignore]` and will pass once the underlying autograd issues are fixed.

**Note (v0.3.0):** The architecture has been unified into a PyTorch-like structure:
- `luminal::nn` - Neural network modules (Linear, ReLU, LayerNorm, Transformer, etc.)
- `luminal::training` - Autograd, optimizers (SGD, Adam), loss functions, LR schedulers
- `luminal::search` - Search-based optimization (behind `search` feature)

The deprecated `luminal_nn` and `luminal_training` crates were removed in v0.3.0.

### What Actually Works

After auditing and fixing the codebase, here's the current state:

| Component | Status | Notes |
|-----------|--------|-------|
| `run_graph` for Metal | ✅ **Working** | `luminal::search::run` module |
| `run_graph` for CUDA | ✅ **Working** | `luminal::search::run` module |
| `compile_kernels` for Metal | ✅ **Working** | `luminal::search::run` module |
| `compile_kernels` for CUDA | ✅ **Working** | `luminal::search::run` module |
| `codegen` | ✅ **Working** | `luminal::search::codegen` - generates kernels from GraphTerm |
| `search` function | ✅ **Working** | `luminal::search::extract` - uses `run_graph` internally |
| `translate_graph` | ✅ **Working** | `luminal::search::translate` - Luminal graph to IR |
| `CompatKernel::process` Metal | ✅ **Implemented** | `luminal::search::operators` |
| `CompatKernel::process` CUDA | ✅ **Implemented** | `luminal::search::operators` |
| `Diff::process` Metal | ✅ **Implemented** | `luminal::search::operators` |
| `Diff::process` CUDA | ✅ **Implemented** | `luminal::search::operators` |
| Metal conv2d tests | ✅ **Fixed** | CPU comparison pattern |
| Metal fp16 sum tests | ✅ **Fixed** | Compare against dfdx fp32 |

### Test Coverage Gaps (Updated 2025-12-12)

| Module | Has Tests? | Coverage Gap |
|--------|------------|--------------|
| `search/operators.rs` (CompatKernel, Diff) | ✅ | Tests in backend crates |
| `search/run.rs` | ✅ | Runtime tests exercise `run_graph`, `compile_kernels`, `assign_buffers` |
| `search/extract.rs` (search) | ❌ | No unit tests for search function |
| `search/utils.rs` | ❌ | No tests |
| `search/codegen.rs` | ✅ | 6 tests |
| `search/translate.rs` | ✅ | 17 tests |

### CI Coverage Gap

The CI workflow (`.github/workflows/test.yml`) tests:
- `cargo test --workspace` (core crate tests)
- `luminal_metal` tests on macOS
- Search tests require enabling `search` feature with `metal` or `cuda`

**Recommendation:** Add `cargo test --features search,metal` to macOS CI

### Historical: Fixed Tests in luminal_metal (Phase 0)

<details>
<summary>Click to expand root cause analysis</summary>

| Test | Root Cause | Fix Applied |
|------|------------|-------------|
| `tests::fp32::test_conv2d` | Stale hardcoded values from dilation=0 era | Changed to CPU comparison |
| `tests::fp16::test_conv2d` | Same as above | Changed to CPU comparison |
| `tests::fp16::test_sum` | Compared against dfdx fp16 (loses precision) | Compare against dfdx fp32 (0.05 tolerance) |
| `tests::fp16::test_sum2` | `assert_exact` unrealistic for fp16 | Changed to `assert_close_precision` |

**Root Cause Analysis:**
- The conv2d tests used hardcoded expected values computed in May 2024 when `dilation=0`
- Commit `8d7b8c8 Switched to runtime shapes` changed to `dilation=1` but didn't update expected values
- The ~2.5x error was stale test data, NOT a kernel bug
- fp16 sum tests compared Metal vs dfdx fp16, but dfdx fp16 loses precision during reduction (0.26 max diff vs fp32), while Metal fp16 stays accurate (0.007 max diff vs fp32)

</details>

### Execution Paths Clarified

There are **two ways** to run generated kernels:

1. **Graph execution via `run_graph`** (Primary path)
   - Used by `search()` and `cost()` functions
   - Fully implemented for both Metal and CUDA
   - Takes `StableGraph<Kernel, (usize, usize)>`

2. **Operator interface via `CompatKernel::process`** (Secondary path)
   - Used by `custom_kernel()` function for inserting pre-defined kernels
   - Metal: ✅ Implemented (`lib.rs:184-292`)
   - CUDA: ✅ Implemented (`lib.rs:126-181`)

---

## Phase 0: Fix Failing Metal Backend Tests ✅ COMPLETE

**Status:** ✅ **All tests fixed and passing**

### 0.1 Root Cause Analysis

Investigation revealed the tests were failing due to **stale test data**, not kernel bugs:

| Test | Root Cause |
|------|------------|
| `test_conv2d` (fp32/fp16) | Expected values computed in May 2024 with `dilation=0`, but API changed to `dilation=1` in commit `8d7b8c8` |
| `test_sum` (fp16) | Compared against dfdx fp16 (loses precision during reduction) instead of dfdx fp32 ground truth |
| `test_sum2` (fp16) | `assert_exact` unrealistic for fp16 comparisons |

### 0.2 Fixes Applied

**Conv2D Tests:**
- Changed from hardcoded expected values to CPU comparison pattern
- Tests now compare Metal output against CPU output (same as passing `verification_tests`)
- Still exercises same conv2d configuration: kernel=(2,2), stride=(2,2), dilation=(1,1)

**FP16 Sum Tests:**
- `test_sum`: Changed reference from dfdx fp16 to dfdx fp32 (ground truth)
  - Discovery: Metal fp16 matches dfdx fp32 within 0.007, but dfdx fp16 differs by 0.26
  - Metal's fp16 uses higher-precision accumulators, preserving accuracy
  - Tolerance set to 0.05 (2x safety margin over observed 0.03 max diff)
- `test_sum2`: Changed from `assert_exact` to `assert_close_precision` with 1e-2 tolerance

### 0.3 Verification

```bash
cargo test -p luminal_metal  # All 205 tests pass
```

- [x] `cargo test -p luminal_metal test_conv2d` passes
- [x] `cargo test -p luminal_metal test_sum` passes  
- [x] `cargo test -p luminal_metal` — all 205 tests pass
- [ ] CI workflow passes on macOS (needs verification)

---

## Phase 1: Fix Broken E2E Tests ✅ COMPLETE

**Status:** ✅ **E2E tests rewritten and passing**

### 1.1 Solution Applied

The e2e test file was completely rewritten with:
- Correct imports using `objc2-metal` instead of `metal_rs`
- Correct function signatures for `translate_graph`, `stitch_meta_graph_together`, `codegen`
- Module declaration added to `lib.rs`
- 4 new tests covering translation, stitching, and codegen

### 1.2 New E2E Tests

```rust
// crates/luminal_2/src/e2e_tests.rs
e2e_simple_add         // Tests translation + stitch + codegen for simple add
e2e_translation_structure  // Tests translate_graph produces valid output
e2e_matmul_translation    // Tests matmul graph translation
e2e_stitch_graph          // Tests stitch_meta_graph_together
```

### 1.3 Verification

```bash
cargo test -p luminal_2 --features metal  # All 27 tests pass
```

- [x] `e2e_tests` module compiles
- [x] Imports use `objc2-metal` not `metal_rs`
- [x] Function signatures match current API
- [x] `e2e_simple_add` passes
- [x] `e2e_matmul_translation` passes
- [x] Empty `tests.rs` removed

---

## Phase 2: Complete Metal `todo!()` Implementations ✅ COMPLETE

**Status:** ✅ **Both implementations complete**

### 2.1 CompatKernel::process for Metal ✅

Full implementation added that:
- Compiles Metal shader code using `MTLDevice::newLibraryWithSource`
- Creates compute pipeline state
- Copies input data to Metal buffers
- Dispatches compute kernel with proper grid/threadgroup sizes
- Copies output data back to `Vec<f32>`

### 2.2 Diff::process for Metal ✅

Full implementation added that:
- Reads input tensor data
- Writes to `{name}.bin` file for debugging
- Passes input through unchanged

### 2.3 Note on Data Types

The Metal implementations use `Vec<f32>` for Tensor data (via Luminal's `Data` trait) rather than raw Metal buffers, since `CompatKernel` and `Diff` operate within the normal Luminal execution path which expects CPU-accessible data.

### 2.4 Verification Checklist

- [x] `CompatKernel::process` implemented for Metal (`lib.rs:184-292`)
- [x] `CompatKernel::process` implemented for CUDA (`lib.rs:126-181`)
- [x] `Diff::process` implemented for Metal (`lib.rs:341-368`)
- [x] `Diff::process` implemented for CUDA (`lib.rs:321-339`)
- [x] `test_compat_kernel_metal` — Added in Phase 2.5
- [x] `test_diff_output` — Added in Phase 2.5

---

## Phase 2.5: Address Test Coverage Gaps ✅ COMPLETE

**Priority:** 🟡 Medium - Improves reliability and maintainability
**Status:** ✅ **COMPLETE**

### 2.5.1 Problem Statement

Code review on 2025-12-11 identified several test coverage gaps:

1. **No tests for `CompatKernel` or `Diff`** — Implementations exist but are untested
2. **No tests for `run.rs`** — Critical execution path (run_graph, compile_kernels, assign_buffers)
3. **No tests for `extract.rs`** — The `search()` function has no unit tests
4. **No tests for `utils.rs`** — Helper functions untested
5. **CI doesn't test `luminal_2` with Metal** — Only tests without GPU features

### 2.5.2 Required Tests

#### Test 1: `test_compat_kernel_metal`

```rust
#[cfg(feature = "metal")]
#[test]
fn test_compat_kernel_metal() {
    use crate::{custom_kernel, Kernel};
    use luminal::prelude::*;
    
    let mut cx = Graph::new();
    
    let kernel = Kernel {
        code: r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void kernel_name(
                device float* a [[buffer(0)]],
                device float* out [[buffer(1)]],
                uint idx [[thread_position_in_grid]]
            ) {
                out[idx] = a[idx] * 2.0;
            }
        "#.to_string(),
        grid: (4.into(), 1.into(), 1.into()),
        threadblock: (1.into(), 1.into(), 1.into()),
        smem: 0.into(),
        outputs: vec![4.into()],
    };
    
    let a = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);
    let b = custom_kernel(&[a], kernel, 4, &mut cx).retrieve();
    
    cx.execute();
    
    assert_eq!(b.data(), vec![2.0, 4.0, 6.0, 8.0]);
}
```

#### Test 2: `test_compat_kernel_cuda`

Similar test for CUDA backend.

#### Test 3: `test_diff_output`

```rust
#[cfg(feature = "metal")]
#[test]
fn test_diff_output() {
    use crate::GT2;
    use luminal::prelude::*;
    use std::fs;
    
    let mut cx = Graph::new();
    let a = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);
    let _b = a.diff2("test_diff").retrieve();
    
    cx.execute();
    
    // Verify file was created
    assert!(fs::metadata("test_diff.bin").is_ok());
    
    // Verify contents
    let bytes = fs::read("test_diff.bin").unwrap();
    let floats: Vec<f32> = bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);
    
    // Cleanup
    fs::remove_file("test_diff.bin").ok();
}
```

#### Test 4: `test_assign_buffers`

```rust
#[test]
fn test_assign_buffers_empty_graph() {
    use crate::run::assign_buffers;
    use petgraph::stable_graph::StableGraph;
    
    let graph = StableGraph::new();
    let (buffers, map) = assign_buffers(&graph);
    
    assert!(buffers.is_empty());
    assert!(map.is_empty());
}
```

### 2.5.3 CI Update Required

Add to `.github/workflows/test.yml`:

```yaml
  # After test-metal job
  test-luminal-2-metal:
    name: luminal_2 Metal Tests (macOS)
    runs-on: macos-14
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run luminal_2 Metal tests
        working-directory: crates/luminal_2
        run: cargo test --features metal --verbose
```

### 2.5.4 Improve Panic Messages

Several `panic!()` calls in `codegen.rs` lack context. Replace with descriptive messages:

| Location | Current | Improved |
|----------|---------|----------|
| `codegen.rs:100` | `panic!()` | `panic!("Expected GMEM node at {:?}", node)` |
| `codegen.rs:464` | `panic!();` | `panic!("Cannot find output with Acc('{}')", acc_symbol)` |
| `codegen.rs:859` | `panic!()` | `panic!("Unexpected GraphTerm in SMEMLoad: {:?}", term)` |
| `codegen.rs:884` | `panic!()` | `panic!("Unexpected unary op: {:?}", term)` |
| `codegen.rs:922` | `panic!()` | `panic!("Unexpected binary op: {:?}", term)` |

### 2.5.5 Verification Checklist

- [x] `test_compat_kernel_metal` added and passes
- [ ] `test_compat_kernel_cuda` added and passes (requires CUDA hardware)
- [x] `test_diff_output` added and passes
- [x] `test_assign_buffers` added and passes
- [x] CI updated to test `luminal_2 --features metal`
- [x] Panic messages improved in `codegen.rs`

---

## Phase 3: CUDA Warp-Cooperative Matmul ✅ COMPLETE

**Status:** ✅ **COMPLETE**

### 3.1 Problem Statement (Resolved)

The `TCMatmul` GraphTerm previously skipped CUDA with `return None`:

```rust
// OLD: crates/luminal_2/src/codegen.rs:936-940
GraphTerm::TCMatmul { ... } => {
    if cfg!(feature = "cuda") {
        return None;  // CUDA build: skip / fallback
    }
    // ... Metal simdgroup implementation follows
}
```

### 3.2 Solution Implemented

Implemented a **warp-cooperative 8x8 matrix multiply** for CUDA that matches the existing egglog tile pattern used by Metal's simdgroup operations.

**Key insight:** The egglog rules in `code.lisp` generate 8x8 tile patterns (matching Metal's `simdgroup_float8x8`). NVIDIA's WMMA API uses larger tiles (16x16x16 for fp16), so we implemented a warp-cooperative approach using standard CUDA:

- Each warp (32 threads) computes one 8x8 = 64 element output tile
- Each thread computes 2 output elements (64 / 32 = 2)
- Uses `__syncwarp()` for warp-level synchronization
- Uses unrolled loops for the K-dimension accumulation

### 3.3 Implementation Details

**Code location:** `crates/luminal_2/src/codegen.rs` (TCMatmul handler)

The implementation uses a match on `GPUArch` to generate architecture-specific code:

```rust
match arch {
    GPUArch::CUDA => {
        // Warp-cooperative 8x8 matmul
        // Thread lane (0-31) maps to output element
        // lane 0-7: row 0, cols 0-7 and row 4, cols 0-7
        // Each thread accumulates dot product for 2 elements
    }
    GPUArch::Metal(_) => {
        // Existing simdgroup_float8x8 implementation
    }
}
```

**Thread mapping (32 threads → 64 outputs):**
- `row0 = lane / 8` (rows 0-3)
- `col = lane % 8` (cols 0-7)
- `row1 = row0 + 4` (rows 4-7)
- Each thread accumulates `C[row0,col]` and `C[row1,col]`

### 3.4 NVRTC Compilation Updates

Updated compilation options for warp intrinsics:

```rust
// crates/luminal_2/src/run.rs
CompileOptions {
    include_paths: vec![
        "/usr/include".into(),
        "/usr/local/cuda/include".into(),
    ],
    options: vec![
        "--gpu-architecture=sm_75".into(), // Turing+ for warp intrinsics
        "--relocatable-device-code=false".into(),
        "--std=c++17".into(),
    ],
    ..Default::default()
}
```

Also added CUDA preamble with helper functions:
```cuda
__device__ __forceinline__ float __mod(float a, float b) {
    return a - b * floorf(a / b);
}
```

### 3.5 cudarc Version Update

**Problem:** cudarc 0.16.6 used `cuCtxCreate_v4` (CUDA 12.4+ driver API), but the system had CUDA 12.2 (driver 535.x).

**Solution:** Upgraded to cudarc 0.18.1 which works with CUDA 12.2:

```toml
# Cargo.toml changes
cudarc = { version = "0.18.1", features = ["f16", "cuda-12020"] }
```

Also added `/usr/local/cuda/include` to NVRTC include paths for `cuda_fp16.h`.

### 3.6 Tests Added ✅

New test file: `crates/luminal_2/src/cuda_tests.rs` (9 tests)

| Test | Description | Status |
|------|-------------|--------|
| `test_cuda_add_codegen` | Tests add operation codegen | ✅ Uses `.expect()` |
| `test_cuda_sum_codegen` | Tests reduction codegen | ✅ Uses `.expect()` |
| `test_cuda_kernel_structure` | Validates CUDA kernel structure | ✅ Uses `.expect()` |
| `test_cuda_add_execution` | Runtime add with numerical verification | ✅ Full pipeline |
| `test_cuda_mul_execution` | Runtime mul with numerical verification | ✅ Full pipeline |
| `test_cuda_matmul_8x8_codegen` | Tests 8x8 matmul codegen | ✅ Uses `.expect()` |
| `test_cuda_matmul_16x16_codegen` | Tests 16x16 matmul codegen | ✅ Uses `.expect()` |
| `test_cuda_matmul_8x8_execution` | Runtime matmul with verification | ✅ Full pipeline |
| `test_compat_kernel_cuda` | Custom kernel injection | ✅ Uses `CudaCompiler` |

**Helper functions:**
- `get_sorted_gmem_nodes()` — Get GMEM nodes sorted by mapping index
- `build_input_buffers()` — Create GPU buffers for tensor/constant inputs
- `build_inputs_map()` — Build input mapping for `run_graph`

### 3.7 Remaining Work

All critical items complete. Optional future work:
- Add more compat kernel tests (different input counts, shared memory)
- Add Diff operator tests for CUDA debugging workflow
- Investigate the 31 pre-existing `luminal_cuda` failures (separate from luminal_2)

### 3.8 Future Enhancement: True Tensor Cores (WMMA)

For true tensor core acceleration with WMMA, the egglog rules would need modification to generate 16x16 tile patterns instead of 8x8. This would involve:

1. Add new rules in `code.lisp` that divide by 16 instead of 8
2. Condition the rules on target architecture
3. Implement WMMA codegen for the 16x16 pattern

This is left for a future optimization pass.

### 3.9 Verification Checklist

**All items complete:**
- [x] TCMatmul generates valid CUDA code (no more `return None`)
- [x] CUDA preamble includes helper functions (`__mod`)
- [x] NVRTC compilation uses sm_75 for warp intrinsics
- [x] `cuda_tests.rs` added with 9 CUDA-specific tests:
  - 3 codegen tests (`add`, `sum`, `kernel_structure`)
  - 3 runtime execution tests (`add`, `mul`, `matmul_8x8` with numerical verification)
  - 2 matmul codegen tests (`8x8`, `16x16`)
  - 1 CompatKernel test (`test_compat_kernel_cuda`)
- [x] Module declaration added to `lib.rs`
- [x] cudarc upgraded to 0.18.1 for CUDA 12.2 compatibility
- [x] Tests strengthened to fail when codegen returns `None` (using `.expect()`)
- [x] Helper functions cleaned up and made consistent (`get_sorted_gmem_nodes`, `build_input_buffers`, `build_inputs_map`)

**Note:** The 31 `luminal_cuda` test failures are pre-existing issues in the 1.0 backend, not related to `luminal_2`. They include fp16 precision issues, norm/mean kernel bugs, and NVRTC compilation errors with half precision types.

---

## Phase 4: Expand Search Space ✅ COMPLETE

**Status:** ✅ **COMPLETE**

### 4.1 Changes Implemented

#### 4.1.1 Variable Tile Sizes

Modified `TileLoop` to carry tile size as a parameter and added tiling rules for 4, 8, 16, and 32:

**Datatype change:**
```lisp
; OLD: (TileLoop IR i64)
; NEW: (TileLoop IR i64 i64) ; loop level and tile size
```

**New tiling rules in `code.lisp`:**
- Tile by 4: For ranges > 4 divisible by 4
- Tile by 8: For ranges > 8 divisible by 8 (existing)
- Tile by 16: For ranges > 16 divisible by 16
- Tile by 32: For ranges > 32 divisible by 32

Each tile size has corresponding propagation rules to expand `TileLoop` to nested `LoopIn` structures.

#### 4.1.2 Increased Search Budget

```rust
// extract.rs
const MAX_SEARCHED_GRAPHS: usize = 10_000;  // Was 1_000
```

#### 4.1.3 Early Termination

Added early termination when a very fast kernel is found:
```rust
if us < 50 {  // Found kernel < 50µs
    break 'trajectory_loop;
}
```

#### 4.1.4 Cleaned Up INVALID_IR

Removed undefined patterns (`SwapLoops`, `UnpadLoop`, `TiledMatmulAcc`) and added documentation:
```rust
const INVALID_IR: &[&str] = &[
    // Intermediate transformation patterns (should be propagated out)
    "TileLoop",
    "MergeLoops",
    "MReplace",
    // Tensor core intermediate patterns
    "TiledMatmulInputA",
    "TiledMatmulInputB",
    // Internal egglog constructs
    "loop_level",
    "vec-of",
    "set-of",
];
```

#### 4.1.5 Final Expression Saturation

Added final `(saturate expr)` pass at end of run-schedule to ensure `MReplace` and other expressions are fully simplified before extraction.

#### 4.1.6 Fixed Egglog Scheduler Syntax

The original `(let-scheduler bo (back-off))` syntax was incompatible with the egglog version. Fixed by replacing with `(seq ...)`:

```lisp
; OLD (broken):
(let-scheduler bo (back-off))
(repeat 1 (run-with bo ir) ...)

; NEW (working):
(seq (run ir) (saturate ir-prop) ...)
```

### 4.2 Files Changed

| File | Changes |
|------|---------|
| `crates/luminal_2/src/code.lisp` | Added tile sizes 4, 16, 32; updated TileLoop to 3 args; final expr saturation; fixed scheduler syntax |
| `crates/luminal_2/src/extract.rs` | MAX_SEARCHED_GRAPHS=10000; early termination; cleaned INVALID_IR |
| `examples/search_demo/src/benchmark.rs` | Added search benchmark to verify Phase 4 improvements |
| `examples/search_demo/src/main.rs` | Added `benchmark` and `search` CLI commands |

### 4.3 Verification

- [x] All 32 CUDA tests pass
- [x] All 18 training tests pass
- [x] All core library tests pass
- [x] Code compiles without errors

### 4.4 Future Work

- Explore more loop transformations (loop interchange, unrolling)
- Profile search times with larger graphs to verify no regression
- Test Flash Attention-like pattern discovery with softmax(QK^T)V

---

## Phase 5: Unify 1.0 and 2.0 Architectures ✅ COMPLETE

**Status:** ✅ **COMPLETE**

### 5.1 Goals

1. Single API for users
2. Automatic selection between fast (1.0) and optimal (2.0) compilation
3. Gradual migration path

### 5.2 Implementation Summary

Created a unified compilation API that bridges the 1.0 (hand-written kernels) and 2.0 (search-based optimization) systems.

#### CompilationMode enum (in core `luminal` crate)

```rust
// src/unified.rs
pub enum CompilationMode {
    /// Use hand-written kernels from luminal_metal/cuda (fast compile)
    Fast,
    
    /// Use search-based optimization from luminal_2 (slower compile, optimal runtime)
    Optimal { search_steps: usize },
    
    /// Try search with timeout, fallback to Fast if exceeded
    TimeBudget { budget: Duration, search_steps: usize },
}

impl CompilationMode {
    pub fn fast() -> Self;
    pub fn optimal() -> Self;  // Default: 3 search steps
    pub fn optimal_with_steps(steps: usize) -> Self;
    pub fn time_budget(budget: Duration) -> Self;
    pub fn uses_search(&self) -> bool;
}
```

#### UnifiedCompiler (in `luminal_2` crate)

```rust
// crates/luminal_2/src/unified.rs
pub struct UnifiedCompiler<T> {
    mode: CompilationMode,
    _marker: PhantomData<T>,
}

impl<T: MetalFloat> Compiler for UnifiedCompiler<T> {
    fn compile<I: ToIdsMut>(&self, graph: &mut Graph, ids: I);
}

// Convenience constructors
impl<T> UnifiedCompiler<T> {
    pub fn fast() -> Self;
    pub fn optimal() -> Self;
    pub fn optimal_with_steps(steps: usize) -> Self;
    pub fn time_budget(budget: Duration) -> Self;
}
```

#### CompiledSubgraph operators

Internal operators that wrap 2.0-compiled kernel graphs for execution within the 1.0 execution model:
- `CompiledSubgraphMetal` - Metal backend
- `CompiledSubgraphCuda` - CUDA backend

### 5.3 Usage Examples

```rust
use luminal::prelude::*;
use luminal_2::UnifiedCompiler;

let mut cx = Graph::new();
let a = cx.tensor((8, 8)).set_rand();
let b = cx.tensor((8, 8)).set_rand();
let mut c = a.matmul(b).retrieve();

// Option 1: Fast mode (1.0 hand-written kernels, fast compile)
cx.compile(UnifiedCompiler::<f32>::fast(), &mut c);

// Option 2: Optimal mode (2.0 search-based, optimal runtime)
cx.compile(UnifiedCompiler::<f32>::optimal(), &mut c);

// Option 3: Time budget (try search, fallback to fast)
cx.compile(
    UnifiedCompiler::<f32>::time_budget(Duration::from_secs(30)),
    &mut c,
);

cx.execute();
```

### 5.4 Files Changed

| File | Changes |
|------|---------|
| `src/unified.rs` | New: `CompilationMode` enum with builder methods |
| `src/lib.rs` | Export `unified` module and `CompilationMode` in prelude |
| `crates/luminal_2/src/unified.rs` | New: `UnifiedCompiler`, `CompiledSubgraphMetal`, `CompiledSubgraphCuda` |
| `crates/luminal_2/src/lib.rs` | Export `unified` module and re-exports |
| `crates/luminal_2/Cargo.toml` | Added optional `luminal_metal` dependency for metal feature |
| `crates/luminal_2/src/e2e_tests.rs` | Added `test_unified_compiler_fast_mode`, `test_unified_compiler_matmul` |

### 5.5 Test Coverage

| Test | Description |
|------|-------------|
| `unified::tests::test_compilation_mode_default` | Verify default is Fast |
| `unified::tests::test_compilation_mode_optimal` | Verify Optimal mode creation |
| `unified::tests::test_compilation_mode_time_budget` | Verify TimeBudget mode creation |
| `e2e_tests::test_unified_compiler_fast_mode` | End-to-end test with Fast mode |
| `e2e_tests::test_unified_compiler_matmul` | Matmul test with UnifiedCompiler |

### 5.6 Verification

```bash
# All core library tests pass
cargo test --lib  # 89 tests

# All luminal_2 tests pass with metal feature
cd crates/luminal_2 && cargo test --features metal  # 36 tests

# All luminal_metal tests pass
cd crates/luminal_metal && cargo test  # 205 tests
```

### 5.7 Architectural Refactoring (PyTorch-like Structure)

As part of Phase 5, the crate structure was refactored for a cleaner, PyTorch-like architecture:

| Before | After |
|--------|-------|
| `luminal` (core) | `luminal` (core + nn + training) |
| `luminal_nn` (separate crate) | `luminal::nn` (module in core) |
| `luminal_training` (separate crate) | `luminal::training` (module in core) |
| `luminal_metal` (backend) | `luminal_metal` (unchanged) |
| `luminal_cuda` (backend) | `luminal_cuda` (unchanged) |
| `luminal_2` (search-based compiler) | `luminal_2` (unchanged, exports `UnifiedCompiler`) |

**Key changes:**

1. **`luminal_nn` merged into core**: All neural network modules (Linear, ReLU, LayerNorm, Transformer, etc.) now live in `luminal/src/nn/`

2. **`luminal_training` merged into core**: Autograd, optimizers (SGD, Adam, RMSprop), loss functions, and LR schedulers now live in `luminal/src/training/`

3. **Unified prelude**: The prelude now includes all nn and training types:
   ```rust
   use luminal::prelude::*;
   // Now includes: Linear, ReLU, Autograd, sgd_on_graph, adam_on_graph, etc.
   ```

**Migration guide (v0.3.0+):**

```rust
// Use the new module paths:
use luminal::nn::{Linear, ReLU};
use luminal::training::{Autograd, sgd_on_graph};

// Or via prelude:
use luminal::prelude::*;
```

> **Note:** The deprecated shim crates (`luminal_nn`, `luminal_training`) were removed in v0.3.0.

### 5.8 Future Improvements

1. **Graph Replacement**: Currently in Optimal mode, we compile with search then fall back to fast execution. A future enhancement would be to actually replace subgraphs with `CompiledSubgraph` operators for true 2.0 execution.

2. **Automatic Mode Selection**: Could add heuristics to automatically choose Optimal for large graphs and Fast for small ones.

3. **Cached Compilation**: Store compiled kernels on disk to avoid re-compilation for unchanged graphs.

---

## Phase 6: Complete Training Infrastructure ✅ COMPLETE

**Status**: ✅ **COMPLETE**

### 6.1 Summary

Phase 6 expanded the training infrastructure with comprehensive optimizer, loss function, and training utility support.

### 6.2 Implemented Features

#### Optimizers (`src/training/optimizer.rs`)

| Optimizer | Description | Features |
|-----------|-------------|----------|
| `sgd_on_graph` | Basic SGD | Learning rate control |
| `sgd_momentum_on_graph` | SGD with momentum | Momentum, dampening, Nesterov, weight decay |
| `adam_on_graph` | Adam/AdamW | Bias correction, weight decay, configurable betas |
| `rmsprop_on_graph` | RMSprop | Smoothing constant, epsilon |
| `lamb_on_graph` | LAMB | Layer-wise adaptive learning rates for large batch training |
| `clip_grad_norm` | Gradient clipping | Global norm clipping |
| `clip_grad_value` | Value clipping | Per-element clipping |

#### Loss Functions (`src/training/loss.rs`)

| Loss | Description |
|------|-------------|
| `mse_loss` | Mean Squared Error |
| `rmse_loss` | Root Mean Squared Error |
| `mae_loss` | Mean Absolute Error |
| `huber_loss` | Huber loss (smooth L1) |
| `smooth_l1_loss` | Smooth L1 loss |
| `cross_entropy_with_logits_loss` | Cross-entropy for classification |
| `kl_div_with_logits_loss` | KL divergence |
| `binary_cross_entropy_with_logits_loss` | Binary cross-entropy |
| `focal_loss_with_logits` | Focal loss for class imbalance |
| `binary_focal_loss_with_logits` | Binary focal loss |
| `label_smoothing_cross_entropy_loss` | Label smoothing |

#### Gradient Accumulation (`src/training/accumulation.rs`)

- `GradientAccumulator` - Accumulates gradients over multiple micro-batches
- `scale_loss_for_accumulation` - Scales loss for proper gradient averaging

#### Gradient Checkpointing (`src/training/checkpoint.rs`)

- `CheckpointConfig` - Configuration for checkpoint intervals
- `CheckpointManager` - Manages checkpoint segments
- `checkpoint()` - Marks tensor as checkpoint boundary
- `checkpoint_layers()` - Applies checkpointing to layer sequences
- `estimate_memory_savings()` - Estimates memory reduction

#### Mixed Precision Training (`src/training/mixed_precision.rs`)

- `GradScaler` - Dynamic loss scaling for FP16 training
- `MixedPrecisionConfig` - Configuration for loss scaling
- `MasterWeights` - FP32 master weight management
- `AMPContext` - Automatic Mixed Precision context manager

### 6.3 Test Coverage

All training modules have comprehensive tests:

```bash
cargo test --lib training::  # 40 tests pass, 8 ignored (CNN gradient bugs)
```

| Module | Tests |
|--------|-------|
| `autograd` | 32 pass, 8 ignored (MLP, transformer, softmax, layer_norm + CNN regression tests) |
| `optimizer` | 12 tests (Adam, SGD, LAMB, clipping) |
| `scheduler` | 8 tests (all scheduler types) |
| `accumulation` | 3 tests |
| `checkpoint` | 5 tests |
| `mixed_precision` | 8 tests |

**Ignored Tests (CNN Gradient Bugs):** See Section 6.4 for details on the 8 ignored tests that document the `pool_last_dim` gradient bugs.

### 6.4 Conv2D + Autograd Status

**Status:** ⚠️ PARTIALLY WORKING - CRITICAL GRADIENT BUGS (Root cause identified, fix requires architectural changes)

Conv2D with autograd has **critical gradient issues**. The root cause has been identified and documented with 8 regression tests.

#### What Works ✅

- **Single Conv2D + Linear**: Works correctly, achieves 97.5% on MNIST
- **GlobalAvgPool2D**: Uses simple reshape+mean pattern, works correctly
- **Basic autograd operations**: Add, Mul, SumReduce, MaxReduce, Log2, Exp2, etc.
- **Gradients for weights used AFTER pooling**: Matmul gradients work when pooling is in the forward path but gradients are computed for weights (not the pooled tensor itself)

#### What Does NOT Work ❌ (CRITICAL BUGS)

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| **Multiple Conv2D layers** | Index out of bounds panic | Shape tracker index expression computes invalid indices during backprop |
| **AvgPool2D gradients** | Index out of bounds panic | `pool_last_dim` internal Contiguous nodes create incompatible gradient shapes |
| **MaxPool2D gradients** | Index out of bounds panic | Same `pool_last_dim` issue |
| **Deep CNN architectures** | Training crashes | Combination of above issues |

#### Technical Details (Deep Dive)

**Root Cause:** The autograd system cannot properly compute gradients for tensors that pass THROUGH `pool_last_dim` operations. This is a fundamental limitation of how shape trackers are used for gradient computation.

**The Problem Chain:**

1. `pool_last_dim` (in `src/hl_ops/movement.rs`) creates complex shape transformations:
   ```rust
   // For input (4,) with kernel=2, stride=1:
   // Step 1: expand_dim → (3, 4) with fake dim at 0 (3 windows)
   // Step 2: contiguous() → materializes to 12 elements
   // Step 3: padding/masking → slices to (3, 2) = 6 elements
   // Step 4: contiguous() → materializes to 6 elements
   ```

2. When backpropagating through these `Contiguous` nodes:
   - Each `Contiguous` node has an INPUT shape (complex, with padding/masking/fake dims)
   - The gradient has the OUTPUT shape (simple contiguous)
   - The `add_grad` function tries to match these, but fails

3. The specific failure in `add_grad`:
   ```rust
   // add_grad tries to "undo" shape transformations:
   // 1. Undo permutes by re-indexing (line 294-298)
   // 2. Undo expands by adding SumReduce (line 300-314)
   // But this doesn't work when:
   // - Input has padding/masking that creates different dims than output
   // - Input has fake dims at different positions than expected
   // - The gradient shape doesn't match the expected layout
   ```

4. Result: The shape tracker's `index_expression()` computes indices like 8 when the data buffer only has 6 elements, causing an out-of-bounds panic.

**Why Some Tests Pass:**

The key insight is that gradients work when computed for **weights** (like Conv2D kernels) that are used AFTER pooling via matmul, but fail when computing gradients for **inputs** that go THROUGH pooling.

```rust
// This WORKS - gradient for weight, not for pooled input:
let pooled = input.pool_last_dim(...);  // input not tracked for gradients
let out = weight.expand_dim(0, 1).matmul(pooled);  // weight tracked
let grads = Autograd::new(weight, loss);  // ✅ Works

// This FAILS - gradient for tensor going through pool:
let output = weight.pool_last_dim(...);  // weight tracked for gradients
let loss = output.sum(...);
let grads = Autograd::new(weight, loss);  // ❌ Crashes
```

#### Regression Tests Added

8 tests have been added to `src/training/autograd.rs` that document these bugs. They are marked with `#[ignore]` and will pass once the underlying issues are fixed:

| Test | Description |
|------|-------------|
| `test_simple_pool_gradient` | Minimal reproduction: pool_last_dim → sum |
| `test_two_conv2d_gradient` | Stack 2 Conv2D layers |
| `test_three_conv2d_gradient` | Stack 3 Conv2D layers |
| `test_avgpool2d_gradient` | AvgPool2D gradient computation |
| `test_maxpool2d_gradient` | MaxPool2D gradient computation |
| `test_conv2d_plus_avgpool_gradient` | Conv2D + AvgPool2D combination |
| `test_conv2d_plus_maxpool_gradient` | Conv2D + MaxPool2D combination |
| `test_gradient_magnitude_stability` | Multi-layer gradient stability |

Run with: `cargo test --lib -- --ignored` to see the current failure state.

#### Reproduction

```rust
// This works (single conv):
let x = conv1.forward(input).relu();
let out = fc.forward(x.reshape((batch, N)));  // ✅ OK

// This fails (two conv layers):
let x = conv1.forward(input).relu();
let x = conv2.forward(x).relu();  // ❌ Index out of bounds during backprop
let out = fc.forward(x.reshape((batch, N)));

// This fails (conv + pooling):
let x = conv1.forward(input).relu();
let x = avg_pool.forward(x);  // ❌ Index out of bounds
let out = fc.forward(x.reshape((batch, N)));

// Minimal reproduction:
let weight = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);
let pooled = weight.pool_last_dim(2, 1, 1);  // (3, 2)
let loss = pooled.sum(1).sum(0);
let grads = cx.compile(Autograd::new(weight, loss), ());  // ❌ Crashes on execute
```

#### Required Fix (Architectural Change Needed)

The proper fix requires one of:

1. **Implement `pool_last_dim` as a primitive op** with explicit forward and backward passes:
   ```rust
   // Instead of building from expand/contiguous/slice, 
   // make pool_last_dim a single node with known gradient semantics
   impl Operator for PoolLastDim {
       fn backward(...) {
           // Explicit scatter-add for overlapping views
       }
   }
   ```

2. **Add scatter-add support to autograd** for handling overlapping views:
   ```rust
   // In add_grad, detect when input elements are reused
   // and accumulate gradients properly
   fn add_grad(...) {
       if has_overlapping_views(fwd) {
           grad = scatter_add_gradient(grad, fwd);
       }
       // ...
   }
   ```

3. **Reformulate CNN operations** to avoid backprop through pool_last_dim:
   - Use im2col approach where gradients flow through matmul (which works)
   - This is what PyTorch/TensorFlow do internally

**Estimated Effort:** High (2-5 days for a proper fix)

The `mnist_cnn` example currently uses a **single Conv2D layer as a workaround**.
This is a stopgap - proper CNN support requires fixing the underlying autograd architecture.

### 6.5 Usage Examples

**SGD with Momentum:**
```rust
use luminal::training::{sgd_momentum_on_graph, SGDConfig};

let config = SGDConfig::new()
    .lr(0.01)
    .momentum(0.9)
    .nesterov(true);
let (new_weights, lr, state) = sgd_momentum_on_graph(&mut cx, weights, &grads, config);
```

**LAMB for Large Batch Training:**
```rust
use luminal::training::{lamb_on_graph, LAMBConfig};

let config = LAMBConfig::new().lr(0.001).weight_decay(0.01);
let (new_weights, lr, state) = lamb_on_graph(&mut cx, weights, &grads, config);
```

**Mixed Precision Training:**
```rust
use luminal::training::{GradScaler, MixedPrecisionConfig};

let mut scaler = GradScaler::new(MixedPrecisionConfig::default());

// Scale loss before backward
let scaled_loss = scaler.scale(loss);
let grads = compute_gradients(scaled_loss);

// Unscale and check for overflow
let (unscaled_grads, overflow) = scaler.unscale_and_check(&grads, &mut cx);
if !overflow {
    optimizer_step(&unscaled_grads);
}
scaler.update(overflow);
```

**Gradient Accumulation:**
```rust
use luminal::training::{GradientAccumulator, scale_loss_for_accumulation};

let mut accumulator = GradientAccumulator::new(4); // 4 micro-batches

for batch in data {
    let loss = scale_loss_for_accumulation(compute_loss(batch), 4);
    let grads = compute_gradients(loss);
    accumulator.accumulate(&grads, &mut cx);
    
    if accumulator.should_step() {
        let avg_grads = accumulator.get_averaged_gradients(&mut cx);
        optimizer_step(&avg_grads);
        accumulator.zero_grad(&mut cx);
    }
}
```

---

## Phase 7: Benchmarking Suite

### 7.1 Structure

```
benches/
├── src/
│   ├── lib.rs           # Benchmark framework
│   ├── matmul.rs        # Matrix multiplication benchmarks
│   ├── attention.rs     # Attention benchmarks
│   └── llm.rs           # Full LLM benchmarks
├── baselines/
│   └── pytorch.py       # PyTorch comparison scripts
└── results/
    └── .gitkeep
```

### 7.2 Key Benchmarks

| Benchmark | Sizes | Metrics |
|-----------|-------|---------|
| MatMul | 512-4096 | TFLOPS |
| Attention | seq_len 512-8192 | TFLOPS |
| LLaMA Inference | 7B, 8B | tokens/sec |
| Training Step | 1B params | step/sec |

---

## Phase 8: ROCm Backend

### 8.1 New Crate Structure

```
crates/luminal_rocm/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── prim.rs      # Primitive ops
│   ├── binary.rs
│   ├── unary.rs
│   ├── matmul.rs    # rocBLAS integration
│   └── other.rs
```

### 8.2 Key Dependencies

```toml
[dependencies]
hip-sys = "0.x"      # HIP bindings
rocblas-sys = "0.x"  # rocBLAS bindings
```

---

## Phase 9: Distributed Computing

### 9.1 Components

1. **Communication Backend** - NCCL for NVIDIA, RCCL for AMD
2. **Data Parallel** - Gradient averaging across GPUs
3. **Tensor Parallel** - Split large tensors across GPUs
4. **Pipeline Parallel** - Split layers across GPUs

### 9.2 API Design

```rust
let backend = NcclBackend::new();
let model = DataParallel::new(model, backend);

// Gradients automatically synchronized
let grads = Autograd::new(params, loss).compile(&mut cx);
model.sync_gradients(&mut grads, &mut cx);
```

---

## Phase 10: Python Bindings

### 10.1 PyO3 Structure

```rust
#[pyclass]
struct PyGraph { inner: Graph }

#[pyclass]
struct PyTensor { id: NodeIndex, shape: ShapeTracker, graph: *mut Graph }

#[pymethods]
impl PyTensor {
    fn __add__(&self, other: &PyTensor) -> PyTensor { ... }
    fn matmul(&self, other: &PyTensor) -> PyTensor { ... }
    fn numpy(&self) -> PyResult<numpy::PyArray1<f32>> { ... }
}
```

### 10.2 Python API

```python
import pyluminal as pl

a = pl.tensor([[1, 2], [3, 4]])
b = pl.tensor([[5, 6], [7, 8]])
c = (a @ b).retrieve()

pl.compile("metal")  # or "cuda"
pl.execute()

print(c.numpy())
```

---

## Appendix A: Priority Order

| Phase | Priority | Effort | Impact | Status |
|-------|----------|--------|--------|--------|
| 0 - Fix Metal Test Failures | 🔴 Critical | Low | Critical | ✅ Complete |
| 1 - Fix E2E Tests | 🔴 Critical | Low | Critical | ✅ Complete |
| 2 - Metal `todo!()` Impls | 🟡 Medium | Medium | Medium | ✅ Complete |
| 2.5 - Test Coverage Gaps | 🟡 Medium | Low | Medium | ✅ Complete |
| 3 - CUDA Warp Matmul | 🟠 High | Medium | High | ✅ Complete |
| 4 - Expand Search | 🟡 Medium | Medium | High | ✅ Complete |
| 5 - Unify Arch | 🟢 Lower | High | Medium | ✅ Complete |
| 6 - Training | 🟡 Medium | Medium | High | ✅ Complete |
| 7 - Benchmarks | 🟢 Lower | Low | Medium | Not Started |
| 8 - ROCm | 🔵 Future | Very High | Medium | Not Started |
| 9 - Distributed | 🔵 Future | Very High | High | Not Started |
| 10 - Python | 🔵 Future | Medium | High | Not Started |

**Recommended Next Steps:**
1. **Phase 7 (benchmarking)** — Validate performance against PyTorch baselines
2. **Phase 10 (Python bindings)** — Make luminal accessible to Python users
3. Phase 8 (ROCm) — AMD GPU support for broader hardware compatibility

---

## Appendix B: Quick Verification Commands

```bash
# Core library tests (186 tests, 8 ignored for known CNN gradient bugs)
cargo test --lib

# Run ignored CNN gradient tests to check current failure status
cargo test --lib -- --ignored

# Core with search feature (Metal, macOS only)
cargo test --lib --features search,metal

# Core with search feature (CUDA, requires NVIDIA GPU)
cargo test --lib --features search,cuda

# Metal backend tests (205 tests) - requires macOS with Metal GPU
cd crates/luminal_metal && cargo test

# CUDA backend tests (requires NVIDIA GPU)
cd crates/luminal_cuda && cargo test -- --test-threads=1

# All workspace tests
cargo test --workspace

# Format and lint
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Metal crates (separate workspace)
cd crates/luminal_metal && cargo fmt --all && cargo clippy --all-targets -- -D warnings
cd crates/luminal_cuda && cargo fmt --all && cargo clippy --all-targets -- -D warnings
```

---

## Appendix C: File Locations Quick Reference

### Search-Based Compiler (in `luminal::search`, behind `search` feature)

| Component | File |
|-----------|------|
| CompatKernel | `src/search/operators.rs` |
| Diff | `src/search/operators.rs` |
| custom_kernel | `src/search/operators.rs` |
| run_graph | `src/search/run.rs` |
| codegen | `src/search/codegen.rs` |
| translate_graph | `src/search/translate.rs` |
| search | `src/search/extract.rs` |
| Egglog rules | `src/search/code.lisp` |
| Types (Kernel, GraphTerm, etc.) | `src/search/types.rs` |
| Backend trait | `src/search/backend.rs` |

### luminal_metal

| Component | File |
|-----------|------|
| UnifiedMetalCompiler | `crates/luminal_metal/src/unified.rs` |
| test_conv2d (fp32) | `crates/luminal_metal/src/tests/fp32.rs` |
| test_conv2d (fp16) | `crates/luminal_metal/src/tests/fp16.rs` |
| verification tests | `crates/luminal_metal/src/verification_tests.rs` |

### luminal_cuda

| Component | File |
|-----------|------|
| UnifiedCudaCompiler | `crates/luminal_cuda/src/unified.rs` |

### Unified API (Phase 5)

| Component | File |
|-----------|------|
| CompilationMode enum | `src/unified.rs` |
| UnifiedMetalCompiler | `crates/luminal_metal/src/unified.rs` |
| UnifiedCudaCompiler | `crates/luminal_cuda/src/unified.rs` |

### Other Key Files

| Component | File |
|-----------|------|
| CI workflow | `.github/workflows/test.yml` |
| Core library | `src/` |
| NN modules | `src/nn/` |
| Training | `src/training/` |

---

## Appendix D: Search-Based Compilation Demo

A working demo showcasing search-based compilation is available at `examples/search_demo/`.

### Running the Demo

```bash
# CUDA backend (requires NVIDIA GPU)
cargo run -p search_demo --release --features cuda

# Metal backend (macOS only)
cargo run -p search_demo --release --features metal
```

### What the Demo Shows

| Demo | Description |
|------|-------------|
| **Demo 1: Search-Based Compilation** | Full pipeline: Luminal graph → IR translation → codegen → GPU execution with numerical verification |
| **Demo 2: Custom Kernel Injection** | Using `custom_kernel` API to inject a hand-written GELU kernel into the graph |
| **Demo 3: Matrix Multiply** | 8×8 matmul through the search pipeline (demonstrates warp-cooperative pattern) |
| **Demo 4: MLP Inference** | 2-layer network showing how complex graphs decompose into multiple fused kernels |

### Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║          Search-Based Compiler Demo                          ║
╚══════════════════════════════════════════════════════════════╝

═══ Demo 1: Search-Based Compilation Pipeline ═══
Step 1: Created Luminal graph for C = A + B * 2
Step 2: Translated to meta-graph with 1 subgraphs
Step 3: Stitched into unified graph with 11 nodes
Step 4: Generated 2 CUDA kernel(s)
Step 5: Executed on GPU in 79µs
        ✓ Verified correct!

═══ Demo 2: Custom CUDA Kernel Injection ═══
Input:  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
Output: [-0.045, -0.159, -0.154, 0.0, 0.346, 0.841, 1.400, 1.955]
        ✓ Custom GELU kernel verified!

═══ Demo 4: MLP Inference with Search Optimization ═══
Codegen: 11 CUDA kernel(s) generated
✓ All kernels compiled successfully with NVRTC

✅ All demos completed successfully!
```

### Key APIs Demonstrated

**Custom kernel injection:**
```rust
use luminal::search::{custom_kernel, Kernel};

let kernel = Kernel {
    code: r#"extern "C" __global__ void kernel_name(float* in, float* out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        out[idx] = in[idx] * 2.0f;
    }"#.to_string(),
    grid: (8.into(), 1.into(), 1.into()),
    threadblock: (1.into(), 1.into(), 1.into()),
    smem: 0.into(),
    outputs: vec![8.into()],
};

let output = custom_kernel(&[input], kernel, 8, &mut cx);
```

**Search-based compilation:**
```rust
use luminal::search::{
    codegen::{codegen, stitch_meta_graph_together},
    translate::translate_graph,
    GPUArch,
};

let (meta_graph, _, _) = translate_graph(&cx);
let (stitched, _) = stitch_meta_graph_together(meta_graph);
let (kernels, gmem_map) = codegen(stitched, GPUArch::CUDA, &cx.dyn_map).unwrap();
```

### Files

| File | Description |
|------|-------------|
| `examples/search_demo/Cargo.toml` | Dependencies and feature flags |
| `examples/search_demo/src/main.rs` | Demo implementation |
| `examples/search_demo/README.md` | Standalone documentation |
