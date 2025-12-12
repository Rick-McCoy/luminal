# Luminal 2.0 Implementation Plan

This document provides a step-by-step implementation guide to complete the search-based compilation system and realize Luminal's full potential.

**Last Updated:** 2025-12-12 — Phases 0-4 complete.

### Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Fix Metal test failures | ✅ Complete |
| 1 | Fix E2E tests | ✅ Complete |
| 2 | Metal `todo!()` implementations | ✅ Complete |
| 2.5 | Address test coverage gaps | ✅ Complete |
| 3 | CUDA warp-cooperative matmul | ✅ Complete |
| 4 | Expand search space | ✅ Complete |
| 5-10 | Remaining phases | Not started |

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Phase 0: Fix Failing Metal Backend Tests](#phase-0-fix-failing-metal-backend-tests) ✅ COMPLETE
3. [Phase 1: Fix Broken E2E Tests](#phase-1-fix-broken-e2e-tests) ✅ COMPLETE
4. [Phase 2: Complete Metal `todo!()` Implementations](#phase-2-complete-metal-todo-implementations) ✅ COMPLETE
5. [Phase 2.5: Address Test Coverage Gaps](#phase-25-address-test-coverage-gaps) ✅ COMPLETE
6. [Phase 3: CUDA Warp-Cooperative Matmul](#phase-3-cuda-warp-cooperative-matmul--complete) ✅ COMPLETE
7. [Phase 4: Expand Search Space](#phase-4-expand-search-space) ✅ COMPLETE
8. [Phase 5: Unify 1.0 and 2.0 Architectures](#phase-5-unify-10-and-20-architectures)
9. [Phase 6: Complete Training Infrastructure](#phase-6-complete-training-infrastructure)
10. [Phase 7: Benchmarking Suite](#phase-7-benchmarking-suite)
11. [Phase 8: ROCm Backend](#phase-8-rocm-backend)
12. [Phase 9: Distributed Computing](#phase-9-distributed-computing)
13. [Phase 10: Python Bindings](#phase-10-python-bindings)
14. [Appendix D: luminal_2 Demo](#appendix-d-luminal_2-demo)

---

## Current State Assessment

### Test Summary (as of 2025-12-12)

| Crate | Tests | Status |
|-------|-------|--------|
| `luminal` (core) | 89 | ✅ All pass |
| `luminal_nn` | 31 | ✅ All pass |
| `luminal_training` | 18 | ✅ All pass |
| `luminal_metal` | 205 | ✅ All pass |
| `luminal_2` (Metal) | 27 | ✅ All pass |
| `luminal_2` (CUDA) | 32 | ✅ All pass (9 CUDA-specific) |
| `luminal_cuda` | 168/199 | ⚠️ 31 pre-existing failures (fp16, norm, conv2d) |

### What Actually Works

After auditing and fixing the codebase, here's the current state:

| Component | Status | Notes |
|-----------|--------|-------|
| `run_graph` for Metal | ✅ **Working** | Full implementation in `run.rs:308-513` |
| `run_graph` for CUDA | ✅ **Working** | Full implementation in `run.rs:157-305` |
| `compile_kernels` for Metal | ✅ **Working** | `run.rs:132-154` |
| `compile_kernels` for CUDA | ✅ **Working** | `run.rs:99-129` |
| `codegen` | ✅ **Working** | Generates kernels from GraphTerm graphs |
| `search` function | ✅ **Working** | Uses `run_graph` internally |
| `translate_graph` | ✅ **Working** | Translates Luminal graph to IR |
| E2E test file | ✅ **Working** | 4 tests covering pipeline |
| `CompatKernel::process` Metal | ✅ **Implemented** | Full Metal kernel execution |
| `CompatKernel::process` CUDA | ✅ **Implemented** | Full CUDA kernel execution |
| `Diff::process` Metal | ✅ **Implemented** | Debugging output to file |
| `Diff::process` CUDA | ✅ **Implemented** | Debugging output to file |
| Metal conv2d tests | ✅ **Fixed** | CPU comparison pattern |
| Metal fp16 sum tests | ✅ **Fixed** | Compare against dfdx fp32 |

### Test Coverage Gaps (Updated 2025-12-12)

| Module | Has Tests? | Coverage Gap |
|--------|------------|--------------|
| `lib.rs` (CompatKernel, Diff) | ✅ | `test_compat_kernel_cuda` added |
| `run.rs` | ✅ | Runtime tests exercise `run_graph`, `compile_kernels`, `assign_buffers` |
| `extract.rs` (search) | ❌ | No unit tests for search function |
| `utils.rs` | ❌ | No tests |
| `codegen.rs` | ✅ | 6 tests |
| `translate.rs` | ✅ | 17 tests |
| `e2e_tests.rs` | ✅ | 4 tests |
| `cuda_tests.rs` | ✅ | 9 CUDA-specific tests (codegen, runtime, matmul, CompatKernel) |

### CI Coverage Gap

The CI workflow (`.github/workflows/test.yml`) does **not** run tests for `luminal_2` with Metal feature:
- Ubuntu runs `cargo test --workspace` (includes luminal_2 without GPU features)
- macOS only tests `luminal_metal`, not `luminal_2 --features metal`

**Recommendation:** Add `luminal_2` Metal tests to CI (see Phase 2.5)

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
| `examples/luminal_2_demo/src/benchmark.rs` | Added search benchmark to verify Phase 4 improvements |
| `examples/luminal_2_demo/src/main.rs` | Added `benchmark` and `search` CLI commands |

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

## Phase 5: Unify 1.0 and 2.0 Architectures

**Priority:** 🟢 Lower - Improves user experience

### 5.1 Goals

1. Single API for users
2. Automatic selection between fast (1.0) and optimal (2.0) compilation
3. Gradual migration path

### 5.2 Design

```rust
pub enum CompilationMode {
    Fast,           // Use hand-written kernels (luminal_metal/cuda)
    Optimal,        // Use search-based (luminal_2)
    TimeBudget(Duration),  // Search with timeout, fallback to fast
}

pub trait UnifiedCompiler {
    fn compile_with_mode<T: ToIdsMut>(
        &self,
        graph: &mut Graph,
        outputs: T,
        mode: CompilationMode,
    );
}
```

### 5.3 Implementation Notes

- Add feature flag `search` that enables luminal_2
- Default to `Fast` mode
- Production builds can opt into `Optimal`

---

## Phase 6: Complete Training Infrastructure

See detailed implementation in sections below. Key additions:

1. **Learning Rate Schedulers** - Cosine, warmup, one-cycle
2. **Mixed Precision** - FP16 forward, FP32 gradients with loss scaling
3. **Gradient Checkpointing** - Trade compute for memory
4. **Gradient Accumulation** - Larger effective batch sizes

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
| **4 - Expand Search** | 🟡 Medium | Medium | High | ✅ Complete |
| 5 - Unify Arch | 🟢 Lower | High | Medium | Not Started |
| 6 - Training | 🟡 Medium | Medium | High | Not Started |
| 7 - Benchmarks | 🟢 Lower | Low | Medium | Not Started |
| 8 - ROCm | 🔵 Future | Very High | Medium | Not Started |
| 9 - Distributed | 🔵 Future | Very High | High | Not Started |
| 10 - Python | 🔵 Future | Medium | High | Not Started |

**Recommended Next Steps:**
1. **Phase 5 (unify architectures)** — Create unified API for 1.0 and 2.0 backends
2. Phase 6 (training infrastructure) — Add LR schedulers, mixed precision, gradient checkpointing
3. Phase 7 (benchmarking) — Validate performance against PyTorch baselines

---

## Appendix B: Quick Verification Commands

```bash
# All Metal backend tests (205 tests) - requires macOS with Metal GPU
cd crates/luminal_metal && cargo test

# All luminal_2 Metal tests (27 tests)
cd crates/luminal_2 && cargo test --features metal

# All luminal_2 CUDA tests (requires NVIDIA GPU with CUDA 12.x)
cd crates/luminal_2 && cargo test --features cuda

# luminal_cuda tests (requires NVIDIA GPU)
cd crates/luminal_cuda && cargo test -- --test-threads=1

# All workspace tests (CPU-only crates)
cargo test --workspace

# Individual crates
cargo test -p luminal          # 89 tests
cargo test -p luminal_nn       # 31 tests
cargo test -p luminal_training # 18 tests

# Format and lint
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Metal crates (separate workspace)
cd crates/luminal_metal && cargo fmt --all && cargo clippy --all-targets -- -D warnings
cd crates/luminal_cuda && cargo fmt --all && cargo clippy --all-targets -- -D warnings
```

---

## Appendix C: File Locations Quick Reference

### luminal_2 (Search-Based Compiler)

| Component | File |
|-----------|------|
| CompatKernel (CUDA) | `crates/luminal_2/src/lib.rs:126-181` |
| CompatKernel (Metal) | `crates/luminal_2/src/lib.rs:184-292` |
| Diff (CUDA) | `crates/luminal_2/src/lib.rs:321-339` |
| Diff (Metal) | `crates/luminal_2/src/lib.rs:341-368` |
| custom_kernel | `crates/luminal_2/src/lib.rs:301-314` |
| run_graph (Metal) | `crates/luminal_2/src/run.rs:308-513` |
| run_graph (CUDA) | `crates/luminal_2/src/run.rs:157-305` |
| codegen | `crates/luminal_2/src/codegen.rs` |
| translate_graph | `crates/luminal_2/src/translate.rs` |
| search | `crates/luminal_2/src/extract.rs:357` |
| Egglog rules | `crates/luminal_2/src/code.lisp` |
| E2E tests (Metal) | `crates/luminal_2/src/e2e_tests.rs` |
| CUDA tests | `crates/luminal_2/src/cuda_tests.rs` (9 tests: codegen, runtime, matmul, CompatKernel) |
| Translation tests | `crates/luminal_2/src/translate.rs` (17 tests) |
| Codegen tests | `crates/luminal_2/src/codegen.rs` (6 tests) |
| TCMatmul (CUDA) | `crates/luminal_2/src/codegen.rs` (warp-coop 8x8) |
| TCMatmul (Metal) | `crates/luminal_2/src/codegen.rs` (simdgroup 8x8) |

### luminal_metal

| Component | File |
|-----------|------|
| test_conv2d (fp32) | `crates/luminal_metal/src/tests/fp32.rs:464-517` |
| test_conv2d (fp16) | `crates/luminal_metal/src/tests/fp16.rs:805-859` |
| test_sum (fp16) | `crates/luminal_metal/src/tests/fp16.rs:114-139` |
| test_sum2 (fp16) | `crates/luminal_metal/src/tests/fp16.rs:141-168` |
| verification tests | `crates/luminal_metal/src/verification_tests.rs` |

### Other Key Files

| Component | File |
|-----------|------|
| CI workflow | `.github/workflows/test.yml` |
| Core library | `src/` |
| NN modules | `crates/luminal_nn/src/` |
| Training | `crates/luminal_training/src/` |

---

## Appendix D: luminal_2 Demo

A working demo showcasing `luminal_2`'s CUDA capabilities is available at `examples/luminal_2_demo/`.

### Running the Demo

```bash
# CUDA backend (requires NVIDIA GPU)
cargo run -p luminal_2_demo --release --features cuda

# Metal backend (macOS only)
cargo run -p luminal_2_demo --release --features metal
```

### What the Demo Shows

| Demo | Description |
|------|-------------|
| **Demo 1: Search-Based Compilation** | Full pipeline: Luminal graph → IR translation → CUDA codegen → GPU execution with numerical verification |
| **Demo 2: Custom Kernel Injection** | Using `custom_kernel` API to inject a hand-written GELU CUDA kernel into the graph |
| **Demo 3: Matrix Multiply** | 8×8 matmul through the search pipeline (demonstrates warp-cooperative pattern) |
| **Demo 4: MLP Inference** | 2-layer network showing how complex graphs decompose into multiple fused CUDA kernels |

### Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║          luminal_2 Search-Based CUDA Compiler Demo           ║
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
use luminal_2::{custom_kernel, Kernel};

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
use luminal_2::{
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
| `examples/luminal_2_demo/Cargo.toml` | Dependencies and feature flags |
| `examples/luminal_2_demo/src/main.rs` | Demo implementation (565 lines) |
| `examples/luminal_2_demo/README.md` | Standalone documentation |
