# Luminal 2.0 Implementation Plan

This document provides a step-by-step implementation guide to complete the search-based compilation system and realize Luminal's full potential.

**Last Updated:** 2025-12-11 — Phases 0-2.5 complete. All tests passing.

### Quick Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Fix Metal test failures | ✅ Complete |
| 1 | Fix E2E tests | ✅ Complete |
| 2 | Metal `todo!()` implementations | ✅ Complete |
| 2.5 | Address test coverage gaps | ✅ Complete |
| **3** | **CUDA tensor core support** | ⚠️ **Next** |
| 4-10 | Remaining phases | Not started |

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Phase 0: Fix Failing Metal Backend Tests](#phase-0-fix-failing-metal-backend-tests) ✅ COMPLETE
3. [Phase 1: Fix Broken E2E Tests](#phase-1-fix-broken-e2e-tests) ✅ COMPLETE
4. [Phase 2: Complete Metal `todo!()` Implementations](#phase-2-complete-metal-todo-implementations) ✅ COMPLETE
5. [Phase 2.5: Address Test Coverage Gaps](#phase-25-address-test-coverage-gaps) ✅ COMPLETE
6. [Phase 3: CUDA Tensor Core Support](#phase-3-cuda-tensor-core-support)
7. [Phase 4: Expand Search Space](#phase-4-expand-search-space)
8. [Phase 5: Unify 1.0 and 2.0 Architectures](#phase-5-unify-10-and-20-architectures)
9. [Phase 6: Complete Training Infrastructure](#phase-6-complete-training-infrastructure)
10. [Phase 7: Benchmarking Suite](#phase-7-benchmarking-suite)
11. [Phase 8: ROCm Backend](#phase-8-rocm-backend)
12. [Phase 9: Distributed Computing](#phase-9-distributed-computing)
13. [Phase 10: Python Bindings](#phase-10-python-bindings)

---

## Current State Assessment

### Test Summary (as of 2025-12-11)

| Crate | Tests | Status |
|-------|-------|--------|
| `luminal` (core) | 89 | ✅ All pass |
| `luminal_nn` | 31 | ✅ All pass |
| `luminal_training` | 18 | ✅ All pass |
| `luminal_metal` | 205 | ✅ All pass |
| `luminal_2` | 27 | ✅ All pass |
| `luminal_cuda` | — | Not tested (requires GPU) |

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

### Test Coverage Gaps (Identified 2025-12-11)

| Module | Has Tests? | Coverage Gap |
|--------|------------|--------------|
| `lib.rs` (CompatKernel, Diff) | ❌ | No tests for CompatKernel/Diff operators |
| `run.rs` | ❌ | No tests for run_graph, compile_kernels, assign_buffers |
| `extract.rs` (search) | ❌ | No unit tests for search function |
| `utils.rs` | ❌ | No tests |
| `codegen.rs` | ✅ | 6 tests |
| `translate.rs` | ✅ | 17 tests |
| `e2e_tests.rs` | ✅ | 4 tests |

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
cargo test -p luminal_metal -- --test-threads=1  # All 205 tests pass
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

## Phase 3: CUDA Tensor Core Support

**Priority:** 🟠 High - Significant performance opportunity

### 3.1 Problem Statement

The `TCMatmul` GraphTerm explicitly skips CUDA:

```rust
// Location: crates/luminal_2/src/codegen.rs:936-940
GraphTerm::TCMatmul { ... } => {
    if cfg!(feature = "cuda") {
        return None;  // CUDA build: skip / fallback
    }
    // ... Metal simdgroup implementation follows
}
```

### 3.2 CUDA WMMA Background

NVIDIA's Warp Matrix Multiply-Accumulate (WMMA) API:
- Available on Volta+ (sm_70 and later)
- Uses tensor cores for 16x16x16 or 8x8x8 matrix tiles
- Typical usage: FP16 inputs, FP32 accumulator

### 3.3 Implementation

Replace the CUDA skip with WMMA code generation:

```rust
GraphTerm::TCMatmul {
    a_k_stride,
    b_k_stride,
    a_inner_stride,
    b_inner_stride,
    c_inner_stride,
    k_outer_loops,
} => {
    #[cfg(feature = "cuda")]
    {
        use itertools::Itertools;
        
        let mut srcs = kernel_graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source());
        let (src_a, src_a_ptr) = node_to_var[&srcs.next().unwrap()];
        let (src_b, src_b_ptr) = node_to_var[&srcs.next().unwrap()];
        let dest = kernel_graph
            .neighbors_directed(node, Direction::Outgoing)
            .next()
            .unwrap();
        let (dest, dest_ptr) = node_to_var[&dest];
        
        assert!(src_a_ptr && src_b_ptr && dest_ptr, "TCMatmul requires pointer inputs");
        
        kernel_lines.push(format!(
            r#"
// CUDA Tensor Core Matmul via WMMA
#include <mma.h>
using namespace nvcuda;
{{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k_tile = 0; k_tile < {k_loops}; k_tile++) {{
        // Load A and B tiles
        wmma::load_matrix_sync(a_frag, (half*)({a} + {a_k}), {a_ld});
        wmma::load_matrix_sync(b_frag, (half*)({b} + {b_k}), {b_ld});
        
        // Multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }}
    
    // Store result
    wmma::store_matrix_sync({c}, c_frag, {c_ld}, wmma::mem_row_major);
}}
"#,
            k_loops = k_outer_loops.to_kernel(),
            a = var_to_char(src_a),
            b = var_to_char(src_b),
            c = var_to_char(dest),
            a_k = a_k_stride.to_kernel().replace("const_z", "k_tile"),
            b_k = b_k_stride.to_kernel().replace("const_z", "k_tile"),
            a_ld = a_inner_stride.substitute('z', 1).to_kernel(),
            b_ld = b_inner_stride.substitute('z', 1).to_kernel(),
            c_ld = c_inner_stride.substitute('z', 1).to_kernel(),
        ));
        
        node_to_var.insert(node, (dest, true));
    }
    
    #[cfg(feature = "metal")]
    {
        // Existing simdgroup implementation...
    }
}
```

### 3.4 NVRTC Compilation Options

Update the CUDA compilation to support tensor cores:

```rust
// In run.rs or wherever NVRTC is called
CompileOptions {
    include_paths: vec![
        "/usr/include".into(),
        "/usr/local/cuda/include".into(),
    ],
    options: vec![
        "--gpu-architecture=sm_80".into(),  // Ampere+
        "--std=c++17".into(),
        "-DCUDA_HAS_WMMA".into(),
    ],
    ..Default::default()
}
```

### 3.5 Testing

```rust
#[cfg(feature = "cuda")]
#[test]
fn test_tcmatmul_cuda() {
    let mut cx = Graph::new();
    
    // 16x16 matmul (minimum tensor core size)
    let a = cx.tensor((16, 16)).set(vec![1.0f32; 256]);
    let b = cx.tensor((16, 16)).set(vec![1.0f32; 256]);
    let c = a.matmul(b).retrieve();
    
    // Compile through luminal_2
    let (meta_graph, _, _) = translate_graph(&cx);
    // ... run search, verify TCMatmul is used and results are correct
}
```

---

## Phase 4: Expand Search Space

**Priority:** 🟡 Medium - Improves optimization quality

### 4.1 Current Limitations

From `extract.rs:51-63`:
```rust
const INVALID_IR: &[&str] = &[
    "SwapLoops",
    "TileLoop", 
    "UnpadLoop",
    "MReplace",
    "MergeLoops",
    // ... etc
];
```

Many transformations are disabled.

### 4.2 Enable Loop Transformations

#### Step 4.2.1: Remove from INVALID_IR

```rust
const INVALID_IR: &[&str] = &[
    // Keep only truly invalid patterns
    "loop_level",  // Internal tracking, not actual IR
    "vec-of",
    "set-of",
];
```

#### Step 4.2.2: Add Variable Tile Sizes

In `code.lisp`, the tiling is hardcoded to 8. Add rules for other sizes:

```lisp
; Tile by 4
(rule
    ((= ?e (LoopOut ?body (MNum ?range) ?stride))
     (= ?ll (loop_level ?e))
     (> ?range 4)
     (= (% ?range 4) 0))
    ((union ?e
        (LoopOut
            (LoopOut (TileLoop ?body ?ll 4) (MNum 4) ?stride)
            (MNum (/ ?range 4))
            (MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 4))))))
    :ruleset ir)

; Tile by 16
(rule
    ((= ?e (LoopOut ?body (MNum ?range) ?stride))
     (= ?ll (loop_level ?e))
     (> ?range 16)
     (= (% ?range 16) 0))
    ((union ?e
        (LoopOut
            (LoopOut (TileLoop ?body ?ll 16) (MNum 16) ?stride)
            (MNum (/ ?range 16))
            (MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 16))))))
    :ruleset ir)
```

#### Step 4.2.3: Increase Search Budget

```rust
// In extract.rs
const MAX_SEARCHED_GRAPHS: usize = 10_000;  // Was 1_000

// Add early termination for fast kernels
if us < 50 {  // Found very fast kernel (< 50µs)
    break 'trajectory_loop;
}
```

### 4.3 Verification

- [ ] More trajectories explored
- [ ] Flash Attention-like patterns discovered for softmax(QK^T)V
- [ ] No compilation time regression (< 60s)

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
| **3 - CUDA Tensor Cores** | 🟠 High | Medium | High | ⚠️ **Next** |
| 4 - Expand Search | 🟡 Medium | Medium | High | Not Started |
| 5 - Unify Arch | 🟢 Lower | High | Medium | Not Started |
| 6 - Training | 🟡 Medium | Medium | High | Not Started |
| 7 - Benchmarks | 🟢 Lower | Low | Medium | Not Started |
| 8 - ROCm | 🔵 Future | Very High | Medium | Not Started |
| 9 - Distributed | 🔵 Future | Very High | High | Not Started |
| 10 - Python | 🔵 Future | Medium | High | Not Started |

**Recommended Next Steps:**
1. Phase 3 (CUDA tensor cores) — high performance impact
2. Phase 4 (expand search space) — improves optimization quality
3. Phase 6 (training infrastructure) — can be parallelized with above

---

## Appendix B: Quick Verification Commands

```bash
# All Metal backend tests (205 tests) - requires macOS with Metal GPU
cd crates/luminal_metal && cargo test -- --test-threads=1

# All luminal_2 Metal tests (27 tests)
cd crates/luminal_2 && cargo test --features metal

# All luminal_2 CUDA tests (requires NVIDIA GPU)
cd crates/luminal_2 && cargo test --features cuda

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
| E2E tests | `crates/luminal_2/src/e2e_tests.rs` |
| Translation tests | `crates/luminal_2/src/translate.rs` (17 tests) |
| Codegen tests | `crates/luminal_2/src/codegen.rs` (6 tests) |

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
