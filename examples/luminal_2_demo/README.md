# luminal_2 Demo

This demo showcases the `luminal_2` search-based CUDA compiler.

## What is luminal_2?

`luminal_2` is an experimental search-based kernel compiler that:

1. **Translates** Luminal computation graphs to an intermediate representation (IR)
2. **Optimizes** using egglog rewrites to explore the optimization space
3. **Generates** optimized CUDA (or Metal) kernels
4. **Executes** them directly on the GPU

Unlike the standard `CudaCompiler` which uses hand-tuned kernels, `luminal_2` discovers optimizations through equality saturation search.

## Demos

### Demo 1: Search-Based Compilation Pipeline

Shows the full pipeline from Luminal graph → IR translation → kernel codegen → GPU execution.

### Demo 2: Custom CUDA Kernel Injection

Demonstrates the `custom_kernel` API for injecting hand-written CUDA kernels (e.g., a custom GELU activation) into a Luminal computation graph.

### Demo 3: Matrix Multiply

Shows matrix multiplication through the search-based pipeline, which can use warp-cooperative 8×8 tile computation on CUDA.

### Demo 4: MLP Inference

Demonstrates a simple MLP forward pass and shows how the search optimizer generates multiple fused kernels.

## Running

```bash
# Run with CUDA backend
cargo run -p luminal_2_demo --release --features cuda

# Run with Metal backend (macOS only)
cargo run -p luminal_2_demo --release --features metal
```

## Example Output

```
╔══════════════════════════════════════════════════════════════╗
║          luminal_2 Search-Based CUDA Compiler Demo           ║
╚══════════════════════════════════════════════════════════════╝

═══ Demo 1: Search-Based Compilation Pipeline ═══

Step 1: Created Luminal graph for C = A + B * 2
        A = [1.0, 2.0, 3.0, 4.0]
        B = [10.0, 20.0, 30.0, 40.0]
Step 2: Translated to meta-graph with 1 subgraphs
Step 3: Stitched into unified graph with 11 nodes
Step 4: Generated 2 CUDA kernel(s)
        - Kernel: 17 lines of CUDA code
        - Kernel: 16 lines of CUDA code
Step 5: Executed on GPU in 79µs
        Result: [21.0, 42.0, 63.0, 84.0]
        Expected: [21.0, 42.0, 63.0, 84.0]
        ✓ Verified correct!

═══ Demo 2: Custom CUDA Kernel Injection ═══

Input:  [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
Output: [-0.045, -0.159, -0.154, 0.0, 0.346, 0.841, 1.400, 1.955]
        ✓ Custom GELU kernel verified!

...
```

## Key APIs

### `custom_kernel` - Inject custom CUDA kernels

```rust
use luminal_2::{custom_kernel, Kernel};

let kernel = Kernel {
    code: r#"
        extern "C" __global__ void kernel_name(float* in, float* out) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            out[idx] = in[idx] * 2.0f;
        }
    "#.to_string(),
    grid: (8.into(), 1.into(), 1.into()),
    threadblock: (1.into(), 1.into(), 1.into()),
    smem: 0.into(),
    outputs: vec![8.into()],
};

let input = cx.tensor(8).set(vec![1.0f32; 8]);
let output = custom_kernel(&[input], kernel, 8, &mut cx);
```

### `translate_graph` / `codegen` - Search-based compilation

```rust
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    translate::translate_graph,
    GPUArch,
};

// Translate Luminal graph to IR
let (meta_graph, _, _) = translate_graph(&cx);
let (stitched, _) = stitch_meta_graph_together(meta_graph);

// Generate CUDA kernels
let (kernels, gmem_map) = codegen(stitched, GPUArch::CUDA, &cx.dyn_map).unwrap();
```

## Requirements

- CUDA toolkit (12.x recommended)
- NVIDIA GPU with compute capability 7.5+ (Turing or newer)
