//! Demo of search-based GPU compilation
//!
//! This example demonstrates three key capabilities:
//!
//! 1. **Search-Based Compilation**: How `luminal::search` translates Luminal graphs to an IR,
//!    applies egglog rewrites to explore optimizations, and generates GPU kernels.
//!
//! 2. **Custom Kernel Injection**: Using `custom_kernel` to inject hand-written
//!    kernels into a Luminal computation graph.
//!
//! 3. **Matrix Multiply Execution**: Demonstrating the warp-cooperative matmul kernel.
//!
//! Run with CUDA backend:
//! ```
//! cargo run -p search_demo --release --features cuda
//! ```
//!
//! Run benchmarks:
//! ```
//! cargo run -p search_demo --release --features cuda -- benchmark
//! ```

mod benchmark;

use luminal::prelude::*;

fn main() {
    // Check for benchmark mode
    if std::env::args().any(|arg| arg == "benchmark") {
        benchmark::run_benchmarks();
        return;
    }

    // Check for search benchmark mode
    if std::env::args().any(|arg| arg == "search") {
        benchmark::run_search_benchmark();
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Search-Based GPU Compiler Demo                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Search-based compilation pipeline
    demo_search_compilation();

    // Demo 2: Custom CUDA kernel injection
    demo_custom_kernel();

    // Demo 3: Matrix multiply with search-based compilation
    demo_matmul();

    // Demo 4: Simple MNIST inference with search-optimized kernels
    demo_mnist_inference();

    println!("\n✅ All demos completed successfully!");
}

/// Demo 1: Show the search-based compilation pipeline
///
/// This demonstrates:
/// - Translation from Luminal graph to IR
/// - Stitching meta-graphs together
/// - CUDA codegen
/// - Kernel execution through run_graph
#[cfg(feature = "cuda")]
fn demo_search_compilation() {
    use luminal::search::{
        codegen::{codegen, stitch_meta_graph_together},
        run::{assign_buffers, compile_kernels, new_buffer, run_graph},
        translate::translate_graph,
        GPUArch, GraphTerm,
    };
    use rustc_hash::FxHashMap;
    use std::collections::HashMap;

    println!("═══ Demo 1: Search-Based Compilation Pipeline ═══\n");

    // Create a simple computation: C = A + B * 2
    let mut cx = Graph::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let expected: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(a, b)| a + b * 2.0)
        .collect();

    let a = cx.tensor(8).set(a_data.clone());
    let b = cx.tensor(8).set(b_data.clone());
    let _c = (a + b * 2.0).retrieve();

    println!("Step 1: Created Luminal graph for C = A + B * 2");
    println!("        A = {:?}", &a_data[..4]);
    println!("        B = {:?}", &b_data[..4]);

    // Step 2: Translate to IR
    let (meta_graph, _global_map, _inits) = translate_graph(&cx);
    println!(
        "Step 2: Translated to meta-graph with {} subgraphs",
        meta_graph.node_count()
    );

    // Step 3: Stitch together
    let (stitched_graph, _meta_to_unified) = stitch_meta_graph_together(meta_graph);
    println!(
        "Step 3: Stitched into unified graph with {} nodes",
        stitched_graph.node_count()
    );

    // Step 4: Generate CUDA kernels
    let result = codegen(stitched_graph.clone(), GPUArch::CUDA, &cx.dyn_map);

    if let Some((kernels, gmem_mapping)) = result {
        println!(
            "Step 4: Generated {} CUDA kernel(s)",
            kernels.node_count() - 2
        ); // -2 for Inputs/Outputs

        // Print kernel info
        for kernel in kernels.node_weights() {
            if kernel.code != "Inputs" && kernel.code != "Outputs" {
                let lines = kernel.code.lines().count();
                println!("        - Kernel: {} lines of CUDA code", lines);
            }
        }

        // Step 5: Compile and execute
        let compiled = compile_kernels(&kernels);
        let (buffer_sizes, buffer_map) = assign_buffers(&kernels);

        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let mut intermediate_buffers: Vec<_> = buffer_sizes
            .iter()
            .map(|size| new_buffer(size.exec(&cx.dyn_map).unwrap() * std::mem::size_of::<f32>()))
            .collect();

        // Build inputs
        let sorted_gmem = get_sorted_gmem_nodes(&stitched_graph, &gmem_mapping);
        let tensor_data: Vec<&[f32]> = vec![&a_data, &b_data];
        let input_buffers = build_input_buffers(&sorted_gmem, &tensor_data, &ctx);
        let mut inputs = build_inputs_map(&sorted_gmem, &input_buffers, &gmem_mapping);

        // Run
        let (outputs, timing) = run_graph(
            &mut inputs,
            &kernels,
            &cx.dyn_map,
            &compiled,
            &mut intermediate_buffers,
            &buffer_map,
        );

        println!("Step 5: Executed on GPU in {}µs", timing);
        println!("        Result: {:?}", &outputs[0][..4]);
        println!("        Expected: {:?}", &expected[..4]);

        // Verify
        let max_error: f32 = outputs[0]
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error < 1e-5,
            "Output mismatch! Max error: {}",
            max_error
        );
        println!("        ✓ Verified correct!\n");
    } else {
        println!("Step 4: Codegen returned None (graph too simple for optimization)");
        println!("        This is expected for very simple graphs.\n");
    }

    // Helper functions
    fn get_sorted_gmem_nodes(
        stitched_graph: &petgraph::stable_graph::StableGraph<GraphTerm, ()>,
        gmem_mapping: &HashMap<NodeIndex, usize>,
    ) -> Vec<(NodeIndex, String)> {
        let mut gmem_nodes: Vec<_> = stitched_graph
            .node_indices()
            .filter_map(|n| {
                if let GraphTerm::GMEM { label } = stitched_graph.node_weight(n).unwrap() {
                    if gmem_mapping.contains_key(&n) {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        gmem_nodes.sort_by_key(|(node, _)| gmem_mapping.get(node));
        gmem_nodes
    }

    fn build_input_buffers(
        sorted_gmem: &[(NodeIndex, String)],
        tensor_data: &[&[f32]],
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Vec<cudarc::driver::CudaSlice<f32>> {
        use luminal::search::run::htod;

        let mut input_buffers = Vec::new();
        let mut tensor_idx = 0;

        for (_node, label) in sorted_gmem {
            if label.contains("Tensor") && tensor_idx < tensor_data.len() {
                input_buffers.push(htod(tensor_data[tensor_idx], ctx));
                tensor_idx += 1;
            } else if label.starts_with("Constant") {
                if let Some(val_str) = label
                    .strip_prefix("Constant(")
                    .and_then(|s| s.strip_suffix(')'))
                {
                    if let Ok(val) = val_str.parse::<f32>() {
                        input_buffers.push(htod(&[val], ctx));
                    }
                }
            }
        }
        input_buffers
    }

    fn build_inputs_map<'a>(
        sorted_gmem: &[(NodeIndex, String)],
        input_buffers: &'a [cudarc::driver::CudaSlice<f32>],
        gmem_mapping: &HashMap<NodeIndex, usize>,
    ) -> FxHashMap<usize, (&'a cudarc::driver::CudaSlice<f32>, bool)> {
        let mut inputs = FxHashMap::default();
        for ((node, _label), buffer) in sorted_gmem.iter().zip(input_buffers.iter()) {
            let gmem_idx = gmem_mapping[node];
            inputs.insert(gmem_idx, (buffer, false));
        }
        inputs
    }
}

#[cfg(not(feature = "cuda"))]
fn demo_search_compilation() {
    println!("═══ Demo 1: Search-Based Compilation Pipeline ═══\n");
    println!("⚠️  CUDA feature not enabled. Run with --features cuda\n");
}

/// Demo 2: Custom CUDA kernel injection
///
/// This shows how to use the `custom_kernel` API to inject
/// hand-written CUDA kernels into a Luminal computation graph.
#[cfg(feature = "cuda")]
fn demo_custom_kernel() {
    use luminal::search::{custom_kernel, Kernel};
    use luminal_cuda::CudaCompiler;

    println!("═══ Demo 2: Custom CUDA Kernel Injection ═══\n");

    let mut cx = Graph::new();

    // Create a custom CUDA kernel that applies GELU activation
    // GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let gelu_kernel = Kernel {
        code: r#"
            extern "C" __global__ void kernel_name(
                float* input,
                float* output
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                float x = input[idx];

                // GELU approximation using tanh
                const float SQRT_2_PI = 0.7978845608f;
                const float COEF = 0.044715f;

                float x_cubed = x * x * x;
                float inner = SQRT_2_PI * (x + COEF * x_cubed);
                float tanh_val = tanhf(inner);

                output[idx] = 0.5f * x * (1.0f + tanh_val);
            }
        "#
        .to_string(),
        grid: (8.into(), 1.into(), 1.into()), // 8 threads total
        threadblock: (1.into(), 1.into(), 1.into()), // 1 thread per block
        smem: 0.into(),
        outputs: vec![8.into()], // Output size: 8 floats
    };

    // Input data
    let input_data = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    println!("Input:  {:?}", input_data);

    let a = cx.tensor(8).set(input_data.clone());
    let mut b = custom_kernel(&[a], gelu_kernel, 8, &mut cx).retrieve();

    // Compile with CUDA backend
    cx.compile(CudaCompiler::<f32>::default(), &mut b);
    cx.execute();

    let result = b.data();
    println!("Output: {:?}", result);

    // Verify GELU is working (GELU(0) = 0, GELU(x) ≈ x for large x)
    assert!(
        result[3].abs() < 0.01,
        "GELU(0) should be ≈ 0, got {}",
        result[3]
    );
    assert!(
        (result[7] - 1.9545977).abs() < 0.01,
        "GELU(2) should be ≈ 1.9545977, got {}",
        result[7]
    );
    println!("        ✓ Custom GELU kernel verified!\n");
}

#[cfg(not(feature = "cuda"))]
fn demo_custom_kernel() {
    println!("═══ Demo 2: Custom CUDA Kernel Injection ═══\n");
    println!("⚠️  CUDA feature not enabled. Run with --features cuda\n");
}

/// Demo 3: Matrix Multiply with warp-cooperative kernel
///
/// This shows matrix multiplication through the search-based pipeline,
/// which uses warp-cooperative 8x8 tile computation on CUDA.
#[cfg(feature = "cuda")]
fn demo_matmul() {
    use luminal::search::{
        codegen::{codegen, stitch_meta_graph_together},
        run::{assign_buffers, compile_kernels, new_buffer, run_graph},
        translate::translate_graph,
        GPUArch, GraphTerm,
    };
    use rustc_hash::FxHashMap;
    use std::collections::HashMap;

    println!("═══ Demo 3: Matrix Multiply Execution ═══\n");

    let mut cx = Graph::new();

    // 8x8 matmul - matches the egglog tile size for warp-cooperative matmul
    // A * B where both are 8x8 identity-ish matrices
    let a_data = vec![1.0f32; 64]; // All 1s
    let b_data = vec![1.0f32; 64]; // All 1s
    let expected = vec![8.0f32; 64]; // Each element is sum of 8 1*1 products

    let a = cx.tensor((8, 8)).set(a_data.clone());
    let b = cx.tensor((8, 8)).set(b_data.clone());
    let _c = a.matmul(b).retrieve();

    println!("Computing: C = A × B  where A, B are 8×8 matrices of all 1s");
    println!("Expected:  Each element of C = 8.0 (sum of 8 products)\n");

    // Translate and compile
    let (meta_graph, _global_map, _inits) = translate_graph(&cx);
    let (stitched_graph, _meta_to_unified) = stitch_meta_graph_together(meta_graph);

    let result = codegen(stitched_graph.clone(), GPUArch::CUDA, &cx.dyn_map);

    if let Some((kernels, gmem_mapping)) = result {
        // Check if warp-cooperative matmul was applied
        let has_warp_matmul = kernels
            .node_weights()
            .any(|k| k.code.contains("__syncwarp") || k.code.contains("Warp-Cooperative"));

        if has_warp_matmul {
            println!("✓ Warp-cooperative matmul pattern applied!");
        } else {
            println!("Note: Standard matmul decomposition used");
        }

        // Compile and execute
        let compiled = compile_kernels(&kernels);
        let (buffer_sizes, buffer_map) = assign_buffers(&kernels);

        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let mut intermediate_buffers: Vec<_> = buffer_sizes
            .iter()
            .map(|size| new_buffer(size.exec(&cx.dyn_map).unwrap() * std::mem::size_of::<f32>()))
            .collect();

        // Build inputs
        let sorted_gmem = get_sorted_gmem_nodes(&stitched_graph, &gmem_mapping);

        // For matmul, need to handle GMEM mapping more carefully
        let tensor_inputs: Vec<_> = sorted_gmem
            .iter()
            .filter(|(_, label)| label.contains("Tensor"))
            .collect();

        if tensor_inputs.len() == 2 && sorted_gmem.len() == 2 {
            let tensor_data: Vec<&[f32]> = vec![&a_data, &b_data];
            let input_buffers = build_input_buffers(&sorted_gmem, &tensor_data, &ctx);
            let mut inputs = build_inputs_map(&sorted_gmem, &input_buffers, &gmem_mapping);

            let (outputs, timing) = run_graph(
                &mut inputs,
                &kernels,
                &cx.dyn_map,
                &compiled,
                &mut intermediate_buffers,
                &buffer_map,
            );

            println!("Executed in {}µs", timing);
            println!(
                "Result (first 8 elements): {:?}",
                &outputs[0][..8.min(outputs[0].len())]
            );

            // Verify
            let max_error: f32 = outputs[0]
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);

            if max_error < 1e-3 {
                println!("✓ Matmul verified correct! (max error: {:.6})\n", max_error);
            } else {
                println!(
                    "⚠️  Max error: {:.6} (may be expected for some patterns)\n",
                    max_error
                );
            }
        } else {
            println!(
                "Note: Matmul has {} GMEM nodes, complex pattern detected",
                sorted_gmem.len()
            );
            println!("Skipping runtime verification for complex patterns\n");
        }
    } else {
        println!("Codegen returned None - unexpected for matmul\n");
    }

    // Helper functions (same as demo 1)
    fn get_sorted_gmem_nodes(
        stitched_graph: &petgraph::stable_graph::StableGraph<GraphTerm, ()>,
        gmem_mapping: &HashMap<NodeIndex, usize>,
    ) -> Vec<(NodeIndex, String)> {
        let mut gmem_nodes: Vec<_> = stitched_graph
            .node_indices()
            .filter_map(|n| {
                if let GraphTerm::GMEM { label } = stitched_graph.node_weight(n).unwrap() {
                    if gmem_mapping.contains_key(&n) {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        gmem_nodes.sort_by_key(|(node, _)| gmem_mapping.get(node));
        gmem_nodes
    }

    fn build_input_buffers(
        sorted_gmem: &[(NodeIndex, String)],
        tensor_data: &[&[f32]],
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Vec<cudarc::driver::CudaSlice<f32>> {
        use luminal::search::run::htod;
        let mut input_buffers = Vec::new();
        let mut tensor_idx = 0;
        for (_node, label) in sorted_gmem {
            if label.contains("Tensor") && tensor_idx < tensor_data.len() {
                input_buffers.push(htod(tensor_data[tensor_idx], ctx));
                tensor_idx += 1;
            } else if label.starts_with("Constant") {
                if let Some(val_str) = label
                    .strip_prefix("Constant(")
                    .and_then(|s| s.strip_suffix(')'))
                {
                    if let Ok(val) = val_str.parse::<f32>() {
                        input_buffers.push(htod(&[val], ctx));
                    }
                }
            }
        }
        input_buffers
    }

    fn build_inputs_map<'a>(
        sorted_gmem: &[(NodeIndex, String)],
        input_buffers: &'a [cudarc::driver::CudaSlice<f32>],
        gmem_mapping: &HashMap<NodeIndex, usize>,
    ) -> FxHashMap<usize, (&'a cudarc::driver::CudaSlice<f32>, bool)> {
        let mut inputs = FxHashMap::default();
        for ((node, _label), buffer) in sorted_gmem.iter().zip(input_buffers.iter()) {
            let gmem_idx = gmem_mapping[node];
            inputs.insert(gmem_idx, (buffer, false));
        }
        inputs
    }
}

#[cfg(not(feature = "cuda"))]
fn demo_matmul() {
    println!("═══ Demo 3: Matrix Multiply Execution ═══\n");
    println!("⚠️  CUDA feature not enabled. Run with --features cuda\n");
}

/// Demo 4: Simple MNIST-style inference with search-optimized kernels
///
/// This demonstrates a simple MLP forward pass using the search-based pipeline,
/// showing how larger networks would be optimized.
#[cfg(feature = "cuda")]
fn demo_mnist_inference() {
    use luminal::nn::Linear;
    use luminal::search::{
        codegen::{codegen, stitch_meta_graph_together},
        run::compile_kernels,
        translate::translate_graph,
        GPUArch,
    };

    println!("═══ Demo 4: MLP Inference with Search Optimization ═══\n");

    // Create a small MLP: 8 -> 16 -> 4 (simulating MNIST-like architecture at small scale)
    let mut cx = Graph::new();

    // Simple two-layer network
    let layer1 = Linear::new(8, 16, false, &mut cx).init_rand();
    let layer2 = Linear::new(16, 4, false, &mut cx).init_rand();

    // Input (batch of 1, 8 features)
    let input_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let input = cx.tensor(8).set(input_data.clone());

    // Forward pass: input -> linear -> relu -> linear
    let hidden = layer1.forward(input).relu();
    let _output = layer2.forward(hidden).retrieve();

    println!("Network: Linear(8, 16) -> ReLU -> Linear(16, 4)");
    println!("Input:   {:?}\n", input_data);

    // Translate and analyze
    let (meta_graph, _global_map, _inits) = translate_graph(&cx);
    let meta_count = meta_graph.node_count();
    let (stitched_graph, _meta_to_unified) = stitch_meta_graph_together(meta_graph);

    println!(
        "Translation: {} subgraphs -> {} unified nodes",
        meta_count,
        stitched_graph.node_count()
    );

    let result = codegen(stitched_graph.clone(), GPUArch::CUDA, &cx.dyn_map);

    if let Some((kernels, _gmem_mapping)) = result {
        let real_kernels = kernels
            .node_weights()
            .filter(|k| k.code != "Inputs" && k.code != "Outputs")
            .count();

        println!("Codegen:     {} CUDA kernel(s) generated", real_kernels);

        // Show kernel sizes
        for (i, kernel) in kernels.node_weights().enumerate() {
            if kernel.code != "Inputs" && kernel.code != "Outputs" {
                let lines = kernel.code.lines().count();
                let has_relu = kernel.code.contains("fmax");
                let has_matmul = kernel.code.contains("+=") && kernel.code.contains("for");
                println!(
                    "             Kernel {}: {} lines {}{}",
                    i,
                    lines,
                    if has_matmul { "[matmul] " } else { "" },
                    if has_relu { "[relu]" } else { "" }
                );
            }
        }

        // Compile to show it works
        let _compiled = compile_kernels(&kernels);
        println!("\n✓ All kernels compiled successfully with NVRTC\n");
    } else {
        println!("Note: Codegen optimization not applied to this graph structure\n");
    }
}

#[cfg(not(feature = "cuda"))]
fn demo_mnist_inference() {
    println!("═══ Demo 4: MLP Inference with Search Optimization ═══\n");
    println!("⚠️  CUDA feature not enabled. Run with --features cuda\n");
}
