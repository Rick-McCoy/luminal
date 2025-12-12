//! Benchmark for Phase 4 search space improvements
//!
//! Run with: cargo run --release --features cuda -- benchmark
//! Run search benchmark: cargo run --release --features cuda -- search

use luminal::prelude::*;
use luminal::search::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::compile_kernels,
    translate::translate_graph,
    GPUArch,
};
use rustc_hash::FxHashMap;
use std::time::Instant;

/// Benchmark result for a single run
pub struct BenchmarkResult {
    pub name: String,
    pub graph_nodes: usize,
    pub kernel_count: usize,
    pub total_kernel_lines: usize,
    pub codegen_time_ms: f64,
    pub compile_time_ms: f64,
}

impl BenchmarkResult {
    pub fn print(&self) {
        println!("  Graph nodes:    {}", self.graph_nodes);
        println!("  Kernels:        {}", self.kernel_count);
        println!("  Kernel lines:   {}", self.total_kernel_lines);
        println!("  Codegen time:   {:.2}ms", self.codegen_time_ms);
        println!("  Compile time:   {:.2}ms", self.compile_time_ms);
    }
}

/// Run a benchmark on a graph
pub fn benchmark_graph(name: &str, cx: &Graph) -> Option<BenchmarkResult> {
    let (meta_graph, _, _) = translate_graph(cx);
    let (stitched, _) = stitch_meta_graph_together(meta_graph);
    let graph_nodes = stitched.node_count();

    let codegen_start = Instant::now();
    let (kernels, _gmem_map) = codegen(stitched, GPUArch::CUDA, &cx.dyn_map)?;
    let codegen_time = codegen_start.elapsed();

    let kernel_count = kernels.node_count();

    // Count kernel lines
    let mut total_lines = 0;
    for node in kernels.node_weights() {
        total_lines += node.code.lines().count();
    }

    let compile_start = Instant::now();
    let _compiled = compile_kernels(&kernels);
    let compile_time = compile_start.elapsed();

    Some(BenchmarkResult {
        name: name.to_string(),
        graph_nodes,
        kernel_count,
        total_kernel_lines: total_lines,
        codegen_time_ms: codegen_time.as_secs_f64() * 1000.0,
        compile_time_ms: compile_time.as_secs_f64() * 1000.0,
    })
}

/// Create various test graphs for benchmarking
pub fn create_benchmark_graphs() -> Vec<(String, Graph)> {
    let mut benchmarks = Vec::new();

    // 1. Simple elementwise (should benefit from tiling)
    {
        let mut cx = Graph::new();
        let a = cx.tensor((64,)).set(vec![1.0; 64]);
        let b = cx.tensor((64,)).set(vec![2.0; 64]);
        let _c = (a + b * 2.0).retrieve();
        benchmarks.push(("elementwise_64".to_string(), cx));
    }

    // 2. Larger elementwise
    {
        let mut cx = Graph::new();
        let a = cx.tensor((256,)).set(vec![1.0; 256]);
        let b = cx.tensor((256,)).set(vec![2.0; 256]);
        let _c = (a + b * 2.0).retrieve();
        benchmarks.push(("elementwise_256".to_string(), cx));
    }

    // 3. Chained operations
    {
        let mut cx = Graph::new();
        let a = cx.tensor((64,)).set(vec![1.0; 64]);
        let b = cx.tensor((64,)).set(vec![2.0; 64]);
        let c = a + b;
        let d = c * 2.0;
        let _e = d.exp2().retrieve();
        benchmarks.push(("chain_64".to_string(), cx));
    }

    // 4. 2D operations (good for tile size exploration)
    {
        let mut cx = Graph::new();
        let a = cx.tensor((16, 16)).set(vec![1.0; 256]);
        let b = cx.tensor((16, 16)).set(vec![2.0; 256]);
        let _c = (a + b).retrieve();
        benchmarks.push(("add_16x16".to_string(), cx));
    }

    // 5. Larger 2D
    {
        let mut cx = Graph::new();
        let a = cx.tensor((32, 32)).set(vec![1.0; 1024]);
        let b = cx.tensor((32, 32)).set(vec![2.0; 1024]);
        let _c = (a + b).retrieve();
        benchmarks.push(("add_32x32".to_string(), cx));
    }

    // 6. Very large 1D (benefits from larger tile sizes)
    {
        let mut cx = Graph::new();
        let a = cx.tensor((1024,)).set(vec![1.0; 1024]);
        let b = cx.tensor((1024,)).set(vec![2.0; 1024]);
        let _c = (a + b).retrieve();
        benchmarks.push(("add_1024".to_string(), cx));
    }

    benchmarks
}

pub fn run_benchmarks() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Phase 4 Search Space Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("This benchmark measures codegen and NVRTC compile times.");
    println!("Phase 4 improvements:");
    println!("  - Variable tile sizes (4, 8, 16, 32)");
    println!("  - Increased search budget (1,000 → 10,000)");
    println!("  - Early termination for fast kernels\n");

    let benchmarks = create_benchmark_graphs();
    let mut results = Vec::new();

    for (name, cx) in &benchmarks {
        println!("Running: {}", name);
        match benchmark_graph(name, cx) {
            Some(result) => {
                result.print();
                results.push(result);
            }
            None => {
                println!("  ⚠ Codegen failed for this graph");
            }
        }
        println!();
    }

    // Summary
    println!("═══ Summary ═══\n");
    println!(
        "{:<20} {:>8} {:>10} {:>12} {:>12}",
        "Benchmark", "Kernels", "Lines", "Codegen(ms)", "Compile(ms)"
    );
    println!("{}", "-".repeat(64));
    for r in &results {
        println!(
            "{:<20} {:>8} {:>10} {:>12.2} {:>12.2}",
            r.name, r.kernel_count, r.total_kernel_lines, r.codegen_time_ms, r.compile_time_ms
        );
    }

    let total_codegen: f64 = results.iter().map(|r| r.codegen_time_ms).sum();
    let total_compile: f64 = results.iter().map(|r| r.compile_time_ms).sum();
    println!("{}", "-".repeat(64));
    println!(
        "{:<20} {:>8} {:>10} {:>12.2} {:>12.2}",
        "TOTAL", "", "", total_codegen, total_compile
    );

    println!("\n✓ Benchmark complete");
    println!("\nTo compare with previous settings:");
    println!("1. Modify MAX_SEARCHED_GRAPHS in extract.rs");
    println!("   (currently 10,000, try changing to 1,000)");
    println!("2. Comment out tile size rules in code.lisp");
    println!("3. Re-run this benchmark and compare results");
}

/// Run the search benchmark - this exercises the full Phase 4 improvements
pub fn run_search_benchmark() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Phase 4 Search Benchmark                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("This benchmark runs the search() function which:");
    println!("  1. Builds egglog search space with different tile sizes");
    println!("  2. Explores up to 10,000 candidate graphs");
    println!("  3. Benchmarks each candidate on GPU");
    println!("  4. Returns the fastest kernel\n");

    println!("Phase 4 improvements:");
    println!("  - Variable tile sizes (4, 8, 16, 32) vs just 8");
    println!("  - MAX_SEARCHED_GRAPHS: 10,000 (was 1,000)");
    println!("  - Early termination for kernels < 50µs\n");

    // Create test graphs
    let test_cases = vec![("add_32", 32), ("add_64", 64), ("add_128", 128)];

    for (name, size) in test_cases {
        println!("═══ {} (size={}) ═══\n", name, size);

        // Create graph
        let mut cx = Graph::new();
        let a = cx.tensor((size,)).set(vec![1.0; size]);
        let b = cx.tensor((size,)).set(vec![2.0; size]);
        let _c = (a + b).retrieve();

        // Translate to IR graph
        let (meta_graph, _, _) = translate_graph(&cx);
        let (stitched, _) = stitch_meta_graph_together(meta_graph);

        println!("Graph: {} nodes", stitched.node_count());

        // Create inputs for search
        let dyn_map: FxHashMap<char, usize> = FxHashMap::default();
        let inputs = make_test_inputs(&stitched, &dyn_map, &[]);

        println!("Inputs: {} tensors", inputs.len());
        println!("\nRunning search (this may take a few seconds)...");

        let start = Instant::now();
        let result = search(&stitched, 10, &inputs, GPUArch::CUDA, &dyn_map);
        let search_time = start.elapsed();

        match result {
            Some(best_graph) => {
                println!("\n✓ Search complete in {:.2}s", search_time.as_secs_f64());
                println!("  Best graph: {} nodes", best_graph.node_count());

                // Codegen the best graph
                if let Some((kernels, _)) = codegen(best_graph, GPUArch::CUDA, &dyn_map) {
                    println!("  Kernels: {}", kernels.node_count());
                    let total_lines: usize =
                        kernels.node_weights().map(|k| k.code.lines().count()).sum();
                    println!("  Total kernel lines: {}", total_lines);
                }
            }
            None => {
                println!("\n⚠ Search returned no valid graph");
            }
        }
        println!();
    }

    println!("═══ Search Benchmark Complete ═══\n");
    println!("The 'FASTEST' and 'Valids' lines printed above show:");
    println!("  - FASTEST (Xms): Best kernel execution time found");
    println!("  - Valids: X / Y: Number of valid graphs explored\n");
    println!("With Phase 4 improvements, you should see:");
    println!("  - More trajectories explored (due to tile size options)");
    println!("  - Similar or better kernel times");
}
