# Demos

This folder contains standalone demonstrations of Luminal's experimental features.

## matmul (HackerNews Demo)

Demonstrates the search-based kernel generation system (luminal_2). This demo searches for optimal GPU kernel implementations at compile time.

### Prerequisites

- **macOS (Metal):** Apple Silicon or AMD GPU
- **Linux (CUDA):** NVIDIA GPU with CUDA 12.0+

### Running

```bash
cd demos/matmul

# macOS with Metal
cargo run --release --features metal

# Linux with CUDA
cargo run --release --features cuda
```

This will:
1. Build a small FFN computation graph (similar to LLaMA's MLP block)
2. Search for optimal kernel implementations using e-graphs
3. Generate and compile GPU kernels
4. Run the computation

## flash_attention

Experimental e-graph rewriting for Flash Attention pattern discovery. This is research code for the luminal_2 search system.

```bash
cd demos/flash_attention
cargo run --release
```

---

> **Note:** These demos use experimental features from `crates/luminal_2` and may require nightly Rust features or specific hardware.

