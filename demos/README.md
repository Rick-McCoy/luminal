# Demos

This folder contains standalone demonstrations of Luminal's features.

## matmul (HackerNews Demo)

Demonstrates the search-based kernel generation system. This demo searches for optimal GPU kernel implementations at compile time using equality saturation (egglog).

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

Experimental e-graph rewriting for Flash Attention pattern discovery.

```bash
cd demos/flash_attention
cargo run --release
```

---

> **Note:** These demos use the `search` feature from the core `luminal` crate and may require specific hardware.

