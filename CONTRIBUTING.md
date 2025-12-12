# Contributing to Luminal

Thank you for your interest in contributing to Luminal! This guide will help you get started.

## Project Structure

```
luminal/
├── src/                    # Core library
│   ├── graph.rs           # Computation graph
│   ├── graph_tensor.rs    # GraphTensor API
│   ├── shape/             # ShapeTracker & symbolic dimensions
│   ├── hl_ops/            # High-level operations
│   ├── op.rs              # Primitive operations
│   ├── nn/                # Neural network modules (Linear, ReLU, Transformer, etc.)
│   ├── training/          # Autograd, optimizers, loss functions
│   └── search/            # Search-based optimization (egglog, behind "search" feature)
├── crates/
│   ├── luminal_cpu/       # CPU compiler & ops
│   ├── luminal_metal/     # Metal (macOS) compiler + UnifiedMetalCompiler
│   ├── luminal_cuda/      # CUDA (NVIDIA) compiler + UnifiedCudaCompiler
│   ├── luminal_nn/        # [Deprecated] Re-exports luminal::nn
│   └── luminal_training/  # [Deprecated] Re-exports luminal::training
├── examples/              # Full model implementations
│   ├── llama/             # LLaMA 3 inference
│   ├── phi/               # Phi-3 inference
│   ├── whisper/           # Whisper speech recognition
│   └── train_math_net/    # Training example
└── demos/                 # Technical demos
```

## Development Setup

### Prerequisites

- Rust 1.78+ (`rustup update stable`)
- For Metal development: macOS with Apple Silicon or AMD GPU
- For CUDA development: NVIDIA GPU with CUDA 12.0+ installed

### Building

```bash
# Build core library and CPU backend
cargo build

# Build with Metal support (macOS only)
cd crates/luminal_metal && cargo build

# Build with CUDA support (requires CUDA)
cd crates/luminal_cuda && cargo build
```

## Testing

### Core Tests (All Platforms)

```bash
# Run all workspace tests
cargo test --workspace

# Run with verbose output
cargo test --workspace -- --nocapture
```

### Metal Tests (macOS Only)

```bash
cd crates/luminal_metal
cargo test
```

### CUDA Tests (Requires NVIDIA GPU)

```bash
cd crates/luminal_cuda
cargo test
```

### Running Examples

```bash
# CPU inference
cd examples/llama
bash setup/setup.sh  # Download model
cargo run --release

# Metal inference (macOS)
cargo run --release --features metal

# CUDA inference (NVIDIA)
cargo run --release --features cuda
```

## Code Quality

### Before Submitting a PR

1. **Format your code:**
   ```bash
   cargo fmt --all
   cd crates/luminal_metal && cargo fmt
   cd crates/luminal_cuda && cargo fmt
   ```

2. **Run clippy:**
   ```bash
   cargo clippy --workspace --all-targets -- -D warnings
   ```

3. **Ensure tests pass:**
   ```bash
   cargo test --workspace
   ```

4. **Build documentation:**
   ```bash
   cargo doc --workspace --no-deps
   ```

## Architecture Overview

### Computation Graph

Luminal uses a static computation graph. Operations don't execute immediately—they're recorded for later execution:

```rust
let mut cx = Graph::new();
let a = cx.tensor((3, 3)).set([[1., 2., 3.], ...]);
let b = cx.tensor((3, 3)).set([[4., 5., 6.], ...]);
let c = a.matmul(b).retrieve();  // Nothing executed yet!

cx.compile(MetalCompiler::default(), &mut c);  // Optimize graph
cx.execute();  // NOW computation happens
println!("{:?}", c.data());
```

### Primitive Operations

All operations reduce to 12 primitives:
- **Unary:** `Log2`, `Exp2`, `Sin`, `Sqrt`, `Recip`
- **Binary:** `Add`, `Mul`, `Mod`, `LessThan`
- **Other:** `SumReduce`, `MaxReduce`, `Contiguous`

### Compilers

Compilers transform the graph. They can:
- Replace primitive ops with optimized GPU kernels
- Fuse operations for better performance
- Handle dtype conversions

```rust
// Compiler pipeline
cx.compile(
    (
        GenericCompiler::default(),  // Generic optimizations
        MetalCompiler::<f16>::default(),  // Metal kernels + fp16
    ),
    &mut outputs,
);
```

## Adding New Features

### Adding a New NN Module

1. Create your module in `src/nn/`:

```rust
// src/nn/my_module.rs
use crate::prelude::*;

pub struct MyModule {
    pub weight: GraphTensor,
}

impl MyModule {
    pub fn new(dim: usize, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("MyModule Weight", dim),
        }
    }
}

impl Module<GraphTensor> for MyModule {
    type Output = GraphTensor;
    fn forward(&self, input: GraphTensor) -> GraphTensor {
        // Implementation
    }
}

impl SerializeModule for MyModule {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
    }
}
```

2. Export it from `src/nn/mod.rs`
3. Add tests comparing against PyTorch/dfdx

### Adding a New Optimizer

See `src/training/optimizer.rs` for the SGD implementation pattern.

### Adding a Backend Optimization

Each backend has compiler passes in their respective crate. Look at:
- `crates/luminal_metal/src/elementwise_fusion.rs`
- `crates/luminal_cuda/src/matmul.rs`

## Testing Guidelines

- All new ops should have tests comparing against `dfdx` or PyTorch
- Use `assert_close` for floating-point comparisons (1e-3 tolerance)
- Use `assert_close_precision` when tighter/looser tolerance is needed
- Test both static and dynamic shapes when applicable

```rust
#[test]
fn test_my_op() {
    let mut cx = Graph::new();
    let a_data = random_vec(6);
    let a = cx.tensor((2, 3)).set(a_data.clone());
    let b = a.my_op().retrieve();
    cx.execute();

    // Compare against dfdx
    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
    let d_b = d_a.my_op();

    assert_close(&b.data(), &d_b.as_vec());
}
```

## Getting Help

- Open an issue for bugs or feature requests
- Join the [Discord](https://discord.gg/APjuwHAbGy) for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under MIT OR Apache-2.0.

