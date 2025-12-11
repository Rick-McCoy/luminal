# Contributor Guide

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed contribution guidelines.

## Quick Reference

### Structure
- Core library: `src/` (graph, GraphTensor API, shape tracker, primitive ops)
- GPU backends: `crates/luminal_metal/` and `crates/luminal_cuda/`
- NN modules: `crates/luminal_nn/`
- Training: `crates/luminal_training/` (autograd, optimizers, loss functions)

### Testing Commands

```bash
# Core tests (all platforms)
cargo test --workspace

# Metal tests (macOS only)
cd crates/luminal_metal && cargo test

# CUDA tests (requires NVIDIA GPU)
cd crates/luminal_cuda && cargo test
```

### PR Checklist
- [ ] `cargo fmt --all` (also in `crates/luminal_metal` and `crates/luminal_cuda`)
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] New features have tests comparing against dfdx/PyTorch
