# Bug Fix: Metal Training Not Converging

## Problem
The `train_math_net` example did not converge when run with the Metal backend (`--features metal`). The training loop ran indefinitely with loss and accuracy remaining constant, indicating that weights were not being updated.

## Root Cause
When GPU backends (Metal/CUDA) compile the graph, they insert `CopyToDevice` nodes between CPU `Function` nodes (where weights are loaded) and GPU operations. The graph structure becomes:

```
Weight (Function) -> MetalCopyToDevice -> MatMul (GPU)
```

During training:
1. `transfer_data_same_graph(&new_weights, &weights, &mut cx)` transfers updated weight data to the CPU `Function` nodes
2. However, `MetalCopyToDevice` nodes had their GPU buffer outputs **cached** from the previous iteration
3. During `cx.execute()`, cached nodes are skipped (optimization to avoid recomputation)
4. Result: GPU operations continued using **stale weight buffers**, never seeing the updated weights

## Solution
Modified `transfer_data_same_graph()` in `src/module.rs` to clear cached outputs of immediate children of destination nodes after transferring data. This forces nodes like `MetalCopyToDevice` to re-execute and copy the updated weights to GPU.

### Key Change
```rust
// Clear cached outputs of immediate children of destination nodes
// This ensures that nodes like MetalCopyToDevice re-execute to pick up new data
for dest in dests.iter() {
    for child in graph.graph.neighbors_directed(*dest, petgraph::Direction::Outgoing).collect::<Vec<_>>() {
        let mut output_num = 0;
        while graph.tensors.remove(&(child, output_num)).is_some() {
            output_num += 1;
        }
    }
}
```

## Additional Fixes
While investigating, two minor bugs were also fixed:

### 1. Type Casting in Metal Binary Operations
**File**: `crates/luminal_metal/src/prim.rs` (line ~360 in `metal_binary_op!` macro)

**Issue**: Hardcoded `0.0h` literal (f16) instead of properly casting `0.0` to the generic type `T`.

**Fix**: Changed to `({type_name})0.0` to work with both `f32` and `f16`.

### 2. Pattern Matching in MetalSubtractionCompiler
**File**: `crates/luminal_metal/src/binary.rs` (line ~34)

**Issue**: Used `unary::<MetalAdd<T>>` instead of `binary::<MetalAdd<T>>` when pattern matching for subtraction optimization `a + (b * -1)`.

**Fix**: Corrected to `binary::<MetalAdd<T>>`.

## Verification
After the fix, `train_math_net` converges successfully:
- Loss: 0.73 → 0.02
- Accuracy: 20% → 99.5%
- Finishes in ~35,000 iterations (17.59s on Apple Silicon)

## Files Modified
- `src/module.rs` - Core fix for cache invalidation
- `crates/luminal_metal/src/prim.rs` - Type casting fix
- `crates/luminal_metal/src/binary.rs` - Pattern matching fix
- `crates/luminal_cuda/src/prim.rs` - Consistency (no functional change)

## Impact
This fix resolves training convergence issues for:
- Metal backend (macOS GPU)
- CUDA backend (should have same issue)
- Any workflow using `transfer_data_same_graph()` with GPU compilation

## Testing
Tested with:
```bash
cd examples/train_math_net
cargo run --release --features metal
```

The model now successfully learns to add 4-bit numbers, reaching 99.5% accuracy.


