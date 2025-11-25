# luminal_nn

Neural network layers and utilities for Luminal.

## Overview

`luminal_nn` provides production-ready neural network components that work seamlessly with Luminal's graph-based computation model.

## Modules

### Layers

- **Linear** - Fully connected layers with optional bias
- **Conv1D/Conv2D/Conv3D** - Convolutional layers with padding, stride, and dilation
- **Embedding** - Token embedding layers

### Normalization

- **LayerNorm** - Layer normalization with optional affine transform
- **RMSNorm** - Root mean square normalization (LLaMA-style)
- **BatchNorm1d/BatchNorm2d** - Batch normalization
- **GroupNorm** - Group normalization

### Pooling

- **MaxPool1D/MaxPool2D** - Max pooling
- **AvgPool1D/AvgPool2D** - Average pooling
- **GlobalMaxPool2D/GlobalAvgPool2D** - Global pooling
- **AdaptiveAvgPool2D** - Adaptive average pooling

### Recurrent

- **LSTM** - Long Short-Term Memory
- **GRU** - Gated Recurrent Unit

### Attention & Transformers

- **SelfAttention** - Multi-head self-attention
- **TransformerEncoderBlock** - Full transformer encoder block
- **TransformerDecoderBlock** - Full transformer decoder block

### Regularization

- **Dropout** - Inverted dropout with training/inference modes

### Activation Functions

- **ReLU**, **Swish**, **Sigmoid**, **Tanh**

## Quick Start

```rust
use luminal::prelude::*;
use luminal_nn::{Linear, ReLU, LayerNorm};
use luminal::module::Module;

fn main() {
    let mut cx = Graph::new();
    
    // Build a simple MLP
    let model = (
        Linear::new(784, 256, true, &mut cx),
        ReLU,
        Linear::new(256, 10, true, &mut cx),
    );
    
    // Create input tensor
    let input = cx.tensor((32, 784)).set(vec![0.0; 32 * 784]);
    
    // Forward pass
    let output = model.forward(input).retrieve();
    
    cx.execute();
    println!("Output shape: {:?}", output.dims());
}
```

## Utilities

The `utils` module provides developer tools:

```rust
use luminal_nn::utils::{
    model_summary,      // Print model architecture
    tensor_stats,       // Debug tensor values
    check_numerics,     // Detect NaN/Inf
    validate_gradients, // Check gradient health
    save_weights,       // Save to binary file
    load_weights,       // Load from binary file
    CheckpointManager,  // Manage training checkpoints
};
```

### Model Summary

```rust
let summary = model_summary(&model, &cx);
println!("{}", summary);
// ┌─────────────────────────────────────────────────────────┐
// │                      Model Summary                       │
// ├──────────────────────────────────┬──────────────────────┤
// │ Layer                            │ Parameters           │
// ├──────────────────────────────────┼──────────────────────┤
// │ 0/weight                         │      200.70K         │
// │ 0/bias                           │          256         │
// │ 2/weight                         │        2.56K         │
// │ 2/bias                           │           10         │
// ├──────────────────────────────────┴──────────────────────┤
// │ Total parameters:                              203.53K   │
// └─────────────────────────────────────────────────────────┘
```

### Checkpointing

```rust
use luminal_nn::utils::CheckpointManager;

let mut ckpt = CheckpointManager::new("./checkpoints", 3);

for epoch in 0..100 {
    // ... training loop ...
    
    // Save checkpoint every 10 epochs
    if epoch % 10 == 0 {
        ckpt.save(&model, &cx, epoch)?;
    }
}

// Load latest checkpoint
if let Some(weights) = ckpt.load_latest()? {
    apply_weights(&model, &weights, &mut cx)?;
}
```

## Weight Initialization

All layers are created with their weights uninitialized. Set weights using:

```rust
let linear = Linear::new(128, 64, true, &mut cx);

// Set specific weights
linear.weight.set(vec![0.1; 128 * 64]);
linear.bias.unwrap().set(vec![0.0; 64]);

// Or use random initialization
use luminal::tests::random_vec;
linear.weight.set(random_vec(128 * 64));
```

## Training Mode

Layers like `Dropout` and `BatchNorm` have different behavior in training vs inference:

```rust
let mut dropout = Dropout::new(0.5, (batch_size, hidden_dim), &mut cx);

// Training
dropout.set_training(true);
dropout.resample_mask(); // Generate new random mask each forward pass

// Inference
dropout.set_training(false); // Dropout becomes identity
```

## Serialization

Models can be serialized using the `SerializeModule` trait:

```rust
use luminal::module::{param_dict, SerializeModule};

// Get all parameter node IDs
let params = param_dict(&model);

// Save weights
save_weights(&model, &cx, "model.bin")?;

// Load weights
let weights = load_weights("model.bin")?;
apply_weights(&model, &weights, &mut cx)?;
```

## License

MIT OR Apache-2.0

