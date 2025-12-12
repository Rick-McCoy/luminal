//! Gradient checkpointing for memory-efficient training.
//!
//! Gradient checkpointing (also known as activation checkpointing or rematerialization)
//! trades compute for memory by recomputing intermediate activations during the backward
//! pass instead of storing them.
//!
//! This is essential for training very large models that would otherwise run out of memory.

use crate::prelude::*;

/// Configuration for gradient checkpointing.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Number of layers to group together in a checkpoint segment.
    /// Smaller values = less memory, more recomputation.
    /// Larger values = more memory, less recomputation.
    pub checkpoint_every_n_layers: usize,

    /// Whether to checkpoint the first layer (usually not necessary).
    pub checkpoint_first_layer: bool,

    /// Whether to checkpoint the last layer (usually not necessary).
    pub checkpoint_last_layer: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_every_n_layers: 1,
            checkpoint_first_layer: false,
            checkpoint_last_layer: false,
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the checkpoint interval.
    pub fn every_n_layers(mut self, n: usize) -> Self {
        self.checkpoint_every_n_layers = n;
        self
    }

    /// Whether to checkpoint the first layer.
    pub fn checkpoint_first(mut self, checkpoint: bool) -> Self {
        self.checkpoint_first_layer = checkpoint;
        self
    }

    /// Whether to checkpoint the last layer.
    pub fn checkpoint_last(mut self, checkpoint: bool) -> Self {
        self.checkpoint_last_layer = checkpoint;
        self
    }
}

/// Checkpoint segment for tracking activation boundaries.
///
/// This is used internally to mark where recomputation should occur.
#[derive(Clone, Debug)]
pub struct CheckpointSegment {
    /// Start node of the segment (input activation).
    pub start: NodeIndex,
    /// End node of the segment (output activation).
    pub end: NodeIndex,
    /// Whether this segment should be recomputed during backward.
    pub recompute: bool,
}

/// Manages checkpoint segments for a model.
///
/// This struct tracks which parts of the computation graph should be
/// recomputed during the backward pass.
#[derive(Clone, Debug, Default)]
pub struct CheckpointManager {
    /// List of checkpoint segments.
    pub segments: Vec<CheckpointSegment>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a checkpoint segment.
    pub fn add_segment(&mut self, start: NodeIndex, end: NodeIndex, recompute: bool) {
        self.segments.push(CheckpointSegment {
            start,
            end,
            recompute,
        });
    }

    /// Clear all segments (for next forward pass).
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Get segments that should be recomputed.
    pub fn segments_to_recompute(&self) -> impl Iterator<Item = &CheckpointSegment> {
        self.segments.iter().filter(|s| s.recompute)
    }
}

/// Estimate memory savings from checkpointing.
///
/// # Arguments
/// - `num_layers`: Total number of layers
/// - `activation_size_per_layer`: Memory per layer's activations (in bytes)
/// - `checkpoint_interval`: Checkpoint every N layers
///
/// # Returns
/// Tuple of (memory_without_checkpoint, memory_with_checkpoint, savings_ratio)
pub fn estimate_memory_savings(
    num_layers: usize,
    activation_size_per_layer: usize,
    checkpoint_interval: usize,
) -> (usize, usize, f32) {
    // Without checkpointing: store all activations
    let memory_without = num_layers * activation_size_per_layer;

    // With checkpointing: store only checkpoint activations + current segment
    let num_checkpoints = num_layers / checkpoint_interval;
    let memory_with = (num_checkpoints + checkpoint_interval) * activation_size_per_layer;

    let savings = 1.0 - (memory_with as f32 / memory_without as f32);

    (memory_without, memory_with, savings)
}

/// Mark a tensor as a checkpoint boundary.
///
/// When autograd encounters this marker during the backward pass, it will
/// recompute the value from the checkpoint rather than using stored activations.
///
/// # Example
/// ```ignore
/// let x = layer1.forward(input);
/// let x = checkpoint(x); // Mark as checkpoint
/// let x = layer2.forward(x);
/// let x = layer3.forward(x);
/// let x = checkpoint(x); // Another checkpoint
/// let output = layer4.forward(x);
/// ```
///
/// During backward pass, activations between checkpoints will be recomputed.
pub fn checkpoint(tensor: GraphTensor) -> GraphTensor {
    // For now, this is a pass-through marker.
    // Full implementation would require modifications to the Autograd compiler to:
    // 1. Detect checkpoint markers during backward graph construction
    // 2. Insert recomputation subgraphs for activations between checkpoints
    // 3. Delete intermediate activation storage after the checkpoint
    //
    // The activation is marked via contiguous() which forces materialization
    // at this point, creating a natural checkpoint boundary.
    tensor.contiguous()
}

/// Apply checkpointing to a sequence of forward operations.
///
/// This is a convenience function for checkpointing between layers.
///
/// # Arguments
/// - `input`: Input tensor
/// - `layers`: Vector of functions that each perform a forward pass
/// - `config`: Checkpointing configuration
///
/// # Returns
/// Output tensor after all layers, with checkpoints inserted according to config
pub fn checkpoint_layers<F>(
    input: GraphTensor,
    layers: &[F],
    config: &CheckpointConfig,
) -> GraphTensor
where
    F: Fn(GraphTensor) -> GraphTensor,
{
    let mut x = input;
    let n_layers = layers.len();

    for (i, layer) in layers.iter().enumerate() {
        // Apply layer
        x = layer(x);

        // Determine if we should checkpoint after this layer
        let should_checkpoint = if i == 0 {
            config.checkpoint_first_layer
        } else if i == n_layers - 1 {
            config.checkpoint_last_layer
        } else {
            (i + 1) % config.checkpoint_every_n_layers == 0
        };

        if should_checkpoint {
            x = checkpoint(x);
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::new()
            .every_n_layers(4)
            .checkpoint_first(true)
            .checkpoint_last(false);

        assert_eq!(config.checkpoint_every_n_layers, 4);
        assert!(config.checkpoint_first_layer);
        assert!(!config.checkpoint_last_layer);
    }

    #[test]
    fn test_checkpoint_manager() {
        let mut manager = CheckpointManager::new();

        manager.add_segment(NodeIndex::new(0), NodeIndex::new(1), true);
        manager.add_segment(NodeIndex::new(1), NodeIndex::new(2), false);
        manager.add_segment(NodeIndex::new(2), NodeIndex::new(3), true);

        let recompute: Vec<_> = manager.segments_to_recompute().collect();
        assert_eq!(recompute.len(), 2);
    }

    #[test]
    fn test_memory_estimation() {
        // 24 layers, 1GB per layer, checkpoint every 4 layers
        let (without, with, savings) = estimate_memory_savings(24, 1_000_000_000, 4);

        // Without: 24GB
        assert_eq!(without, 24_000_000_000);
        // With: ~10GB (6 checkpoints + 4 for current segment = 10)
        assert_eq!(with, 10_000_000_000);
        // ~58% savings
        assert!((savings - 0.583).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_passthrough() {
        let mut cx = Graph::new();
        let x = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);
        let y = checkpoint(x).retrieve();
        cx.execute();

        assert_eq!(y.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_checkpoint_layers() {
        let mut cx = Graph::new();
        let x = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);

        // Manual application to test the checkpoint concept
        let mut y = x;
        y = y * 2.0; // layer 0
        y = y + 1.0; // layer 1
        y = checkpoint(y); // checkpoint after every 2 layers
        y = (y * 0.5).retrieve(); // layer 2

        cx.execute();

        // (([1,2,3,4] * 2) + 1) * 0.5 = ([2,4,6,8] + 1) * 0.5 = [3,5,7,9] * 0.5 = [1.5, 2.5, 3.5, 4.5]
        let data = y.data();
        assert!((data[0] - 1.5).abs() < 1e-5);
        assert!((data[1] - 2.5).abs() < 1e-5);
        assert!((data[2] - 3.5).abs() < 1e-5);
        assert!((data[3] - 4.5).abs() < 1e-5);
    }
}
