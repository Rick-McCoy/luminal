//! Gradient accumulation utilities for training with larger effective batch sizes.
//!
//! Gradient accumulation allows training with larger effective batch sizes than
//! what fits in memory by accumulating gradients over multiple forward/backward
//! passes before updating weights.

use crate::prelude::*;

/// Accumulator for gradients over multiple micro-batches.
///
/// This struct helps manage gradient accumulation across multiple forward/backward
/// passes before applying optimizer updates.
///
/// # Example
/// ```ignore
/// let mut accumulator = GradientAccumulator::new(4); // Accumulate over 4 micro-batches
///
/// for epoch in 0..num_epochs {
///     for (batch_idx, batch) in data.enumerate() {
///         // Forward and backward pass
///         let grads = compute_gradients(&batch, &mut cx);
///         
///         // Accumulate gradients
///         accumulator.accumulate(&grads, &mut cx);
///         
///         // Update weights when accumulation is complete
///         if accumulator.should_step() {
///             let avg_grads = accumulator.get_averaged_gradients(&mut cx);
///             optimizer.step(&avg_grads);
///             accumulator.zero_grad(&mut cx);
///         }
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct GradientAccumulator {
    /// Number of micro-batches to accumulate over
    pub accumulation_steps: usize,
    /// Current step count
    current_step: usize,
    /// Accumulated gradient tensors (None until first accumulation)
    accumulated_grads: Option<Vec<(NodeIndex, ShapeTracker)>>,
}

impl GradientAccumulator {
    /// Create a new gradient accumulator.
    ///
    /// # Arguments
    /// - `accumulation_steps`: Number of micro-batches to accumulate before stepping
    pub fn new(accumulation_steps: usize) -> Self {
        assert!(
            accumulation_steps > 0,
            "accumulation_steps must be at least 1"
        );
        Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: None,
        }
    }

    /// Returns the current step count within the accumulation window.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Returns true if we should perform an optimizer step.
    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get the effective batch size multiplier.
    pub fn effective_batch_multiplier(&self) -> usize {
        self.accumulation_steps
    }

    /// Accumulate gradients from a micro-batch.
    ///
    /// Adds the gradients to the running sum. Call `get_averaged_gradients()`
    /// to get the mean gradients when `should_step()` returns true.
    pub fn accumulate(&mut self, grads: &[(NodeIndex, ShapeTracker)], graph: &mut Graph) {
        self.current_step += 1;

        if let Some(ref mut acc_grads) = self.accumulated_grads {
            // Add to existing accumulated gradients
            for ((acc_id, acc_shape), (grad_id, grad_shape)) in
                acc_grads.iter_mut().zip(grads.iter())
            {
                let acc = GraphTensor::from_id(*acc_id, *acc_shape, graph);
                let grad = GraphTensor::from_id(*grad_id, *grad_shape, graph);
                let new_acc = acc + grad;
                new_acc.keep();
                *acc_id = new_acc.id;
            }
        } else {
            // First accumulation: clone the gradients
            let mut acc_grads = Vec::with_capacity(grads.len());
            for (grad_id, grad_shape) in grads.iter() {
                let grad = GraphTensor::from_id(*grad_id, *grad_shape, graph);
                // Create a copy that we own
                let acc = grad.contiguous();
                acc.keep();
                acc_grads.push((acc.id, *grad_shape));
            }
            self.accumulated_grads = Some(acc_grads);
        }
    }

    /// Get the averaged gradients after accumulation.
    ///
    /// Returns gradients divided by `accumulation_steps` to compute the mean.
    /// Call this when `should_step()` returns true.
    pub fn get_averaged_gradients(&self, graph: &mut Graph) -> Vec<(NodeIndex, ShapeTracker)> {
        let acc_grads = self
            .accumulated_grads
            .as_ref()
            .expect("No gradients accumulated yet");

        let scale = 1.0 / self.accumulation_steps as f32;

        acc_grads
            .iter()
            .map(|(acc_id, acc_shape)| {
                let acc = GraphTensor::from_id(*acc_id, *acc_shape, graph);
                let averaged = acc * scale;
                averaged.keep();
                (averaged.id, *acc_shape)
            })
            .collect()
    }

    /// Zero out accumulated gradients for the next accumulation window.
    ///
    /// Call this after `get_averaged_gradients()` and the optimizer step.
    pub fn zero_grad(&mut self, _graph: &mut Graph) {
        self.current_step = 0;
        self.accumulated_grads = None;
    }
}

/// Helper function to scale gradients for gradient accumulation.
///
/// When using gradient accumulation, you typically want to scale the loss
/// by 1/accumulation_steps so that the accumulated gradients have the same
/// magnitude as a single large batch.
///
/// # Arguments
/// - `loss`: The loss tensor to scale
/// - `accumulation_steps`: Number of micro-batches being accumulated
///
/// # Returns
/// Scaled loss tensor
pub fn scale_loss_for_accumulation(loss: GraphTensor, accumulation_steps: usize) -> GraphTensor {
    loss * (1.0 / accumulation_steps as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_steps() {
        let accumulator = GradientAccumulator::new(4);
        assert_eq!(accumulator.current_step(), 0);
        assert!(!accumulator.should_step());
        assert_eq!(accumulator.effective_batch_multiplier(), 4);
    }

    #[test]
    fn test_accumulator_accumulate() {
        let mut cx = Graph::new();
        let mut accumulator = GradientAccumulator::new(2);

        // Create mock gradients
        let grad1 = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);
        let grads = vec![(grad1.id, grad1.shape)];

        // First accumulation
        accumulator.accumulate(&grads, &mut cx);
        assert_eq!(accumulator.current_step(), 1);
        assert!(!accumulator.should_step());

        // Second accumulation
        let grad2 = cx.tensor(4).set(vec![2.0, 4.0, 6.0, 8.0]);
        let grads2 = vec![(grad2.id, grad2.shape)];
        accumulator.accumulate(&grads2, &mut cx);
        assert_eq!(accumulator.current_step(), 2);
        assert!(accumulator.should_step());

        // Get averaged gradients
        let avg_grads = accumulator.get_averaged_gradients(&mut cx);
        let avg = GraphTensor::from_id(avg_grads[0].0, avg_grads[0].1, &mut cx).retrieve();
        cx.execute();

        // Average of [1,2,3,4] and [2,4,6,8] = [1.5, 3.0, 4.5, 6.0]
        let data = avg.data();
        assert!((data[0] - 1.5).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        assert!((data[2] - 4.5).abs() < 1e-5);
        assert!((data[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale_loss() {
        let mut cx = Graph::new();
        let loss = cx.tensor(1).set(vec![4.0]);
        let scaled = scale_loss_for_accumulation(loss, 4).retrieve();
        cx.execute();

        assert!((scaled.data()[0] - 1.0).abs() < 1e-5);
    }
}
