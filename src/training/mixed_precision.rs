//! Mixed precision training utilities.
//!
//! Mixed precision training uses FP16 (or BF16) for forward/backward passes
//! while keeping FP32 master weights and gradients. This provides:
//! - ~2x memory reduction for activations and gradients
//! - Faster computation on GPUs with tensor cores
//! - Same model quality as FP32 training
//!
//! Key components:
//! - Dynamic loss scaling to prevent gradient underflow
//! - FP32 master weights for optimizer updates
//! - Automatic gradient unscaling

use crate::prelude::*;

/// Configuration for mixed precision training.
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    /// Initial loss scale (default: 65536.0)
    pub init_scale: f32,
    /// Factor to increase scale by when no overflow (default: 2.0)
    pub scale_factor: f32,
    /// Factor to decrease scale by on overflow (default: 0.5)
    pub backoff_factor: f32,
    /// Number of consecutive non-overflow steps before increasing scale (default: 2000)
    pub growth_interval: usize,
    /// Minimum allowed scale (default: 1.0)
    pub min_scale: f32,
    /// Maximum allowed scale (default: 65536.0 * 65536.0)
    pub max_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            scale_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 65536.0 * 65536.0,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial loss scale.
    pub fn init_scale(mut self, scale: f32) -> Self {
        self.init_scale = scale;
        self
    }

    /// Set scale growth factor.
    pub fn scale_factor(mut self, factor: f32) -> Self {
        self.scale_factor = factor;
        self
    }

    /// Set scale backoff factor on overflow.
    pub fn backoff_factor(mut self, factor: f32) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set growth interval.
    pub fn growth_interval(mut self, interval: usize) -> Self {
        self.growth_interval = interval;
        self
    }
}

/// Dynamic loss scaler for mixed precision training.
///
/// Automatically scales the loss to prevent gradient underflow in FP16,
/// and detects gradient overflow to adjust the scale dynamically.
///
/// # Algorithm
///
/// 1. Multiply loss by `scale` before backward pass
/// 2. After backward pass, check gradients for inf/nan
/// 3. If overflow detected:
///    - Skip optimizer step
///    - Reduce scale by `backoff_factor`
/// 4. If no overflow:
///    - Increment success counter
///    - If counter reaches `growth_interval`, increase scale by `scale_factor`
///
/// # Example
/// ```ignore
/// let mut scaler = GradScaler::new(MixedPrecisionConfig::default());
///
/// for batch in data {
///     // Forward pass (in FP16)
///     let loss = model.forward(batch);
///     
///     // Scale loss before backward
///     let scaled_loss = scaler.scale(loss);
///     
///     // Backward pass
///     let grads = compute_gradients(scaled_loss);
///     
///     // Unscale gradients and check for overflow
///     let (unscaled_grads, overflow) = scaler.unscale_and_check(&grads, &mut cx);
///     
///     if !overflow {
///         optimizer.step(&unscaled_grads);
///     }
///     
///     // Update scaler state
///     scaler.update(overflow);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct GradScaler {
    /// Current loss scale.
    scale: f32,
    /// Number of consecutive non-overflow steps.
    growth_tracker: usize,
    /// Configuration.
    config: MixedPrecisionConfig,
    /// Whether scaler is enabled.
    enabled: bool,
}

impl GradScaler {
    /// Create a new gradient scaler.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let scale = config.init_scale;
        Self {
            scale,
            growth_tracker: 0,
            config,
            enabled: true,
        }
    }

    /// Create with default configuration.
    pub fn default_scaler() -> Self {
        Self::new(MixedPrecisionConfig::default())
    }

    /// Disable the scaler (pass-through mode).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable the scaler.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Get the current scale value.
    pub fn get_scale(&self) -> f32 {
        if self.enabled {
            self.scale
        } else {
            1.0
        }
    }

    /// Scale a loss tensor.
    ///
    /// Multiplies the loss by the current scale factor.
    pub fn scale(&self, loss: GraphTensor) -> GraphTensor {
        if self.enabled {
            loss * self.scale
        } else {
            loss
        }
    }

    /// Unscale gradients.
    ///
    /// Divides gradients by the current scale factor.
    pub fn unscale(
        &self,
        grads: &[(NodeIndex, ShapeTracker)],
        graph: &mut Graph,
    ) -> Vec<(NodeIndex, ShapeTracker)> {
        if !self.enabled {
            return grads.to_vec();
        }

        let inv_scale = 1.0 / self.scale;
        grads
            .iter()
            .map(|(grad_id, grad_shape)| {
                let grad = GraphTensor::from_id(*grad_id, *grad_shape, graph);
                let unscaled = grad * inv_scale;
                unscaled.keep();
                (unscaled.id, *grad_shape)
            })
            .collect()
    }

    /// Check if any gradient contains inf or nan values.
    ///
    /// This should be called after unscaling to detect overflow.
    ///
    /// Returns true if overflow/nan is detected.
    pub fn check_for_overflow(grads: &[(NodeIndex, ShapeTracker)], graph: &mut Graph) -> bool {
        // Sum all gradient values and check if result is finite
        // This is a simple but effective check
        let mut total = graph.constant(0.0);

        for (grad_id, grad_shape) in grads.iter() {
            let grad = GraphTensor::from_id(*grad_id, *grad_shape, graph);
            // Sum of absolute values to detect both +inf and -inf
            let abs_sum = grad.abs().sum(grad_shape.all_axes());
            total += abs_sum;
        }

        // Execute to get the result
        let result = total.retrieve();
        graph.execute();

        let sum_value = result.data()[0];
        sum_value.is_nan() || sum_value.is_infinite()
    }

    /// Unscale gradients and check for overflow in one operation.
    ///
    /// Returns (unscaled_gradients, overflow_detected)
    pub fn unscale_and_check(
        &self,
        grads: &[(NodeIndex, ShapeTracker)],
        graph: &mut Graph,
    ) -> (Vec<(NodeIndex, ShapeTracker)>, bool) {
        let unscaled = self.unscale(grads, graph);
        let overflow = Self::check_for_overflow(&unscaled, graph);
        (unscaled, overflow)
    }

    /// Update the scaler after a training step.
    ///
    /// Call this after each optimizer step (or skipped step on overflow).
    ///
    /// # Arguments
    /// - `overflow`: Whether overflow was detected in this step
    pub fn update(&mut self, overflow: bool) {
        if !self.enabled {
            return;
        }

        if overflow {
            // Reduce scale on overflow
            self.scale = (self.scale * self.config.backoff_factor).max(self.config.min_scale);
            self.growth_tracker = 0;
        } else {
            // Track successful steps
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                // Increase scale after enough successful steps
                self.scale = (self.scale * self.config.scale_factor).min(self.config.max_scale);
                self.growth_tracker = 0;
            }
        }
    }

    /// Get scaler state for checkpointing.
    pub fn state(&self) -> (f32, usize) {
        (self.scale, self.growth_tracker)
    }

    /// Load scaler state from checkpoint.
    pub fn load_state(&mut self, state: (f32, usize)) {
        self.scale = state.0;
        self.growth_tracker = state.1;
    }
}

/// Helper struct for managing FP32 master weights.
///
/// In mixed precision training, we keep FP32 copies of weights for optimizer
/// updates to maintain precision, while using FP16 weights for forward/backward.
#[derive(Clone)]
pub struct MasterWeights {
    /// FP32 master weight tensors
    pub fp32_weights: Vec<NodeIndex>,
    /// FP16/BF16 weight tensors used in model
    pub model_weights: Vec<NodeIndex>,
    /// Shape trackers for each weight
    pub shapes: Vec<ShapeTracker>,
}

impl MasterWeights {
    /// Create master weights from model weights.
    ///
    /// This copies the FP32 weights to serve as master weights.
    pub fn from_model_weights(weights: &[NodeIndex], graph: &mut Graph) -> Self {
        let mut fp32_weights = vec![];
        let mut shapes = vec![];

        for &weight_id in weights {
            // Get the weight tensor and its shape
            let sources = graph.get_sources(weight_id);
            let shape = if let Some((_, _, sh)) = sources.first() {
                *sh
            } else {
                ShapeTracker::new(1) // Fallback
            };

            // Create FP32 copy
            let weight = GraphTensor::from_id(weight_id, shape, graph);
            let master = weight.contiguous();
            master.keep();

            fp32_weights.push(master.id);
            shapes.push(shape);
        }

        Self {
            fp32_weights,
            model_weights: weights.to_vec(),
            shapes,
        }
    }

    /// Update model weights from master weights.
    ///
    /// Call this after optimizer update to sync the FP16 model weights.
    pub fn sync_to_model(&self, graph: &mut Graph) {
        transfer_data_same_graph(&self.fp32_weights, &self.model_weights, graph);
    }

    /// Get master weight tensors for optimizer.
    pub fn get_master_weights(&self) -> Vec<(NodeIndex, ShapeTracker)> {
        self.fp32_weights
            .iter()
            .zip(&self.shapes)
            .map(|(&id, &shape)| (id, shape))
            .collect()
    }
}

/// Automatic Mixed Precision (AMP) context manager.
///
/// Combines gradient scaling and master weights management.
///
/// # Example
/// ```ignore
/// let mut amp = AMPContext::new(model_weights, &mut cx);
///
/// for batch in data {
///     // Forward in FP16
///     let loss = model.forward(batch);
///     
///     // Backward with scaled loss
///     let scaled_loss = amp.scale_loss(loss);
///     let grads = autograd(&scaled_loss);
///     
///     // Optimizer step with AMP handling
///     if amp.step(&grads, &mut optimizer, &mut cx) {
///         // Step was successful
///     } else {
///         // Overflow detected, step skipped
///     }
/// }
/// ```
pub struct AMPContext {
    /// Gradient scaler.
    pub scaler: GradScaler,
    /// Master weights (if using).
    pub master_weights: Option<MasterWeights>,
}

impl AMPContext {
    /// Create a new AMP context.
    pub fn new() -> Self {
        Self {
            scaler: GradScaler::default_scaler(),
            master_weights: None,
        }
    }

    /// Create with master weights.
    pub fn with_master_weights(weights: &[NodeIndex], graph: &mut Graph) -> Self {
        Self {
            scaler: GradScaler::default_scaler(),
            master_weights: Some(MasterWeights::from_model_weights(weights, graph)),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: MixedPrecisionConfig) -> Self {
        Self {
            scaler: GradScaler::new(config),
            master_weights: None,
        }
    }

    /// Scale loss for backward pass.
    pub fn scale_loss(&self, loss: GraphTensor) -> GraphTensor {
        self.scaler.scale(loss)
    }

    /// Perform optimizer step with AMP handling.
    ///
    /// Returns true if step was performed, false if skipped due to overflow.
    pub fn step<F>(
        &mut self,
        grads: &[(NodeIndex, ShapeTracker)],
        graph: &mut Graph,
        optimizer_step: F,
    ) -> bool
    where
        F: FnOnce(&[(NodeIndex, ShapeTracker)], &mut Graph),
    {
        let (unscaled, overflow) = self.scaler.unscale_and_check(grads, graph);

        if !overflow {
            optimizer_step(&unscaled, graph);

            // Sync master weights if using
            if let Some(ref master) = self.master_weights {
                master.sync_to_model(graph);
            }
        }

        self.scaler.update(overflow);
        !overflow
    }

    /// Get current loss scale.
    pub fn get_scale(&self) -> f32 {
        self.scaler.get_scale()
    }
}

impl Default for AMPContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_scaler_basic() {
        let scaler = GradScaler::default_scaler();
        assert_eq!(scaler.get_scale(), 65536.0);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let mut cx = Graph::new();
        let scaler = GradScaler::new(MixedPrecisionConfig::default().init_scale(4.0));

        let loss = cx.tensor(1).set(vec![2.5]);
        let scaled = scaler.scale(loss).retrieve();
        cx.execute();

        assert!((scaled.data()[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_scaler_unscale() {
        let mut cx = Graph::new();
        let scaler = GradScaler::new(MixedPrecisionConfig::default().init_scale(4.0));

        let grad = cx.tensor(4).set(vec![4.0, 8.0, 12.0, 16.0]);
        let grads = vec![(grad.id, grad.shape)];

        let unscaled = scaler.unscale(&grads, &mut cx);
        let result = GraphTensor::from_id(unscaled[0].0, unscaled[0].1, &mut cx).retrieve();
        cx.execute();

        let data = result.data();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_scaler_update_no_overflow() {
        let mut scaler = GradScaler::new(
            MixedPrecisionConfig::default()
                .init_scale(1.0)
                .growth_interval(2)
                .scale_factor(2.0),
        );

        assert_eq!(scaler.get_scale(), 1.0);

        // First successful step
        scaler.update(false);
        assert_eq!(scaler.get_scale(), 1.0);

        // Second successful step - should trigger growth
        scaler.update(false);
        assert_eq!(scaler.get_scale(), 2.0);
    }

    #[test]
    fn test_grad_scaler_update_overflow() {
        let mut scaler = GradScaler::new(
            MixedPrecisionConfig::default()
                .init_scale(4.0)
                .backoff_factor(0.5),
        );

        assert_eq!(scaler.get_scale(), 4.0);

        // Overflow detected
        scaler.update(true);
        assert_eq!(scaler.get_scale(), 2.0);

        // Another overflow
        scaler.update(true);
        assert_eq!(scaler.get_scale(), 1.0);
    }

    #[test]
    fn test_grad_scaler_disabled() {
        let mut cx = Graph::new();
        let mut scaler = GradScaler::default_scaler();
        scaler.disable();

        let loss = cx.tensor(1).set(vec![2.5]);
        let scaled = scaler.scale(loss).retrieve();
        cx.execute();

        // Should not scale when disabled
        assert!((scaled.data()[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_amp_context() {
        let amp = AMPContext::new();
        assert_eq!(amp.get_scale(), 65536.0);
    }

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::new()
            .init_scale(1024.0)
            .scale_factor(4.0)
            .backoff_factor(0.25)
            .growth_interval(100);

        assert_eq!(config.init_scale, 1024.0);
        assert_eq!(config.scale_factor, 4.0);
        assert_eq!(config.backoff_factor, 0.25);
        assert_eq!(config.growth_interval, 100);
    }
}
