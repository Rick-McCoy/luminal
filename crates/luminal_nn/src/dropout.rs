use luminal::prelude::*;
use rand::{rng, Rng};

/// Dropout regularization layer
///
/// During training, randomly zeroes elements with probability `p` and scales
/// the remaining elements by `1/(1-p)`.
///
/// During inference (when `training` is false), passes input through unchanged.
///
/// # Usage
///
/// The dropout mask needs to be regenerated for each forward pass during training.
/// Call `resample_mask()` before each training step:
///
/// ```ignore
/// let mut dropout = Dropout::new(0.5, (batch_size, hidden_dim), &mut cx);
/// dropout.set_training(true);
///
/// for batch in data {
///     dropout.resample_mask(); // Generate new random mask
///     let output = model.forward(input);
///     // ... rest of training step
/// }
/// ```
pub struct Dropout {
    /// Dropout probability (fraction of elements to zero out)
    pub p: f32,
    /// Whether the layer is in training mode
    pub training: bool,
    /// The mask tensor (set before each forward pass)
    pub mask: GraphTensor,
    /// Shape of the mask
    shape: Vec<usize>,
}

impl Dropout {
    /// Create a new Dropout layer
    ///
    /// # Arguments
    /// - `p`: Probability of an element being zeroed (0.0 to 1.0)
    /// - `shape`: Shape of the input tensor (for mask generation)
    /// - `cx`: The computation graph
    pub fn new(p: f32, shape: impl ToShape + Copy, cx: &mut Graph) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        let shape_tracker = ShapeTracker::new(shape);
        let dims: Vec<usize> = shape_tracker
            .dims()
            .iter()
            .map(|e| e.to_usize().expect("Dropout requires static shape"))
            .collect();
        let n_elements: usize = dims.iter().product();

        // Initialize mask to all ones (no dropout)
        let mask = cx
            .named_tensor("Dropout mask", shape)
            .set(vec![1.0; n_elements])
            .keep();

        Self {
            p,
            training: true,
            mask,
            shape: dims,
        }
    }

    /// Set whether the layer is in training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        if !training {
            // In inference mode, set mask to all ones
            let n_elements: usize = self.shape.iter().product();
            self.mask.set(vec![1.0; n_elements]);
        }
    }

    /// Resample the dropout mask
    ///
    /// Call this before each forward pass during training to generate a new random mask.
    pub fn resample_mask(&self) {
        if !self.training {
            return;
        }

        let mut rng = rng();
        let scale = 1.0 / (1.0 - self.p);
        let n_elements: usize = self.shape.iter().product();

        let mask_data: Vec<f32> = (0..n_elements)
            .map(|_| {
                if rng.random::<f32>() >= self.p {
                    scale
                } else {
                    0.0
                }
            })
            .collect();

        self.mask.set(mask_data);
    }
}

impl SerializeModule for Dropout {
    fn serialize(&self, _s: &mut Serializer) {
        // Dropout has no learnable parameters
    }
}

impl Module<GraphTensor> for Dropout {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        // Multiply input by mask (which is either random dropout mask or all ones)
        input * self.mask.expand(input.shape)
    }
}

/// A simpler dropout that doesn't require pre-specifying shape
///
/// This version creates a mask tensor dynamically but requires the user
/// to provide the mask data externally.
pub struct DropoutWithMask {
    /// Dropout probability
    pub p: f32,
    /// Whether in training mode
    pub training: bool,
}

impl DropoutWithMask {
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Self { p, training: true }
    }

    /// Generate a dropout mask for the given shape
    pub fn generate_mask(&self, shape: &[usize]) -> Vec<f32> {
        if !self.training {
            return vec![1.0; shape.iter().product()];
        }

        let mut rng = rng();
        let scale = 1.0 / (1.0 - self.p);
        let n_elements: usize = shape.iter().product();

        (0..n_elements)
            .map(|_| {
                if rng.random::<f32>() >= self.p {
                    scale
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl SerializeModule for DropoutWithMask {
    fn serialize(&self, _s: &mut Serializer) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::tests::assert_close;

    #[test]
    fn test_dropout_inference() {
        let mut cx = Graph::new();
        let input = cx.tensor((2, 3)).set(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut dropout = Dropout::new(0.5, (2, 3), &mut cx);
        dropout.set_training(false); // Inference mode

        let output = dropout.forward(input).retrieve();
        cx.execute();

        // In inference mode, output should equal input
        assert_close(&output.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dropout_training() {
        let mut cx = Graph::new();
        let input = cx.tensor((2, 3)).set(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let dropout = Dropout::new(0.5, (2, 3), &mut cx);
        dropout.resample_mask();

        let output = dropout.forward(input).retrieve();
        cx.execute();

        let result = output.data();

        // Check that some values are zeroed and others are scaled by 2
        let zeros = result.iter().filter(|&&x| x == 0.0).count();
        let scaled = result.iter().filter(|&&x| (x - 2.0).abs() < 1e-5).count();

        // With p=0.5, we expect roughly half zeros and half scaled values
        // Allow for some variance due to randomness
        assert!(zeros + scaled == 6, "All values should be either 0 or 2");
    }

    #[test]
    fn test_dropout_expected_value() {
        // Test that the expected value is preserved (E[dropout(x)] ≈ x)
        let mut cx = Graph::new();
        let input_data = vec![1.0; 1000];
        let input = cx.tensor(1000).set(input_data.clone());

        let dropout = Dropout::new(0.3, 1000, &mut cx);

        let mut total_sum = 0.0;
        let n_trials = 100;

        for _ in 0..n_trials {
            dropout.resample_mask();
            let output = dropout.forward(input).retrieve();
            cx.execute();

            total_sum += output.data().iter().sum::<f32>();
            output.drop();
        }

        let avg = total_sum / (n_trials as f32 * 1000.0);
        // Expected value should be close to 1.0
        assert!(
            (avg - 1.0).abs() < 0.1,
            "Expected value should be ~1.0, got {}",
            avg
        );
    }

    #[test]
    fn test_dropout_zero_probability() {
        let mut cx = Graph::new();
        let input = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]);

        let dropout = Dropout::new(0.0, 4, &mut cx);
        dropout.resample_mask();

        let output = dropout.forward(input).retrieve();
        cx.execute();

        // With p=0, no dropout should occur
        assert_close(&output.data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
