use luminal::{prelude::*, tests::random_vec_rng};
use rand::rng;

/// A simple layer norm with an optional weight and bias
#[derive(Default)]
pub struct LayerNorm {
    pub weight: Option<GraphTensor>,
    pub bias: Option<GraphTensor>,
    mean_norm: bool,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(
        dim: usize,
        weight: bool,
        bias: bool,
        mean_norm: bool,
        epsilon: f32,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: if weight {
                Some(cx.named_tensor("LayerNorm Weight", dim))
            } else {
                None
            },
            bias: if bias {
                Some(cx.named_tensor("LayerNorm Bias", dim))
            } else {
                None
            },
            mean_norm,
            epsilon,
        }
    }
    pub fn initialize(self) -> Self {
        // Init weight as uniform(-1, 1)
        let mut rng = rng();
        if let Some(w) = self.weight {
            w.set(random_vec_rng(
                w.shape.n_elements().to_usize().unwrap(),
                &mut rng,
            ));
        }
        if let Some(b) = self.bias {
            b.set(random_vec_rng(
                b.shape.n_elements().to_usize().unwrap(),
                &mut rng,
            ));
        }
        self
    }
}

impl Module<GraphTensor> for LayerNorm {
    type Output = GraphTensor;
    fn forward(&self, mut input: GraphTensor) -> Self::Output {
        if self.mean_norm {
            input = input.mean_norm(input.shape.last_axis());
        }
        input = input.std_norm(input.shape.last_axis(), self.epsilon);
        if let Some(w) = self.weight {
            input *= w.expand(input.shape);
        }
        if let Some(b) = self.bias {
            input += b.expand(input.shape);
        }
        input
    }
}

impl SerializeModule for LayerNorm {
    fn serialize(&self, s: &mut Serializer) {
        if let Some(w) = self.weight {
            s.tensor("weight", w);
        }
        if let Some(b) = self.bias {
            s.tensor("bias", b);
        }
    }
}

/// RMSNorm (Root Mean Square Layer Normalization)
///
/// Used in LLaMA and other modern architectures. Simpler than LayerNorm
/// as it only normalizes by RMS without centering.
///
/// `output = input / sqrt(mean(input^2) + epsilon) * weight`
pub struct RMSNorm {
    pub weight: GraphTensor,
    epsilon: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, epsilon: f32, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("RMSNorm Weight", dim).set(vec![1.0; dim]),
            epsilon,
        }
    }

    /// Initialize weights to ones (standard initialization)
    pub fn init_ones(self) -> Self {
        let n = self.weight.shape.n_elements().to_usize().unwrap();
        self.weight.set(vec![1.0; n]);
        self
    }
}

impl Module<GraphTensor> for RMSNorm {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        // RMS = sqrt(mean(x^2))
        // output = x / (RMS + epsilon) * weight
        let rms = (input * input)
            .mean(input.shape.last_axis())
            .sqrt()
            .expand(input.shape);
        (input / (rms + self.epsilon)) * self.weight.expand(input.shape)
    }
}

impl SerializeModule for RMSNorm {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
    }
}

/// Batch Normalization for 1D inputs (N, C) or (N, C, L)
///
/// During training, normalizes using batch statistics.
/// During inference, uses running mean/variance.
///
/// Note: In luminal's static graph model, you need to update running stats
/// externally after each forward pass during training.
pub struct BatchNorm1d {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub running_mean: GraphTensor,
    pub running_var: GraphTensor,
    pub num_features: usize,
    pub epsilon: f32,
    pub momentum: f32,
    pub training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize, epsilon: f32, momentum: f32, cx: &mut Graph) -> Self {
        Self {
            weight: cx
                .named_tensor("BatchNorm1d Weight", num_features)
                .set(vec![1.0; num_features])
                .keep(),
            bias: cx
                .named_tensor("BatchNorm1d Bias", num_features)
                .set(vec![0.0; num_features])
                .keep(),
            running_mean: cx
                .named_tensor("BatchNorm1d Running Mean", num_features)
                .set(vec![0.0; num_features])
                .keep(),
            running_var: cx
                .named_tensor("BatchNorm1d Running Var", num_features)
                .set(vec![1.0; num_features])
                .keep(),
            num_features,
            epsilon,
            momentum,
            training: true,
        }
    }

    /// Create with default epsilon=1e-5 and momentum=0.1
    pub fn default_new(num_features: usize, cx: &mut Graph) -> Self {
        Self::new(num_features, 1e-5, 0.1, cx)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Update running statistics after a training forward pass
    /// Call this with the batch mean and variance computed during forward
    pub fn update_running_stats(&self, batch_mean: &[f32], batch_var: &[f32]) {
        let current_mean = self.running_mean.data();
        let current_var = self.running_var.data();

        let new_mean: Vec<f32> = current_mean
            .iter()
            .zip(batch_mean.iter())
            .map(|(r, b)| (1.0 - self.momentum) * r + self.momentum * b)
            .collect();

        let new_var: Vec<f32> = current_var
            .iter()
            .zip(batch_var.iter())
            .map(|(r, b)| (1.0 - self.momentum) * r + self.momentum * b)
            .collect();

        self.running_mean.set(new_mean);
        self.running_var.set(new_var);
    }
}

impl BatchNorm1d {
    /// Forward pass for shape (N, C) or (N, C, L)
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let ndim = input.shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "BatchNorm1d expects 2D (N,C) or 3D (N,C,L) input"
        );

        // Normalize over all dimensions except the channel dimension (dim 1)
        // For training: use batch statistics
        // For inference: use running statistics

        if self.training {
            // Compute batch statistics
            // For (N, C): normalize over N
            // For (N, C, L): normalize over N and L
            let axes: Vec<usize> = if ndim == 2 { vec![0] } else { vec![0, 2] };

            let mean = input.mean(axes.clone());
            let variance = ((input - mean.expand(input.shape))
                * (input - mean.expand(input.shape)))
            .mean(axes);

            let normalized = (input - mean.expand(input.shape))
                / (variance.expand(input.shape) + self.epsilon).sqrt();

            // Apply affine transformation
            if ndim == 2 {
                normalized * self.weight.expand(input.shape) + self.bias.expand(input.shape)
            } else {
                // For 3D, weight/bias are (C,) and need to expand to (1, C, 1)
                let w = self.weight.expand_dim(0, 1).expand_dim(2, input.dims()[2]);
                let b = self.bias.expand_dim(0, 1).expand_dim(2, input.dims()[2]);
                normalized * w.expand(input.shape) + b.expand(input.shape)
            }
        } else {
            // Use running statistics
            let normalized = (input - self.running_mean.expand(input.shape))
                / (self.running_var.expand(input.shape) + self.epsilon).sqrt();

            if ndim == 2 {
                normalized * self.weight.expand(input.shape) + self.bias.expand(input.shape)
            } else {
                let w = self.weight.expand_dim(0, 1).expand_dim(2, input.dims()[2]);
                let b = self.bias.expand_dim(0, 1).expand_dim(2, input.dims()[2]);
                normalized * w.expand(input.shape) + b.expand(input.shape)
            }
        }
    }
}

impl SerializeModule for BatchNorm1d {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
        s.tensor("bias", self.bias);
        s.tensor("running_mean", self.running_mean);
        s.tensor("running_var", self.running_var);
    }
}

/// Batch Normalization for 2D inputs (N, C, H, W)
///
/// Normalizes over the batch and spatial dimensions, keeping channel dimension.
pub struct BatchNorm2d {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub running_mean: GraphTensor,
    pub running_var: GraphTensor,
    pub num_features: usize,
    pub epsilon: f32,
    pub momentum: f32,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize, epsilon: f32, momentum: f32, cx: &mut Graph) -> Self {
        Self {
            weight: cx
                .named_tensor("BatchNorm2d Weight", num_features)
                .set(vec![1.0; num_features])
                .keep(),
            bias: cx
                .named_tensor("BatchNorm2d Bias", num_features)
                .set(vec![0.0; num_features])
                .keep(),
            running_mean: cx
                .named_tensor("BatchNorm2d Running Mean", num_features)
                .set(vec![0.0; num_features])
                .keep(),
            running_var: cx
                .named_tensor("BatchNorm2d Running Var", num_features)
                .set(vec![1.0; num_features])
                .keep(),
            num_features,
            epsilon,
            momentum,
            training: true,
        }
    }

    /// Create with default epsilon=1e-5 and momentum=0.1
    pub fn default_new(num_features: usize, cx: &mut Graph) -> Self {
        Self::new(num_features, 1e-5, 0.1, cx)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Update running statistics after a training forward pass
    pub fn update_running_stats(&self, batch_mean: &[f32], batch_var: &[f32]) {
        let current_mean = self.running_mean.data();
        let current_var = self.running_var.data();

        let new_mean: Vec<f32> = current_mean
            .iter()
            .zip(batch_mean.iter())
            .map(|(r, b)| (1.0 - self.momentum) * r + self.momentum * b)
            .collect();

        let new_var: Vec<f32> = current_var
            .iter()
            .zip(batch_var.iter())
            .map(|(r, b)| (1.0 - self.momentum) * r + self.momentum * b)
            .collect();

        self.running_mean.set(new_mean);
        self.running_var.set(new_var);
    }
}

impl BatchNorm2d {
    /// Forward pass for shape (N, C, H, W)
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        assert!(
            input.shape.len() == 4,
            "BatchNorm2d expects 4D (N,C,H,W) input"
        );

        let (batch, channels, height, width) = input.dims4();

        // Reshape to (N, C, H*W) for easier normalization
        let reshaped = input.reshape((batch, channels, height * width));

        // Normalize over batch and spatial dimensions (0 and 2)
        let axes = vec![0, 2];

        if self.training {
            let mean = reshaped.mean(axes.clone());
            let centered = reshaped - mean.expand(reshaped.shape);
            let variance = (centered * centered).mean(axes);

            let normalized = centered / (variance.expand(reshaped.shape) + self.epsilon).sqrt();

            // Apply affine: weight and bias are (C,), expand to (1, C, 1)
            let w = self.weight.expand_dim(0, 1).expand_dim(2, height * width);
            let b = self.bias.expand_dim(0, 1).expand_dim(2, height * width);

            let output = normalized * w.expand(reshaped.shape) + b.expand(reshaped.shape);
            output.reshape((batch, channels, height, width))
        } else {
            // Use running statistics
            let rm = self
                .running_mean
                .expand_dim(0, 1)
                .expand_dim(2, height * width);
            let rv = self
                .running_var
                .expand_dim(0, 1)
                .expand_dim(2, height * width);

            let normalized = (reshaped - rm.expand(reshaped.shape))
                / (rv.expand(reshaped.shape) + self.epsilon).sqrt();

            let w = self.weight.expand_dim(0, 1).expand_dim(2, height * width);
            let b = self.bias.expand_dim(0, 1).expand_dim(2, height * width);

            let output = normalized * w.expand(reshaped.shape) + b.expand(reshaped.shape);
            output.reshape((batch, channels, height, width))
        }
    }
}

impl SerializeModule for BatchNorm2d {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
        s.tensor("bias", self.bias);
        s.tensor("running_mean", self.running_mean);
        s.tensor("running_var", self.running_var);
    }
}

/// Group Normalization
///
/// Divides channels into groups and normalizes within each group.
/// Unlike BatchNorm, doesn't depend on batch size.
pub struct GroupNorm {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub num_groups: usize,
    pub num_channels: usize,
    pub epsilon: f32,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize, epsilon: f32, cx: &mut Graph) -> Self {
        assert!(
            num_channels.is_multiple_of(num_groups),
            "num_channels must be divisible by num_groups"
        );
        Self {
            weight: cx
                .named_tensor("GroupNorm Weight", num_channels)
                .set(vec![1.0; num_channels]),
            bias: cx
                .named_tensor("GroupNorm Bias", num_channels)
                .set(vec![0.0; num_channels]),
            num_groups,
            num_channels,
            epsilon,
        }
    }
}

impl GroupNorm {
    /// Forward for (N, C, *) where * is any number of spatial dimensions
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let shape = input.dims();
        let batch = shape[0];
        let channels = shape[1];
        let channels_per_group = channels / self.num_groups;

        // Reshape to (N, num_groups, channels_per_group, *)
        // Then normalize over (channels_per_group, *)
        let spatial: Expression = shape[2..].iter().fold(1.into(), |acc, &x| acc * x);

        let reshaped = input.reshape((batch, self.num_groups, channels_per_group * spatial));

        // Normalize over the last dimension
        let mean = reshaped.mean(2);
        let centered = reshaped - mean.expand(reshaped.shape);
        let variance = (centered * centered).mean(2);
        let normalized = centered / (variance.expand(reshaped.shape) + self.epsilon).sqrt();

        // Reshape back
        let mut output_shape = vec![batch, channels];
        output_shape.extend_from_slice(&shape[2..]);
        let normalized = normalized.reshape(output_shape.clone());

        // Apply affine transformation
        // Weight and bias are (C,), need to match output shape
        let mut weight_shape = vec![1.into(), channels];
        weight_shape.extend(shape[2..].iter().map(|_| Expression::from(1)));
        let w = self.weight.reshape(weight_shape.clone());
        let b = self.bias.reshape(weight_shape);

        normalized * w.expand(normalized.shape) + b.expand(normalized.shape)
    }
}

impl SerializeModule for GroupNorm {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight", self.weight);
        s.tensor("bias", self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::tests::assert_close;

    #[test]
    fn test_rmsnorm() {
        let mut cx = Graph::new();
        let input = cx
            .tensor((2, 4))
            .set(vec![1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]);

        let norm = RMSNorm::new(4, 1e-5, &mut cx);
        let output = norm.forward(input).retrieve();

        cx.execute();

        let result = output.data();
        // For [1,2,3,4]: RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // normalized ≈ [0.365, 0.730, 1.095, 1.461]
        assert!(result[0] > 0.3 && result[0] < 0.4);
        assert!(result[3] > 1.4 && result[3] < 1.5);
    }

    #[test]
    fn test_batchnorm1d_inference() {
        let mut cx = Graph::new();

        let mut bn = BatchNorm1d::default_new(3, &mut cx);
        bn.set_training(false);

        // Set known running stats
        bn.running_mean.set(vec![0.0, 0.0, 0.0]);
        bn.running_var.set(vec![1.0, 1.0, 1.0]);

        let input = cx.tensor((2, 3)).set(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = bn.forward(input).retrieve();

        cx.execute();

        // With mean=0, var=1, weight=1, bias=0, output should equal input
        assert_close(&output.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_batchnorm2d_shape() {
        let mut cx = Graph::new();

        let mut bn = BatchNorm2d::default_new(3, &mut cx);
        bn.set_training(false);

        let input = cx.tensor((2, 3, 4, 4)).set(vec![1.0; 2 * 3 * 4 * 4]);
        let output = bn.forward(input).retrieve();

        cx.execute();

        // Check output has correct shape
        assert_eq!(output.data().len(), 2 * 3 * 4 * 4);
    }
}
