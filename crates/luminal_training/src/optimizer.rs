use luminal::compiler_utils::ToIdsMut;
use luminal::prelude::*;

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (Old weight inputs, Gradient inputs, New weight outputs, Optimizer Graph, Learning Rate Tensor)
pub fn sgd(
    grads: &[(NodeIndex, ShapeTracker)],
) -> (
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Vec<NodeIndex>,
    Graph,
    GraphTensor,
) {
    let mut opt_graph = Graph::new();
    let (old_weights, gradients): (Vec<NodeIndex>, Vec<NodeIndex>) = grads
        .iter()
        .map(|_| (opt_graph.tensor(1).id, opt_graph.tensor(1).id))
        .unzip();

    let (new_weights, lr) = sgd_on_graph(
        &mut opt_graph,
        &old_weights,
        &gradients
            .iter()
            .zip(grads)
            .map(|(a, (_, b))| (*a, *b))
            .collect::<Vec<_>>(),
    );
    (old_weights, gradients, new_weights, opt_graph, lr)
}

/// [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
///
/// `new_weight = old_weight - (gradient * learning_rate)`
///
/// Output: (New weight outputs, Learning Rate Tensor)
pub fn sgd_on_graph(
    graph: &mut Graph,
    old_weights: impl ToIds,
    grads: &[(NodeIndex, ShapeTracker)],
) -> (Vec<NodeIndex>, GraphTensor) {
    let lr = graph.named_tensor("Learning Rate", 1).set(3e-4).keep(); // Karpathy constant
    let mut new_weights = vec![];
    for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weights.to_ids()) {
        let old_weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
        let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);

        // SGD
        let new_weight = old_weight - (gradient * lr.expand(grad_shape));
        new_weight.keep();

        new_weights.push(new_weight.id);
    }

    (new_weights, lr)
}

/// Configuration for the Adam optimizer
#[derive(Clone, Debug)]
pub struct AdamConfig {
    /// Learning rate (default: 1e-3)
    pub lr: f32,
    /// Coefficient for first moment estimate (default: 0.9)
    pub beta1: f32,
    /// Coefficient for second moment estimate (default: 0.999)
    pub beta2: f32,
    /// Term added to denominator for numerical stability (default: 1e-8)
    pub epsilon: f32,
    /// Weight decay coefficient for AdamW (default: 0.0, meaning standard Adam)
    pub weight_decay: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamConfig {
    /// Create a new Adam config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the learning rate
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set beta1 (first moment coefficient)
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (second moment coefficient)
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set epsilon (numerical stability term)
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay (0.0 for standard Adam, >0 for AdamW)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

/// State tensors for the Adam optimizer
///
/// These need to be kept between training steps and updated after each step.
#[derive(Clone)]
pub struct AdamState {
    /// First moment estimates (one per parameter tensor)
    pub m: Vec<GraphTensor>,
    /// Second moment estimates (one per parameter tensor)
    pub v: Vec<GraphTensor>,
    /// Timestep counter
    pub t: GraphTensor,
    /// Updated first moment estimates (copy back to m after each step)
    pub m_new: Vec<GraphTensor>,
    /// Updated second moment estimates (copy back to v after each step)
    pub v_new: Vec<GraphTensor>,
    /// Updated timestep (copy back to t after each step)
    pub t_new: GraphTensor,
}

impl AdamState {
    /// Get all state tensor node indices for keeping in the graph
    pub fn all_tensors(&self) -> Vec<NodeIndex> {
        let mut tensors = vec![self.t.id, self.t_new.id];
        tensors.extend(self.m.iter().map(|t| t.id));
        tensors.extend(self.v.iter().map(|t| t.id));
        tensors.extend(self.m_new.iter().map(|t| t.id));
        tensors.extend(self.v_new.iter().map(|t| t.id));
        tensors
    }


    /// Transfer updated states back to the original state tensors
    ///
    /// Call this after each training step along with transferring new weights.
    pub fn step(&self, graph: &mut Graph) {
        // Transfer m_new -> m
        let m_new_ids: Vec<_> = self.m_new.iter().map(|t| t.id).collect();
        let m_ids: Vec<_> = self.m.iter().map(|t| t.id).collect();
        transfer_data_same_graph(&m_new_ids, &m_ids, graph);

        // Transfer v_new -> v
        let v_new_ids: Vec<_> = self.v_new.iter().map(|t| t.id).collect();
        let v_ids: Vec<_> = self.v.iter().map(|t| t.id).collect();
        transfer_data_same_graph(&v_new_ids, &v_ids, graph);

        // Transfer t_new -> t
        transfer_data_same_graph(vec![self.t_new.id], vec![self.t.id], graph);
    }
}

/// Implement ToIdsMut so AdamState can be passed to Graph::compile()
/// This is necessary for Metal/CUDA compilation which changes node IDs.
impl ToIdsMut for AdamState {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        let mut ids = vec![&mut self.t.id, &mut self.t_new.id];
        ids.extend(self.m.iter_mut().map(|t| &mut t.id));
        ids.extend(self.v.iter_mut().map(|t| &mut t.id));
        ids.extend(self.m_new.iter_mut().map(|t| &mut t.id));
        ids.extend(self.v_new.iter_mut().map(|t| &mut t.id));
        ids
    }
}

/// [Adam optimizer](https://arxiv.org/abs/1412.6980) with optional weight decay (AdamW)
///
/// The Adam algorithm:
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
/// ```
///
/// With weight decay (AdamW), an additional term is added:
/// ```text
/// θ_t = θ_t - α * λ * θ_{t-1}
/// ```
///
/// # Returns
/// - `new_weights`: Updated parameter tensors
/// - `lr`: Learning rate tensor (can be modified between steps)
/// - `state`: Adam state containing momentum tensors (must be stepped after each iteration)
pub fn adam_on_graph(
    graph: &mut Graph,
    old_weights: impl ToIds,
    grads: &[(NodeIndex, ShapeTracker)],
    config: AdamConfig,
) -> (Vec<NodeIndex>, GraphTensor, AdamState) {
    let old_weight_ids = old_weights.to_ids();

    // Create learning rate tensor
    let lr = graph.named_tensor("Learning Rate", 1).set(config.lr).keep();

    // Create timestep tensor (starts at 0, will be incremented)
    let t = graph.named_tensor("Adam t", 1).set(0.0).keep();
    let t_new = (t + 1.0).retrieve();
    t_new.keep();

    // Precompute bias correction terms
    // bias1 = 1 - β1^t, bias2 = 1 - β2^t
    let beta1_tensor = graph.constant(config.beta1);
    let beta2_tensor = graph.constant(config.beta2);
    let one = graph.constant(1.0);

    // β1^t and β2^t using exp(t * log(β))
    let beta1_pow_t = (t_new * beta1_tensor.log()).exp();
    let beta2_pow_t = (t_new * beta2_tensor.log()).exp();
    let bias_correction1 = one - beta1_pow_t;
    let bias_correction2 = one - beta2_pow_t;

    let mut new_weights = vec![];
    let mut m_states = vec![];
    let mut v_states = vec![];
    let mut m_new_states = vec![];
    let mut v_new_states = vec![];

    for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weight_ids) {
        let old_weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
        let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);

        // Initialize m and v to zeros with the same shape as the gradient
        let n_elements = grad_shape.n_elements().to_usize().unwrap_or(1);
        let m = graph
            .named_tensor("Adam m", grad_shape)
            .set(vec![0.0; n_elements])
            .keep();
        let v = graph
            .named_tensor("Adam v", grad_shape)
            .set(vec![0.0; n_elements])
            .keep();

        // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
        let m_new = m * config.beta1 + gradient * (1.0 - config.beta1);
        m_new.keep();

        // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
        let v_new = v * config.beta2 + (gradient * gradient) * (1.0 - config.beta2);
        v_new.keep();

        // Compute bias-corrected estimates
        // m̂_t = m_t / (1 - β1^t)
        let m_hat = m_new / bias_correction1.expand(grad_shape);
        // v̂_t = v_t / (1 - β2^t)
        let v_hat = v_new / bias_correction2.expand(grad_shape);

        // Compute update: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        let update = lr.expand(grad_shape) * m_hat / (v_hat.sqrt() + config.epsilon);
        let mut new_weight = old_weight - update;

        // Apply weight decay if specified (AdamW)
        if config.weight_decay > 0.0 {
            new_weight -= lr.expand(grad_shape) * config.weight_decay * old_weight;
        }

        new_weight.keep();
        new_weights.push(new_weight.id);
        m_states.push(m);
        v_states.push(v);
        m_new_states.push(GraphTensor::from_id(m_new.id, grad_shape, graph));
        v_new_states.push(GraphTensor::from_id(v_new.id, grad_shape, graph));
    }

    let state = AdamState {
        m: m_states,
        v: v_states,
        t,
        m_new: m_new_states,
        v_new: v_new_states,
        t_new: GraphTensor::from_id(t_new.id, ShapeTracker::new(1), graph),
    };

    (new_weights, lr, state)
}

/// [RMSprop optimizer](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
///
/// ```text
/// v_t = α * v_{t-1} + (1 - α) * g_t²
/// θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
/// ```
///
/// # Arguments
/// - `alpha`: Smoothing constant (default: 0.99)
/// - `epsilon`: Term added to denominator for numerical stability (default: 1e-8)
///
/// # Returns
/// - `new_weights`: Updated parameter tensors
/// - `lr`: Learning rate tensor
/// - `v_states`: Running average of squared gradients (keep and update between steps)
/// - `v_new_states`: Updated v states (transfer to v_states after each step)
pub fn rmsprop_on_graph(
    graph: &mut Graph,
    old_weights: impl ToIds,
    grads: &[(NodeIndex, ShapeTracker)],
    lr_value: f32,
    alpha: f32,
    epsilon: f32,
) -> (
    Vec<NodeIndex>,
    GraphTensor,
    Vec<GraphTensor>,
    Vec<GraphTensor>,
) {
    let old_weight_ids = old_weights.to_ids();
    let lr = graph.named_tensor("Learning Rate", 1).set(lr_value).keep();

    let mut new_weights = vec![];
    let mut v_states = vec![];
    let mut v_new_states = vec![];

    for ((grad_id, grad_shape), old_weight_id) in grads.iter().copied().zip(old_weight_ids) {
        let old_weight = GraphTensor::from_id(old_weight_id, grad_shape, graph);
        let gradient = GraphTensor::from_id(grad_id, grad_shape, graph);

        // Initialize v to zeros
        let n_elements = grad_shape.n_elements().to_usize().unwrap_or(1);
        let v = graph
            .named_tensor("RMSprop v", grad_shape)
            .set(vec![0.0; n_elements])
            .keep();

        // Update running average: v_t = α * v_{t-1} + (1 - α) * g_t²
        let v_new = v * alpha + (gradient * gradient) * (1.0 - alpha);
        v_new.keep();

        // Compute update: θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)
        let new_weight = old_weight - lr.expand(grad_shape) * gradient / (v_new.sqrt() + epsilon);
        new_weight.keep();

        new_weights.push(new_weight.id);
        v_states.push(v);
        v_new_states.push(GraphTensor::from_id(v_new.id, grad_shape, graph));
    }

    (new_weights, lr, v_states, v_new_states)
}

/// Clip gradients by their global norm
///
/// If the global norm of all gradients exceeds `max_norm`, scales all gradients
/// down proportionally so the global norm equals `max_norm`.
///
/// # Arguments
/// - `grads`: Gradient tensors with their shapes
/// - `max_norm`: Maximum allowed global norm
///
/// # Returns
/// Clipped gradient tensors (same shapes as input)
pub fn clip_grad_norm(
    graph: &mut Graph,
    grads: &[(NodeIndex, ShapeTracker)],
    max_norm: f32,
) -> Vec<GraphTensor> {
    // Compute global norm: sqrt(sum of all squared gradient elements)
    let mut total_norm_sq = graph.constant(0.0);

    for (grad_id, grad_shape) in grads.iter().copied() {
        let grad = GraphTensor::from_id(grad_id, grad_shape, graph);
        let grad_norm_sq = (grad * grad).sum(grad_shape.all_axes());
        total_norm_sq += grad_norm_sq;
    }

    let total_norm = total_norm_sq.sqrt();

    // Compute clip coefficient: min(max_norm / total_norm, 1.0)
    let max_norm_tensor = graph.constant(max_norm);
    let clip_coef = (max_norm_tensor / (total_norm + 1e-6)).minimum_f32(1.0);

    // Apply clipping to each gradient
    grads
        .iter()
        .copied()
        .map(|(grad_id, grad_shape)| {
            let grad = GraphTensor::from_id(grad_id, grad_shape, graph);
            grad * clip_coef.expand(grad_shape)
        })
        .collect()
}

/// Clip gradients by value
///
/// Clamps all gradient values to be within `[-clip_value, clip_value]`.
///
/// # Arguments
/// - `grads`: Gradient tensors with their shapes
/// - `clip_value`: Maximum absolute value for any gradient element
///
/// # Returns
/// Clipped gradient tensors (same shapes as input)
pub fn clip_grad_value(
    graph: &mut Graph,
    grads: &[(NodeIndex, ShapeTracker)],
    clip_value: f32,
) -> Vec<GraphTensor> {
    grads
        .iter()
        .copied()
        .map(|(grad_id, grad_shape)| {
            let grad = GraphTensor::from_id(grad_id, grad_shape, graph);
            grad.maximum_f32(-clip_value).minimum_f32(clip_value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::tests::assert_close;

    #[test]
    fn test_adam_basic() {
        let mut cx = Graph::new();

        // Create a simple parameter and gradient
        let weight = cx.tensor(4).set(vec![1.0, 2.0, 3.0, 4.0]).keep();
        let grad = cx.tensor(4).set(vec![0.1, 0.2, 0.3, 0.4]).keep();

        let grads = vec![(grad.id, grad.shape)];
        let (new_weights, _lr, state) =
            adam_on_graph(&mut cx, vec![weight.id], &grads, AdamConfig::default());

        cx.keep_tensors(state.all_tensors());
        cx.keep_tensors(new_weights.clone());

        // First step
        cx.execute();

        let new_weight = GraphTensor::from_id(new_weights[0], weight.shape, &mut cx);
        let result = new_weight.data();

        // After first step, weights should have decreased (gradient descent)
        assert!(result[0] < 1.0);
        assert!(result[1] < 2.0);
        assert!(result[2] < 3.0);
        assert!(result[3] < 4.0);

        // Transfer states for next iteration
        state.step(&mut cx);
        transfer_data_same_graph(new_weights, vec![weight.id], &mut cx);
    }

    #[test]
    fn test_adam_converges() {
        // Test that Adam can minimize a simple quadratic loss
        let mut cx = Graph::new();

        // Parameter to optimize (target is 0)
        let weight = cx.tensor(1).set(vec![10.0]).keep();
        let grad_input = cx.tensor(1).keep(); // We'll set this to 2*weight each step

        let grads = vec![(grad_input.id, grad_input.shape)];
        let config = AdamConfig::default().lr(0.1);
        let (new_weights, _lr, state) = adam_on_graph(&mut cx, vec![weight.id], &grads, config);

        cx.keep_tensors(state.all_tensors());
        cx.keep_tensors(new_weights.clone());
        cx.compile(GenericCompiler::default(), ());

        let mut current = 10.0f32;
        for _ in 0..100 {
            // Gradient of x^2 is 2x
            grad_input.set(vec![2.0 * current]);
            cx.execute();

            let new_weight = GraphTensor::from_id(new_weights[0], weight.shape, &mut cx);
            current = new_weight.data()[0];

            state.step(&mut cx);
            transfer_data_same_graph(new_weights.clone(), vec![weight.id], &mut cx);
        }

        // Should be close to 0
        assert!(current.abs() < 0.1, "Adam did not converge: {}", current);
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut cx = Graph::new();

        let grad1 = cx.tensor(3).set(vec![3.0, 4.0, 0.0]); // norm = 5
        let grad2 = cx.tensor(2).set(vec![0.0, 0.0]); // norm = 0

        let grads = vec![(grad1.id, grad1.shape), (grad2.id, grad2.shape)];
        let clipped = clip_grad_norm(&mut cx, &grads, 2.5); // max_norm = 2.5, scale = 0.5

        let out1 = clipped[0].retrieve();
        let out2 = clipped[1].retrieve();
        cx.execute();

        // Global norm is 5, max_norm is 2.5, so scale factor is 0.5
        assert_close(&out1.data(), &[1.5, 2.0, 0.0]);
        assert_close(&out2.data(), &[0.0, 0.0]);
    }

    #[test]
    fn test_clip_grad_value() {
        let mut cx = Graph::new();

        let grad = cx.tensor(4).set(vec![-2.0, -0.5, 0.5, 2.0]);
        let grads = vec![(grad.id, grad.shape)];
        let clipped = clip_grad_value(&mut cx, &grads, 1.0);

        let out = clipped[0].retrieve();
        cx.execute();

        assert_close(&out.data(), &[-1.0, -0.5, 0.5, 1.0]);
    }
}
