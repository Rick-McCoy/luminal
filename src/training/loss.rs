use crate::prelude::*;

/// [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error).
///
/// This computes `(prediction - target).square().mean()`.
pub fn mse_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    (prediction - target)
        .square()
        .mean(prediction.shape.all_axes())
}

/// [Root Mean square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation).
///
/// This computes `(prediction - target).square().mean().sqrt()`
pub fn rmse_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    mse_loss(prediction, target).sqrt()
}

/// [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error).
///
/// This computes `(prediction - target).abs().mean()`
pub fn mae_loss(prediction: GraphTensor, target: GraphTensor) -> GraphTensor {
    (prediction - target)
        .abs()
        .mean(prediction.shape.all_axes())
}

/// [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss)
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < delta`: `0.5 * (x - y)^2`
/// 2. otherwise: `delta * (|x - y| - 0.5 * delta)`
pub fn huber_loss(
    prediction: GraphTensor,
    target: GraphTensor,
    delta: impl Into<f32>,
) -> GraphTensor {
    let delta: f32 = delta.into();
    let abs_error = (prediction - target).abs();
    let delta_tensor = prediction.graph().constant(delta);
    let huber_error = (0.5 * (prediction - target).square())
        * abs_error.lt(delta_tensor.expand(abs_error.shape))
        + (delta * (abs_error - 0.5 * delta)) * abs_error.ge(delta_tensor.expand(abs_error.shape));
    huber_error.mean(huber_error.shape.all_axes())
}

/// Smooth l1 loss (closely related to [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss))
/// uses absolute error when the error is higher than `beta`, and squared error when the
/// error is lower than `beta`.
///
/// It computes:
/// 1. if `|x - y| < beta`: `0.5 * (x - y)^2 / beta`
/// 2. otherwise: `|x - y| - 0.5 * beta`
pub fn smooth_l1_loss(
    prediction: GraphTensor,
    target: GraphTensor,
    delta: impl Into<f32> + Copy,
) -> GraphTensor {
    huber_loss(prediction, target, delta) / delta.into()
}

/// [Cross entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression).
/// This computes: `-(logits.log_softmax() * target_probs).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not the
/// output from** [softmax()] or [log_softmax()] already.
///
/// ### Inputs
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probabilities`: Target containing probability vectors **NOT** class indices.
pub fn cross_entropy_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(*logits.shape.dims().last().unwrap());
    let probs = logits.log_softmax(logits.shape.last_axis());
    (-(probs * target_probabilities).mean(target_probabilities.shape.all_axes()))
        / inv_last_axis_numel
}

/// [KL Divergence loss](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
/// This computes `(target_probs * (target_probs.log() - logits.log_softmax())).sum(-1).mean()`
///
/// This will call `log_softmax(logits)`, so make sure logits is **not the
/// output from** [softmax()] or [log_softmax()] already.
///
/// ### Inputs
///
/// - `logits`: The un-normalized output from a model. [log_softmax()] is called **in** this function
/// - `target_probs`: Target containing probability vectors **NOT** class indices.
pub fn kl_div_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let inv_last_axis_numel = 1.0
        / logits
            .graph()
            .constant(*logits.shape.dims().last().unwrap());
    let probs = logits.log_softmax(logits.shape.last_axis());
    (-((probs - target_probabilities.log()) * target_probabilities)
        .mean(target_probabilities.shape.all_axes()))
        / inv_last_axis_numel
}

/// [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression)
/// With Logits in numerically stable way.
///
/// Computes `target_probs * log(sigmoid(logits)) + (1 - target_probs) * log(1 - sigmoid(logits))`
/// as `(1 - target_probs) * logits + log(1 + exp(-logits))`.
///
/// ### Inputs
/// - `logits` - unnormalized inputs. **NOT** output of sigmoid
/// - `target_probs` - target values between 0 and 1.
pub fn binary_cross_entropy_with_logits_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
) -> GraphTensor {
    let bce = (1.0 - target_probabilities) * logits + (1.0 + (-logits).exp()).log();
    bce.mean(bce.shape.all_axes())
}

/// [Focal Loss](https://arxiv.org/abs/1708.02002) for dense object detection.
///
/// Focal loss addresses class imbalance by down-weighting easy examples and
/// focusing on hard negatives. It modifies cross-entropy loss with a modulating
/// factor (1 - p_t)^gamma.
///
/// The formula:
/// ```text
/// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
/// ```
///
/// where:
/// - `p_t` is the model's estimated probability for the correct class
/// - `γ` (gamma) is the focusing parameter (default: 2.0)
/// - `α` (alpha) is the balancing factor for class weights
///
/// ### Arguments
/// - `logits`: Unnormalized predictions (NOT after softmax)
/// - `target_probabilities`: One-hot encoded target labels
/// - `gamma`: Focusing parameter. Higher values down-weight easy examples more (default: 2.0)
/// - `alpha`: Optional class balancing weight. None means uniform weighting.
///
/// ### Returns
/// Scalar loss value (mean over all elements)
pub fn focal_loss_with_logits(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
    gamma: f32,
    alpha: Option<f32>,
) -> GraphTensor {
    // Compute softmax probabilities
    let probs = logits.softmax(logits.shape.last_axis());

    // p_t = p if y=1, (1-p) if y=0
    // For one-hot targets: p_t = sum(target * probs)
    let p_t = (probs * target_probabilities).sum(logits.shape.last_axis());

    // Focal weight: (1 - p_t)^gamma
    let focal_weight = (1.0 - p_t).pow(gamma);

    // Cross-entropy: -log(p_t)
    let ce = -p_t.log();

    // Focal loss
    let focal = focal_weight * ce;

    // Apply alpha weighting if provided
    let loss = if let Some(a) = alpha {
        focal * a
    } else {
        focal
    };

    loss.mean(loss.shape.all_axes())
}

/// Binary focal loss for binary classification tasks.
///
/// Similar to focal loss but for binary classification.
///
/// ```text
/// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
/// ```
///
/// ### Arguments
/// - `logits`: Unnormalized predictions (NOT after sigmoid)
/// - `targets`: Binary target values (0 or 1)
/// - `gamma`: Focusing parameter (default: 2.0)
/// - `alpha`: Weight for positive class (weight for negative is 1-alpha)
pub fn binary_focal_loss_with_logits(
    logits: GraphTensor,
    targets: GraphTensor,
    gamma: f32,
    alpha: f32,
) -> GraphTensor {
    // Compute sigmoid probabilities
    let probs = logits.sigmoid();

    // p_t = p if target=1, (1-p) if target=0
    let p_t = probs * targets + (1.0 - probs) * (1.0 - targets);

    // alpha_t = alpha if target=1, (1-alpha) if target=0
    let alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha);

    // Focal weight: (1 - p_t)^gamma
    let focal_weight = (1.0 - p_t).pow(gamma);

    // Cross-entropy: -log(p_t)
    // Use numerically stable version: max(logits, 0) - logits * target + log(1 + exp(-|logits|))
    let ce = logits.maximum_f32(0.0) - logits * targets + (1.0 + (-logits.abs()).exp()).log();

    // Focal loss
    let loss = alpha_t * focal_weight * ce;

    loss.mean(loss.shape.all_axes())
}

/// [Label Smoothing](https://arxiv.org/abs/1906.02629) cross-entropy loss.
///
/// Replaces hard one-hot targets with soft targets to prevent overconfidence.
///
/// ```text
/// y_smooth = (1 - ε) * y_onehot + ε / num_classes
/// ```
///
/// ### Arguments
/// - `logits`: Unnormalized predictions
/// - `target_probabilities`: One-hot encoded target labels
/// - `smoothing`: Label smoothing factor (typically 0.1)
pub fn label_smoothing_cross_entropy_loss(
    logits: GraphTensor,
    target_probabilities: GraphTensor,
    smoothing: f32,
) -> GraphTensor {
    let num_classes = *logits.shape.dims().last().unwrap();
    let num_classes_f = num_classes.to_usize().unwrap() as f32;

    // Smooth the targets: (1 - smoothing) * target + smoothing / num_classes
    let smooth_targets = target_probabilities * (1.0 - smoothing) + (smoothing / num_classes_f);

    // Cross entropy with smoothed targets
    cross_entropy_with_logits_loss(logits, smooth_targets)
}
