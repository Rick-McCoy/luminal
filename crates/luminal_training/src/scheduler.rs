use luminal::prelude::*;

/// A learning rate scheduler trait
pub trait LRScheduler {
    /// Get the learning rate for the given step
    fn get_lr(&self, step: usize) -> f32;

    /// Update the learning rate tensor for the given step
    fn step(&self, step: usize, lr_tensor: GraphTensor) {
        lr_tensor.set(vec![self.get_lr(step)]);
    }
}

/// Constant learning rate (no scheduling)
#[derive(Clone, Debug)]
pub struct ConstantLR {
    pub lr: f32,
}

impl ConstantLR {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Step learning rate decay
///
/// Multiplies the learning rate by `gamma` every `step_size` steps.
///
/// `lr = initial_lr * gamma^(step / step_size)`
#[derive(Clone, Debug)]
pub struct StepLR {
    pub initial_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

/// Exponential learning rate decay
///
/// `lr = initial_lr * gamma^step`
#[derive(Clone, Debug)]
pub struct ExponentialLR {
    pub initial_lr: f32,
    pub gamma: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing learning rate scheduler
///
/// Decreases the learning rate following a cosine curve from `initial_lr` to `min_lr`
/// over `total_steps` steps.
///
/// `lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / total_steps))`
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub total_steps: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, min_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.min_lr;
        }
        let progress = step as f32 / self.total_steps as f32;
        self.min_lr
            + 0.5
                * (self.initial_lr - self.min_lr)
                * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Linear warmup followed by cosine annealing
///
/// Linearly increases the learning rate from 0 to `initial_lr` over `warmup_steps`,
/// then decreases following a cosine curve to `min_lr` over the remaining steps.
#[derive(Clone, Debug)]
pub struct WarmupCosineAnnealingLR {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl WarmupCosineAnnealingLR {
    pub fn new(initial_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(
            warmup_steps < total_steps,
            "Warmup steps must be less than total steps"
        );
        Self {
            initial_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LRScheduler for WarmupCosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.initial_lr * (step as f32 / self.warmup_steps as f32)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine annealing
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            self.min_lr
                + 0.5
                    * (self.initial_lr - self.min_lr)
                    * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

/// Linear learning rate decay
///
/// Linearly decreases from `initial_lr` to `end_lr` over `total_steps` steps.
#[derive(Clone, Debug)]
pub struct LinearLR {
    pub initial_lr: f32,
    pub end_lr: f32,
    pub total_steps: usize,
}

impl LinearLR {
    pub fn new(initial_lr: f32, end_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            end_lr,
            total_steps,
        }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.end_lr;
        }
        let progress = step as f32 / self.total_steps as f32;
        self.initial_lr + (self.end_lr - self.initial_lr) * progress
    }
}

/// One Cycle learning rate policy
///
/// Increases LR from `initial_lr` to `max_lr` over `pct_start` fraction of steps,
/// then decreases from `max_lr` to `final_lr` over the remaining steps.
/// Both phases use cosine annealing.
#[derive(Clone, Debug)]
pub struct OneCycleLR {
    pub initial_lr: f32,
    pub max_lr: f32,
    pub final_lr: f32,
    pub total_steps: usize,
    pub pct_start: f32,
}

impl OneCycleLR {
    pub fn new(
        initial_lr: f32,
        max_lr: f32,
        final_lr: f32,
        total_steps: usize,
        pct_start: f32,
    ) -> Self {
        assert!(
            (0.0..1.0).contains(&pct_start),
            "pct_start must be in [0, 1)"
        );
        Self {
            initial_lr,
            max_lr,
            final_lr,
            total_steps,
            pct_start,
        }
    }

    /// Create with default values (30% warmup)
    pub fn default_cycle(max_lr: f32, total_steps: usize) -> Self {
        Self::new(max_lr / 25.0, max_lr, max_lr / 10000.0, total_steps, 0.3)
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, step: usize) -> f32 {
        let step = step.min(self.total_steps);
        let warmup_steps = (self.total_steps as f32 * self.pct_start) as usize;

        if step < warmup_steps {
            // Warmup phase: cosine from initial_lr to max_lr
            let progress = step as f32 / warmup_steps as f32;
            self.initial_lr
                + 0.5
                    * (self.max_lr - self.initial_lr)
                    * (1.0 - (std::f32::consts::PI * progress).cos())
        } else {
            // Annealing phase: cosine from max_lr to final_lr
            let progress = (step - warmup_steps) as f32 / (self.total_steps - warmup_steps) as f32;
            self.final_lr
                + 0.5
                    * (self.max_lr - self.final_lr)
                    * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}

/// Polynomial learning rate decay
///
/// `lr = (initial_lr - end_lr) * (1 - step/total_steps)^power + end_lr`
#[derive(Clone, Debug)]
pub struct PolynomialLR {
    pub initial_lr: f32,
    pub end_lr: f32,
    pub total_steps: usize,
    pub power: f32,
}

impl PolynomialLR {
    pub fn new(initial_lr: f32, end_lr: f32, total_steps: usize, power: f32) -> Self {
        Self {
            initial_lr,
            end_lr,
            total_steps,
            power,
        }
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.end_lr;
        }
        let decay = (1.0 - step as f32 / self.total_steps as f32).powf(self.power);
        (self.initial_lr - self.end_lr) * decay + self.end_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(0.1, 10, 0.1);
        assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(9) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let scheduler = ExponentialLR::new(1.0, 0.9);
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(1) - 0.9).abs() < 1e-6);
        assert!((scheduler.get_lr(2) - 0.81).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealingLR::new(1.0, 0.0, 100);
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(50) - 0.5).abs() < 1e-6);
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_cosine() {
        let scheduler = WarmupCosineAnnealingLR::new(1.0, 0.0, 10, 100);
        // Warmup phase
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-6);
        assert!((scheduler.get_lr(5) - 0.5).abs() < 1e-6);
        assert!((scheduler.get_lr(10) - 1.0).abs() < 1e-6);
        // Cosine phase
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_lr() {
        let scheduler = LinearLR::new(1.0, 0.0, 100);
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(50) - 0.5).abs() < 1e-6);
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_cycle() {
        let scheduler = OneCycleLR::new(0.0, 1.0, 0.0, 100, 0.3);
        // Start at initial_lr
        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-5);
        // Peak at max_lr around 30%
        assert!((scheduler.get_lr(30) - 1.0).abs() < 1e-5);
        // End at final_lr
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_polynomial_lr() {
        let scheduler = PolynomialLR::new(1.0, 0.0, 100, 1.0); // Linear decay
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((scheduler.get_lr(50) - 0.5).abs() < 1e-6);
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-6);

        let scheduler = PolynomialLR::new(1.0, 0.0, 100, 2.0); // Quadratic decay
        assert!((scheduler.get_lr(50) - 0.25).abs() < 1e-6);
    }
}
