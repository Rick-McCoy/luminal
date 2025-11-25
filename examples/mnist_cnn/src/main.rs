//! Train an MLP on MNIST using luminal
//!
//! This example demonstrates:
//! - Building an MLP with Linear layers
//! - Using the Adam optimizer with cosine annealing learning rate schedule
//! - Training on the MNIST dataset
//! - GPU acceleration with Metal (macOS) or CUDA
//!
//! First, download the MNIST dataset:
//! ```
//! cd examples/mnist_cnn && ./setup.sh
//! ```
//!
//! Run with Metal backend (macOS):
//! ```
//! cargo run -p mnist_cnn --release --features metal
//! ```
//!
//! Run with CUDA backend:
//! ```
//! cargo run -p mnist_cnn --release --features cuda
//! ```
//!
//! Run with CPU only:
//! ```
//! cargo run -p mnist_cnn --release
//! ```

use luminal::prelude::*;
use luminal_nn::Linear;
use luminal_training::{
    adam_on_graph, cross_entropy_with_logits_loss, AdamConfig, Autograd, CosineAnnealingLR,
    LRScheduler,
};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// MLP architecture for MNIST:
/// Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)
///
/// Note: CNN version is available but has issues with Metal backend.
/// Using MLP for more reliable cross-platform training.
struct MnistMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MnistMLP {
    fn new(cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new(784, 256, false, cx),
            fc2: Linear::new(256, 128, false, cx),
            fc3: Linear::new(128, 10, false, cx),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        // x: (batch, 1, 28, 28) -> flatten to (batch, 784)
        let batch = x.dims()[0];
        let x = x.reshape((batch, 784));

        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x) // logits
    }

    fn init_rand(self) -> Self {
        Self {
            fc1: self.fc1.init_rand(),
            fc2: self.fc2.init_rand(),
            fc3: self.fc3.init_rand(),
        }
    }
}

impl SerializeModule for MnistMLP {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
        s.module("fc3", &self.fc3);
    }
}

fn main() {
    println!("Loading MNIST dataset...");

    // Load MNIST data
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("examples/mnist_cnn/data")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    // Normalize pixel values to [0, 1]
    let train_images: Vec<f32> = trn_img.iter().map(|&x| x as f32 / 255.0).collect();
    let train_labels: Vec<u8> = trn_lbl;
    let test_images: Vec<f32> = tst_img.iter().map(|&x| x as f32 / 255.0).collect();
    let test_labels: Vec<u8> = tst_lbl;

    println!(
        "Loaded {} training images, {} test images",
        train_labels.len(),
        test_labels.len()
    );

    // Training hyperparameters
    let batch_size = 64;
    let epochs = 3;
    let max_iters = (train_labels.len() / batch_size) * epochs;
    let initial_lr = 0.001;
    let final_lr = 0.0001;

    println!("\nBuilding MLP model and optimizer...");

    // Build computation graph
    let mut cx = Graph::new();
    let model = MnistMLP::new(&mut cx).init_rand();

    // Input: (batch, 1, 28, 28), Target: (batch, 10) one-hot
    let mut input = cx.tensor((batch_size, 1, 28, 28));
    let mut target = cx.tensor((batch_size, 10));

    let mut output = model.forward(input).retrieve();
    let mut loss = cross_entropy_with_logits_loss(output, target).retrieve();

    // Get model parameters and compute gradients
    let mut weights = params(&model);
    let grads = cx.compile(Autograd::new(&weights, loss), ());

    // Adam optimizer
    let adam_config = AdamConfig::default().lr(initial_lr);
    let (mut new_weights, lr, mut adam_state) =
        adam_on_graph(&mut cx, &weights, &grads, adam_config);

    // Learning rate scheduler
    let scheduler = CosineAnnealingLR::new(initial_lr, final_lr, max_iters);

    // Keep tensors across iterations
    cx.keep_tensors(adam_state.all_tensors());
    cx.keep_tensors(new_weights.clone());
    cx.keep_tensors(&weights);

    // Compile to backend - pass ALL tensors including adam_state for ID remapping
    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            luminal_metal::MetalCompiler::<f32>::default(),
            #[cfg(feature = "cuda")]
            luminal_cuda::CudaCompiler::<f32>::default(),
        ),
        (
            &mut input,
            &mut target,
            &mut loss,
            &mut output,
            &mut weights,
            &mut new_weights,
            &mut adam_state, // Important: include adam_state for Metal/CUDA!
        ),
    );

    println!("Training MLP on MNIST...");
    println!("Epochs: {epochs}, Batch size: {batch_size}, Total iterations: {max_iters}");
    #[cfg(feature = "metal")]
    println!("Backend: Metal GPU");
    #[cfg(feature = "cuda")]
    println!("Backend: CUDA GPU");
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    println!("Backend: CPU");
    println!();

    // Training loop
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..train_labels.len()).collect();
    let mut loss_avg = ExponentialAverage::new(2.0);
    let mut acc_avg = ExponentialAverage::new(0.0);
    let mut iter = 0;
    let start = std::time::Instant::now();

    for epoch in 0..epochs {
        // Shuffle training data each epoch
        indices.shuffle(&mut rng);

        for batch_start in (0..train_labels.len()).step_by(batch_size) {
            if batch_start + batch_size > train_labels.len() {
                break; // Skip incomplete batch
            }

            // Update learning rate
            scheduler.step(iter, lr);

            // Prepare batch
            let batch_indices = &indices[batch_start..batch_start + batch_size];
            let (batch_images, batch_targets) =
                prepare_batch(batch_indices, &train_images, &train_labels, batch_size);

            input.set(batch_images);
            target.set(batch_targets.clone());

            // Forward + backward + optimizer step
            cx.execute();

            // Update weights
            transfer_data_same_graph(new_weights.clone(), &weights, &mut cx);
            adam_state.step(&mut cx);

            // Compute batch accuracy
            let predictions = output.data();
            let batch_acc = compute_batch_accuracy(&predictions, &batch_targets, batch_size);

            // Update metrics
            loss_avg.update(loss.data()[0]);
            acc_avg.update(batch_acc);

            loss.drop();
            output.drop();

            // Print progress every 100 iterations
            if iter % 100 == 0 {
                println!(
                    "Epoch {} | Iter {:5} | Loss: {:.4} | Acc: {:5.1}% | LR: {:.6}",
                    epoch + 1,
                    iter,
                    loss_avg.value,
                    acc_avg.value * 100.0,
                    scheduler.get_lr(iter)
                );
            }
            iter += 1;
        }
    }

    let train_time = start.elapsed();
    println!("\n=== Training Complete ===");
    println!(
        "Final Loss: {:.4}, Final Train Acc: {:.1}%",
        loss_avg.value,
        acc_avg.value * 100.0
    );
    println!(
        "Time: {:.1}s ({:.1}ms / iter)",
        train_time.as_secs_f32(),
        train_time.as_millis() as f32 / iter as f32
    );

    // Evaluate on test set
    println!("\n=== Evaluating on Test Set ===");
    let test_acc = evaluate_test_set(
        &mut cx,
        &mut input,
        &mut output,
        &test_images,
        &test_labels,
        batch_size,
    );
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);
}

fn prepare_batch(
    indices: &[usize],
    images: &[f32],
    labels: &[u8],
    batch_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut batch_images = Vec::with_capacity(batch_size * 28 * 28);
    let mut batch_targets = Vec::with_capacity(batch_size * 10);

    for &idx in indices {
        // Image: 28x28 = 784 pixels
        let img_start = idx * 784;
        batch_images.extend_from_slice(&images[img_start..img_start + 784]);

        // One-hot encode label
        let mut one_hot = [0.0f32; 10];
        one_hot[labels[idx] as usize] = 1.0;
        batch_targets.extend_from_slice(&one_hot);
    }

    (batch_images, batch_targets)
}

fn compute_batch_accuracy(predictions: &[f32], targets: &[f32], batch_size: usize) -> f32 {
    let mut correct = 0;
    for i in 0..batch_size {
        let pred_start = i * 10;
        let target_start = i * 10;

        // Find argmax of predictions (handle NaN by treating as -inf)
        let pred_class = predictions[pred_start..pred_start + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Find argmax of targets
        let target_class = targets[target_start..target_start + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if pred_class == target_class {
            correct += 1;
        }
    }
    correct as f32 / batch_size as f32
}

fn evaluate_test_set(
    cx: &mut Graph,
    input: &mut GraphTensor,
    output: &mut GraphTensor,
    test_images: &[f32],
    test_labels: &[u8],
    batch_size: usize,
) -> f32 {
    let mut total_correct = 0;
    let mut total_samples = 0;

    for batch_start in (0..test_labels.len()).step_by(batch_size) {
        if batch_start + batch_size > test_labels.len() {
            break;
        }

        let indices: Vec<usize> = (batch_start..batch_start + batch_size).collect();
        let (batch_images, batch_targets) =
            prepare_batch(&indices, test_images, test_labels, batch_size);

        input.set(batch_images);
        cx.execute();

        let predictions = output.data();
        let batch_correct =
            (compute_batch_accuracy(&predictions, &batch_targets, batch_size) * batch_size as f32)
                as usize;

        total_correct += batch_correct;
        total_samples += batch_size;
        output.drop();
    }

    total_correct as f32 / total_samples as f32
}

/// Exponential moving average for smooth metrics
pub struct ExponentialAverage {
    beta: f32,
    moment: f32,
    pub value: f32,
    t: i32,
}

impl ExponentialAverage {
    fn new(initial: f32) -> Self {
        Self {
            beta: 0.99,
            moment: 0.,
            value: initial,
            t: 0,
        }
    }

    pub fn update(&mut self, value: f32) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value;
        self.value = self.moment / (1. - f32::powi(self.beta, self.t));
    }
}
