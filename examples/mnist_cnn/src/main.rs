//! Train an MLP on MNIST using luminal
//!
//! This example demonstrates:
//! - Building an MLP with Linear layers
//! - Using the Adam optimizer with cosine annealing learning rate schedule
//! - Training on the MNIST dataset
//! - GPU acceleration with Metal (macOS) or CUDA
//! - Optional search-based kernel optimization
//!
//! ## Setup
//!
//! Download the MNIST dataset:
//! ```bash
//! cd examples/mnist_cnn && ./setup.sh
//! ```
//!
//! ## Running
//!
//! Fast mode (default, hand-written kernels):
//! ```bash
//! cargo run -p mnist_cnn --release --features metal
//! cargo run -p mnist_cnn --release --features cuda
//! ```
//!
//! Optimal mode (search-based compilation):
//! ```bash
//! cargo run -p mnist_cnn --release --features metal,search -- --optimal
//! cargo run -p mnist_cnn --release --features cuda,search -- --optimal
//! ```

use std::time::{Duration, Instant};

use clap::Parser;
use luminal::prelude::*;
use luminal::nn::Linear;
use luminal::training::{
    adam_on_graph, cross_entropy_with_logits_loss, AdamConfig, Autograd, CosineAnnealingLR,
    LRScheduler,
};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Train an MLP on MNIST with luminal
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Number of training epochs
    #[arg(short, long, default_value_t = 3)]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Use search-based optimal compilation (slower compile, faster runtime)
    #[arg(long)]
    optimal: bool,

    /// Print detailed performance metrics
    #[arg(short, long)]
    verbose: bool,
}

/// MLP architecture for MNIST:
/// Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)
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
        let batch = x.dims()[0];
        let x = x.reshape((batch, 784));
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x)
    }

    fn init_rand(self) -> Self {
        Self {
            fc1: self.fc1.init_rand(),
            fc2: self.fc2.init_rand(),
            fc3: self.fc3.init_rand(),
        }
    }

    fn param_count(&self) -> usize {
        784 * 256 + 256 * 128 + 128 * 10
    }
}

impl SerializeModule for MnistMLP {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
        s.module("fc3", &self.fc3);
    }
}

/// Performance metrics tracker
struct Metrics {
    compile_time: Duration,
    train_start: Instant,
    iter_times: Vec<Duration>,
    loss_avg: ExponentialAverage,
    acc_avg: ExponentialAverage,
}

impl Metrics {
    fn new() -> Self {
        Self {
            compile_time: Duration::ZERO,
            train_start: Instant::now(),
            iter_times: Vec::new(),
            loss_avg: ExponentialAverage::new(2.0),
            acc_avg: ExponentialAverage::new(0.0),
        }
    }

    fn record_iter(&mut self, duration: Duration, loss: f32, acc: f32) {
        self.iter_times.push(duration);
        self.loss_avg.update(loss);
        self.acc_avg.update(acc);
    }

    fn throughput(&self, batch_size: usize) -> f32 {
        if self.iter_times.is_empty() {
            return 0.0;
        }
        let total_time: Duration = self.iter_times.iter().sum();
        let samples = self.iter_times.len() * batch_size;
        samples as f32 / total_time.as_secs_f32()
    }

    fn avg_iter_time(&self) -> Duration {
        if self.iter_times.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.iter_times.iter().sum();
        total / self.iter_times.len() as u32
    }

    fn print_summary(&self, batch_size: usize, total_iters: usize) {
        let train_time = self.train_start.elapsed();

        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    Training Summary                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Final Loss:        {:>8.4}                                  ║", self.loss_avg.value);
        println!("║ Final Train Acc:   {:>7.2}%                                  ║", self.acc_avg.value * 100.0);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Compile Time:      {:>8.2}s                                  ║", self.compile_time.as_secs_f32());
        println!("║ Training Time:     {:>8.2}s                                  ║", train_time.as_secs_f32());
        println!("║ Total Iterations:  {:>8}                                    ║", total_iters);
        println!("║ Avg Iter Time:     {:>8.2}ms                                 ║", self.avg_iter_time().as_secs_f32() * 1000.0);
        println!("║ Throughput:        {:>8.0} samples/sec                       ║", self.throughput(batch_size));
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

fn main() {
    let args = Args::parse();

    print_header(&args);

    // Load MNIST data
    println!("📦 Loading MNIST dataset...");
    let (train_images, train_labels, test_images, test_labels) = load_mnist();
    println!("   {} training images, {} test images", train_labels.len(), test_labels.len());

    // Calculate training parameters
    let max_iters = (train_labels.len() / args.batch_size) * args.epochs;
    let final_lr = args.lr * 0.1;

    // Build model and optimizer
    println!("\n🔧 Building model and optimizer...");
    let mut cx = Graph::new();
    let model = MnistMLP::new(&mut cx).init_rand();
    println!("   Model: MLP with {} parameters", model.param_count());

    // Input: (batch, 1, 28, 28), Target: (batch, 10) one-hot
    let mut input = cx.tensor((args.batch_size, 1, 28, 28));
    let mut target = cx.tensor((args.batch_size, 10));

    let mut output = model.forward(input).retrieve();
    let mut loss = cross_entropy_with_logits_loss(output, target).retrieve();

    // Get model parameters and compute gradients
    let mut weights = params(&model);
    let grads = cx.compile(Autograd::new(&weights, loss), ());

    // Adam optimizer
    let adam_config = AdamConfig::default().lr(args.lr);
    let (mut new_weights, lr, mut adam_state) = adam_on_graph(&mut cx, &weights, &grads, adam_config);

    // Learning rate scheduler
    let scheduler = CosineAnnealingLR::new(args.lr, final_lr, max_iters);

    // Keep tensors across iterations
    cx.keep_tensors(adam_state.all_tensors());
    cx.keep_tensors(new_weights.clone());
    cx.keep_tensors(&weights);

    // Compile to backend
    println!("\n⚡ Compiling computation graph...");
    let compile_start = Instant::now();

    compile_graph(
        &mut cx,
        args.optimal,
        &mut input,
        &mut target,
        &mut loss,
        &mut output,
        &mut weights,
        &mut new_weights,
        &mut adam_state,
    );

    let mut metrics = Metrics::new();
    metrics.compile_time = compile_start.elapsed();
    println!("   Compilation took {:.2}s", metrics.compile_time.as_secs_f32());

    // Training loop
    println!("\n🚀 Training...");
    println!("   Epochs: {}, Batch size: {}, Total iterations: {}", args.epochs, args.batch_size, max_iters);
    println!();

    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..train_labels.len()).collect();
    let mut iter = 0;
    metrics.train_start = Instant::now();

    for epoch in 0..args.epochs {
        indices.shuffle(&mut rng);

        for batch_start in (0..train_labels.len()).step_by(args.batch_size) {
            if batch_start + args.batch_size > train_labels.len() {
                break;
            }

            let iter_start = Instant::now();

            scheduler.step(iter, lr);

            let batch_indices = &indices[batch_start..batch_start + args.batch_size];
            let (batch_images, batch_targets) = prepare_batch(batch_indices, &train_images, &train_labels, args.batch_size);

            input.set(batch_images);
            target.set(batch_targets.clone());

            cx.execute();

            transfer_data_same_graph(new_weights.clone(), &weights, &mut cx);
            adam_state.step(&mut cx);

            let predictions = output.data();
            let batch_acc = compute_batch_accuracy(&predictions, &batch_targets, args.batch_size);
            let batch_loss = loss.data()[0];

            metrics.record_iter(iter_start.elapsed(), batch_loss, batch_acc);

            loss.drop();
            output.drop();

            if iter % 100 == 0 {
                print_progress(epoch + 1, iter, &metrics, scheduler.get_lr(iter), args.verbose);
            }
            iter += 1;
        }
    }

    metrics.print_summary(args.batch_size, iter);

    // Evaluate on test set
    println!("\n📊 Evaluating on test set...");
    let test_acc = evaluate_test_set(&mut cx, &mut input, &mut output, &test_images, &test_labels, args.batch_size);
    println!("   Test Accuracy: {:.2}%", test_acc * 100.0);
}

fn print_header(args: &Args) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              MNIST Training with Luminal                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    #[cfg(feature = "metal")]
    let backend = if args.optimal { "Metal (search-optimized)" } else { "Metal (fast)" };
    #[cfg(feature = "cuda")]
    let backend = if args.optimal { "CUDA (search-optimized)" } else { "CUDA (fast)" };
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let backend = "CPU";

    println!("║ Backend: {:<51} ║", backend);
    println!("║ Epochs: {:<52} ║", args.epochs);
    println!("║ Batch Size: {:<48} ║", args.batch_size);
    println!("║ Learning Rate: {:<45} ║", args.lr);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_progress(epoch: usize, iter: usize, metrics: &Metrics, lr: f32, verbose: bool) {
    if verbose {
        println!(
            "Epoch {:2} | Iter {:5} | Loss: {:.4} | Acc: {:5.1}% | LR: {:.6} | {:.1}ms/iter",
            epoch, iter, metrics.loss_avg.value, metrics.acc_avg.value * 100.0, lr,
            metrics.avg_iter_time().as_secs_f32() * 1000.0
        );
    } else {
        println!(
            "Epoch {:2} | Iter {:5} | Loss: {:.4} | Acc: {:5.1}%",
            epoch, iter, metrics.loss_avg.value, metrics.acc_avg.value * 100.0
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_graph(
    cx: &mut Graph,
    optimal: bool,
    input: &mut GraphTensor,
    target: &mut GraphTensor,
    loss: &mut GraphTensor,
    output: &mut GraphTensor,
    weights: &mut Vec<petgraph::graph::NodeIndex>,
    new_weights: &mut Vec<petgraph::graph::NodeIndex>,
    adam_state: &mut luminal::training::AdamState,
) {
    let remap = (input, target, loss, output, weights, new_weights, adam_state);

    #[cfg(feature = "metal")]
    {
        if optimal {
            #[cfg(feature = "search")]
            {
                cx.compile(
                    (GenericCompiler::default(), luminal_metal::UnifiedMetalCompiler::<f32>::optimal()),
                    remap,
                );
            }
            #[cfg(not(feature = "search"))]
            {
                eprintln!("Warning: --optimal requires 'search' feature. Using fast mode.");
                cx.compile(
                    (GenericCompiler::default(), luminal_metal::MetalCompiler::<f32>::default()),
                    remap,
                );
            }
        } else {
            cx.compile(
                (GenericCompiler::default(), luminal_metal::MetalCompiler::<f32>::default()),
                remap,
            );
        }
    }

    #[cfg(feature = "cuda")]
    {
        if optimal {
            #[cfg(feature = "search")]
            {
                cx.compile(
                    (GenericCompiler::default(), luminal_cuda::UnifiedCudaCompiler::<f32>::optimal()),
                    remap,
                );
            }
            #[cfg(not(feature = "search"))]
            {
                eprintln!("Warning: --optimal requires 'search' feature. Using fast mode.");
                cx.compile(
                    (GenericCompiler::default(), luminal_cuda::CudaCompiler::<f32>::default()),
                    remap,
                );
            }
        } else {
            cx.compile(
                (GenericCompiler::default(), luminal_cuda::CudaCompiler::<f32>::default()),
                remap,
            );
        }
    }

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = optimal;
        cx.compile(GenericCompiler::default(), remap);
    }
}

fn load_mnist() -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path("examples/mnist_cnn/data")
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let train_images: Vec<f32> = trn_img.iter().map(|&x| x as f32 / 255.0).collect();
    let test_images: Vec<f32> = tst_img.iter().map(|&x| x as f32 / 255.0).collect();

    (train_images, trn_lbl, test_images, tst_lbl)
}

fn prepare_batch(indices: &[usize], images: &[f32], labels: &[u8], batch_size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut batch_images = Vec::with_capacity(batch_size * 784);
    let mut batch_targets = Vec::with_capacity(batch_size * 10);

    for &idx in indices {
        let img_start = idx * 784;
        batch_images.extend_from_slice(&images[img_start..img_start + 784]);

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

        let pred_class = predictions[pred_start..pred_start + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

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
        let (batch_images, batch_targets) = prepare_batch(&indices, test_images, test_labels, batch_size);

        input.set(batch_images);
        cx.execute();

        let predictions = output.data();
        let batch_correct = (compute_batch_accuracy(&predictions, &batch_targets, batch_size) * batch_size as f32) as usize;

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
        Self { beta: 0.99, moment: 0., value: initial, t: 0 }
    }

    pub fn update(&mut self, value: f32) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value;
        self.value = self.moment / (1. - f32::powi(self.beta, self.t));
    }
}
