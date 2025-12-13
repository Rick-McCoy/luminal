//! Train a CNN on MNIST using luminal
//!
//! This example demonstrates:
//! - Building a LeNet-5 style CNN with multiple Conv2D layers and pooling
//! - Using the Adam optimizer with cosine annealing learning rate schedule
//! - Training on the MNIST dataset
//! - GPU acceleration with Metal (macOS) or CUDA
//! - Optional search-based optimization via `luminal::search`
//!
//! The LeNet CNN achieves ~98%+ test accuracy in 3 epochs.
//!
//! ## Architecture (LeNet-5 style)
//!
//! ```text
//! Input (1, 28, 28)
//!   ↓
//! Conv2D(1→6, 5x5, padding=2) → ReLU → AvgPool2D(2x2) → (6, 14, 14)
//!   ↓
//! Conv2D(6→16, 5x5) → ReLU → AvgPool2D(2x2) → (16, 5, 5)
//!   ↓
//! Flatten → Linear(400→120) → ReLU
//!   ↓
//! Linear(120→84) → ReLU
//!   ↓
//! Linear(84→10)
//! ```
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
//! LeNet CNN (default, fast compilation):
//! ```bash
//! cargo run -p mnist_cnn --release --features metal
//! cargo run -p mnist_cnn --release --features cuda
//! ```
//!
//! LeNet CNN with search-based optimization:
//! ```bash
//! cargo run -p mnist_cnn --release --features metal,search -- --optimal
//! cargo run -p mnist_cnn --release --features cuda,search -- --optimal
//! ```
//!
//! Simple CNN (single conv layer):
//! ```bash
//! cargo run -p mnist_cnn --release --features metal -- --simple
//! ```
//!
//! MLP (for comparison):
//! ```bash
//! cargo run -p mnist_cnn --release --features metal -- --mlp
//! ```

use std::time::{Duration, Instant};

use clap::Parser;
use luminal::nn::{AvgPool2D, Conv2D, Linear};
use luminal::prelude::*;
use luminal::training::{
    adam_on_graph, cross_entropy_with_logits_loss, AdamConfig, Autograd, CosineAnnealingLR,
    LRScheduler,
};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Train a CNN on MNIST
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Number of training epochs
    #[arg(short, long, default_value_t = 1)]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Use MLP instead of CNN (for comparison)
    #[arg(long)]
    mlp: bool,

    /// Use simple single-layer CNN instead of LeNet
    #[arg(long)]
    simple: bool,

    /// Print detailed performance metrics
    #[arg(short, long)]
    verbose: bool,

    /// Use search-based optimization (requires 'search' feature)
    ///
    /// This uses luminal::search with egglog equality saturation to find
    /// optimal kernel implementations. Slower compile time, potentially
    /// faster runtime.
    #[arg(long)]
    optimal: bool,
}

/// LeNet-5 style CNN for MNIST
///
/// Architecture:
/// ```text
/// Input (1, 28, 28)
///   ↓
/// Conv2D(1→6, 5x5, padding=2) → ReLU → AvgPool2D(2x2) → (6, 14, 14)
///   ↓
/// Conv2D(6→16, 5x5) → ReLU → AvgPool2D(2x2) → (16, 5, 5)
///   ↓
/// Flatten → Linear(400→120) → ReLU
///   ↓
/// Linear(120→84) → ReLU
///   ↓
/// Linear(84→10)
/// ```
struct LeNet {
    conv1: Conv2D,
    conv2: Conv2D,
    pool: AvgPool2D,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl LeNet {
    fn new(cx: &mut Graph) -> Self {
        Self {
            // (1, 28, 28) -> (6, 28, 28) with padding=2, then pool -> (6, 14, 14)
            conv1: Conv2D::with_padding(1, 6, (5, 5), (1, 1), (1, 1), (2, 2), false, cx),
            // (6, 14, 14) -> (16, 10, 10), then pool -> (16, 5, 5)
            conv2: Conv2D::new(6, 16, (5, 5), (1, 1), (1, 1), false, cx),
            // Pooling layer (shared)
            pool: AvgPool2D::new((2, 2), (2, 2)),
            // 16 * 5 * 5 = 400 -> 120
            fc1: Linear::new(400, 120, false, cx),
            // 120 -> 84
            fc2: Linear::new(120, 84, false, cx),
            // 84 -> 10
            fc3: Linear::new(84, 10, false, cx),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let batch = x.dims()[0];

        // Conv block 1: Conv -> ReLU -> Pool
        let x = self.conv1.forward(x).relu(); // (batch, 6, 28, 28)
        let x = self.pool.forward(x); // (batch, 6, 14, 14)

        // Conv block 2: Conv -> ReLU -> Pool
        let x = self.conv2.forward(x).relu(); // (batch, 16, 10, 10)
        let x = self.pool.forward(x); // (batch, 16, 5, 5)

        // Flatten and fully connected layers
        let x = x.reshape((batch, 400));
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x)
    }

    fn init_rand(self) -> Self {
        Self {
            conv1: init_conv(self.conv1),
            conv2: init_conv(self.conv2),
            pool: self.pool,
            fc1: init_linear(self.fc1),
            fc2: init_linear(self.fc2),
            fc3: init_linear(self.fc3),
        }
    }

    fn param_count(&self) -> usize {
        // conv1: 1 * 6 * 5 * 5 = 150
        // conv2: 6 * 16 * 5 * 5 = 2400
        // fc1: 400 * 120 = 48000
        // fc2: 120 * 84 = 10080
        // fc3: 84 * 10 = 840
        150 + 2400 + 48000 + 10080 + 840
    }
}

impl SerializeModule for LeNet {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.module("conv1", &self.conv1);
        s.module("conv2", &self.conv2);
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
        s.module("fc3", &self.fc3);
    }
}

/// Simple CNN for MNIST using a single strided convolution
/// (Legacy architecture, use --simple flag)
struct SimpleCNN {
    conv1: Conv2D,
    fc: Linear,
}

impl SimpleCNN {
    fn new(cx: &mut Graph) -> Self {
        Self {
            // (1, 28, 28) -> (8, 12, 12)
            conv1: Conv2D::new(1, 8, (5, 5), (2, 2), (1, 1), false, cx),
            // 8 * 12 * 12 = 1152 -> 10
            fc: Linear::new(1152, 10, false, cx),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let batch = x.dims()[0];
        // Conv layer with strided downsampling
        let x = self.conv1.forward(x).relu();
        // Flatten and classify
        let x = x.reshape((batch, 1152));
        self.fc.forward(x)
    }

    fn init_rand(self) -> Self {
        Self {
            conv1: init_conv(self.conv1),
            fc: init_linear(self.fc),
        }
    }

    fn param_count(&self) -> usize {
        // conv1: 1 * 8 * 5 * 5 = 200
        // fc: 1152 * 10 = 11520
        200 + 11520
    }
}

impl SerializeModule for SimpleCNN {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.module("conv1", &self.conv1);
        s.module("fc", &self.fc);
    }
}

/// MLP for comparison
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
            fc1: init_linear(self.fc1),
            fc2: init_linear(self.fc2),
            fc3: init_linear(self.fc3),
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

fn init_conv(conv: Conv2D) -> Conv2D {
    let fan_in = conv.weight.dims()[1].to_usize().unwrap();
    let fan_out = conv.weight.dims()[0].to_usize().unwrap();
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let data: Vec<f32> = (0..conv.weight.shape.n_elements().to_usize().unwrap())
        .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * bound)
        .collect();
    conv.weight.set(data);
    conv
}

fn init_linear(linear: Linear) -> Linear {
    let fan_in = linear.weight.dims()[1].to_usize().unwrap();
    let fan_out = linear.weight.dims()[0].to_usize().unwrap();
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let data: Vec<f32> = (0..linear.weight.shape.n_elements().to_usize().unwrap())
        .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * bound)
        .collect();
    linear.weight.set(data);
    linear
}

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
        println!(
            "║ Final Loss:        {:>8.4}                                  ║",
            self.loss_avg.value
        );
        println!(
            "║ Final Train Acc:   {:>7.2}%                                  ║",
            self.acc_avg.value * 100.0
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Compile Time:      {:>8.2}s                                  ║",
            self.compile_time.as_secs_f32()
        );
        println!(
            "║ Training Time:     {:>8.2}s                                  ║",
            train_time.as_secs_f32()
        );
        println!(
            "║ Total Iterations:  {:>8}                                    ║",
            total_iters
        );
        println!(
            "║ Avg Iter Time:     {:>8.2}ms                                 ║",
            self.avg_iter_time().as_secs_f32() * 1000.0
        );
        println!(
            "║ Throughput:        {:>8.0} samples/sec                       ║",
            self.throughput(batch_size)
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

fn main() {
    let args = Args::parse();

    print_header(&args);

    println!("📦 Loading MNIST dataset...");
    let (train_images, train_labels, test_images, test_labels) = load_mnist();
    println!(
        "   {} training images, {} test images",
        train_labels.len(),
        test_labels.len()
    );

    let max_iters = (train_labels.len() / args.batch_size) * args.epochs;
    let final_lr = args.lr * 0.1;

    println!("\n🔧 Building model and optimizer...");
    let mut cx = Graph::new();

    let mut input = cx.tensor((args.batch_size, 1, 28, 28));
    let mut target = cx.tensor((args.batch_size, 10));

    let (mut output, mut weights, model_name, param_count) = if args.mlp {
        let model = MnistMLP::new(&mut cx).init_rand();
        let out = model.forward(input).retrieve();
        let w = params(&model);
        (out, w, "MLP", model.param_count())
    } else if args.simple {
        let model = SimpleCNN::new(&mut cx).init_rand();
        let out = model.forward(input).retrieve();
        let w = params(&model);
        (out, w, "Simple CNN", model.param_count())
    } else {
        let model = LeNet::new(&mut cx).init_rand();
        let out = model.forward(input).retrieve();
        let w = params(&model);
        (out, w, "LeNet", model.param_count())
    };

    println!("   Model: {} with {} parameters", model_name, param_count);

    let mut loss = cross_entropy_with_logits_loss(output, target).retrieve();
    let grads = cx.compile(Autograd::new(&weights, loss), ());

    let adam_config = AdamConfig::default().lr(args.lr);
    let (mut new_weights, lr, mut adam_state) =
        adam_on_graph(&mut cx, &weights, &grads, adam_config);

    let scheduler = CosineAnnealingLR::new(args.lr, final_lr, max_iters);

    cx.keep_tensors(adam_state.all_tensors());
    cx.keep_tensors(new_weights.clone());
    cx.keep_tensors(&weights);

    println!("\n⚡ Compiling computation graph...");
    let compile_start = Instant::now();

    let remap = (
        &mut input,
        &mut target,
        &mut loss,
        &mut output,
        &mut weights,
        &mut new_weights,
        &mut adam_state,
    );

    #[cfg(feature = "metal")]
    {
        if args.optimal {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_metal::UnifiedMetalCompiler::<f32>::optimal(),
                ),
                remap,
            );
        } else {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_metal::UnifiedMetalCompiler::<f32>::fast(),
                ),
                remap,
            );
        }
    }
    #[cfg(feature = "cuda")]
    {
        if args.optimal {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_cuda::UnifiedCudaCompiler::<f32>::optimal(),
                ),
                remap,
            );
        } else {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_cuda::UnifiedCudaCompiler::<f32>::fast(),
                ),
                remap,
            );
        }
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = args.optimal; // Suppress unused warning
        cx.compile(GenericCompiler::default(), remap);
    }

    let mut metrics = Metrics::new();
    metrics.compile_time = compile_start.elapsed();
    println!(
        "   Compilation took {:.2}s",
        metrics.compile_time.as_secs_f32()
    );

    println!("\n🚀 Training...");
    println!(
        "   Epochs: {}, Batch size: {}, Total iterations: {}",
        args.epochs, args.batch_size, max_iters
    );
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
            let (batch_images, batch_targets) =
                prepare_batch(batch_indices, &train_images, &train_labels, args.batch_size);

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
                print_progress(
                    epoch + 1,
                    iter,
                    &metrics,
                    scheduler.get_lr(iter),
                    args.verbose,
                );
            }
            iter += 1;
        }
    }

    metrics.print_summary(args.batch_size, iter);

    println!("\n📊 Evaluating on test set...");
    let test_acc = evaluate_test_set(
        &mut cx,
        &mut input,
        &mut output,
        &test_images,
        &test_labels,
        args.batch_size,
    );
    println!("   Test Accuracy: {:.2}%", test_acc * 100.0);
}

fn print_header(args: &Args) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              MNIST Training with Luminal                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    #[cfg(feature = "metal")]
    let backend = "Metal";
    #[cfg(feature = "cuda")]
    let backend = "CUDA";
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let backend = "CPU";
    let model = if args.mlp {
        "MLP"
    } else if args.simple {
        "Simple CNN"
    } else {
        "LeNet"
    };
    let compile_mode = if args.optimal {
        "Optimal (search)"
    } else {
        "Fast"
    };
    println!("║ Backend: {:<51} ║", backend);
    println!("║ Model: {:<53} ║", model);
    println!("║ Compilation: {:<47} ║", compile_mode);
    println!("║ Epochs: {:<52} ║", args.epochs);
    println!("║ Batch Size: {:<48} ║", args.batch_size);
    println!("║ Learning Rate: {:<45} ║", args.lr);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_progress(epoch: usize, iter: usize, metrics: &Metrics, lr: f32, verbose: bool) {
    if verbose {
        println!(
            "Epoch {:2} | Iter {:5} | Loss: {:.4} | Acc: {:5.1}% | LR: {:.6} | {:.1}ms/iter",
            epoch,
            iter,
            metrics.loss_avg.value,
            metrics.acc_avg.value * 100.0,
            lr,
            metrics.avg_iter_time().as_secs_f32() * 1000.0
        );
    } else {
        println!(
            "Epoch {:2} | Iter {:5} | Loss: {:.4} | Acc: {:5.1}%",
            epoch,
            iter,
            metrics.loss_avg.value,
            metrics.acc_avg.value * 100.0
        );
    }
}

fn load_mnist() -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
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

    let train_images: Vec<f32> = trn_img.iter().map(|&x| x as f32 / 255.0).collect();
    let test_images: Vec<f32> = tst_img.iter().map(|&x| x as f32 / 255.0).collect();

    (train_images, trn_lbl, test_images, tst_lbl)
}

fn prepare_batch(
    indices: &[usize],
    images: &[f32],
    labels: &[u8],
    batch_size: usize,
) -> (Vec<f32>, Vec<f32>) {
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
        let (batch_images, batch_targets) =
            prepare_batch(&indices, test_images, test_labels, batch_size);

        input.set(batch_images);
        cx.execute();

        let predictions = output.data();
        let batch_correct = (compute_batch_accuracy(&predictions, &batch_targets, batch_size)
            * batch_size as f32) as usize;

        total_correct += batch_correct;
        total_samples += batch_size;
        output.drop();
    }

    total_correct as f32 / total_samples as f32
}

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
