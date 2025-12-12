//! Train an MLP to add 4-bit numbers
//!
//! This example demonstrates:
//! - Building a simple MLP with Linear layers and Swish activation
//! - Using the Adam optimizer with cosine annealing learning rate schedule
//! - Training until accuracy threshold is reached
//! - GPU acceleration with Metal (macOS) or CUDA
//! - Optional search-based kernel optimization
//!
//! ## Running
//!
//! Fast mode (default):
//! ```bash
//! cargo run -p train_math_net --release --features metal
//! cargo run -p train_math_net --release --features cuda
//! ```
//!
//! Optimal mode (search-based compilation):
//! ```bash
//! cargo run -p train_math_net --release --features metal,search -- --optimal
//! cargo run -p train_math_net --release --features cuda,search -- --optimal
//! ```

use std::time::{Duration, Instant};

use clap::Parser;
use luminal::prelude::*;
use luminal::nn::{Linear, Swish};
use luminal::training::{
    adam_on_graph, mse_loss, AdamConfig, Autograd, CosineAnnealingLR, LRScheduler,
};
use rand::{rngs::ThreadRng, thread_rng, Rng};

/// Train an MLP to add 4-bit numbers
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Maximum training iterations
    #[arg(short, long, default_value_t = 2000)]
    max_iters: usize,

    /// Target accuracy to stop training (0.0 - 1.0)
    #[arg(long, default_value_t = 0.995)]
    target_acc: f32,

    /// Initial learning rate
    #[arg(long, default_value_t = 0.01)]
    lr: f32,

    /// Use search-based optimal compilation
    #[arg(long)]
    optimal: bool,

    /// Print detailed performance metrics
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    print_header(&args);

    // Build computation graph
    println!("\n🔧 Building model...");
    let mut cx = Graph::new();
    let model = (
        Linear::new(8, 16, false, &mut cx).init_rand(),
        Swish,
        Linear::new(16, 16, false, &mut cx).init_rand(),
        Swish,
        Linear::new(16, 5, false, &mut cx).init_rand(),
    );

    let mut input = cx.tensor(8);
    let mut target = cx.tensor(5);
    let mut output = model.forward(input).retrieve();
    let mut loss = mse_loss(output, target).retrieve();

    // Get model parameters and compute gradients
    let mut weights = params(&model);
    let grads = cx.compile(Autograd::new(&weights, loss), ());

    // Adam optimizer with cosine annealing
    let adam_config = AdamConfig::default().lr(args.lr);
    let (mut new_weights, lr, adam_state) = adam_on_graph(&mut cx, &weights, &grads, adam_config);

    // Keep tensors across iterations
    cx.keep_tensors(adam_state.all_tensors());
    cx.keep_tensors(new_weights.clone());
    cx.keep_tensors(&weights);

    // Learning rate scheduler
    let scheduler = CosineAnnealingLR::new(args.lr, args.lr * 0.01, args.max_iters);

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
    );

    let compile_time = compile_start.elapsed();
    println!("   Compilation took {:.2}s", compile_time.as_secs_f32());

    // Training loop
    println!("\n🚀 Training...");
    println!("   Max iterations: {}, Target accuracy: {:.1}%",
             args.max_iters, args.target_acc * 100.0);
    println!();

    let mut rng = thread_rng();
    let (mut loss_avg, mut acc_avg) = (ExponentialAverage::new(1.0), ExponentialAverage::new(0.0));
    let mut iter = 0;
    let mut iter_times = Vec::new();
    let start = Instant::now();

    while acc_avg.value < args.target_acc && iter < args.max_iters {
        let iter_start = Instant::now();

        // Update learning rate
        scheduler.step(iter, lr);

        // Generate problem
        let (problem, answer) = make_problem(&mut rng);
        input.set(problem);
        target.set(answer);

        // Execute graph (forward + backward + optimizer step)
        cx.execute();

        // Update weights and optimizer state
        transfer_data_same_graph(new_weights.clone(), &weights, &mut cx);
        adam_state.step(&mut cx);

        // Record metrics
        iter_times.push(iter_start.elapsed());
        loss_avg.update(loss.data()[0]);
        loss.drop();

        acc_avg.update(
            output
                .data()
                .into_iter()
                .zip(answer)
                .filter(|(a, b)| (a - b).abs() < 0.5)
                .count() as f32
                / 5.0,
        );
        output.drop();

        if iter % 100 == 0 {
            print_progress(iter, &loss_avg, &acc_avg, &scheduler, &iter_times, args.verbose);
        }
        iter += 1;
    }

    // Print summary
    print_summary(iter, &loss_avg, &acc_avg, compile_time, start.elapsed(), &iter_times);
}

fn print_header(args: &Args) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║             4-bit Adder Training with Luminal                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    #[cfg(feature = "metal")]
    let backend = if args.optimal { "Metal (search-optimized)" } else { "Metal (fast)" };
    #[cfg(feature = "cuda")]
    let backend = if args.optimal { "CUDA (search-optimized)" } else { "CUDA (fast)" };
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let backend = "CPU";

    println!("║ Backend: {:<51} ║", backend);
    println!("║ Max Iterations: {:<44} ║", args.max_iters);
    println!("║ Target Accuracy: {:>5.1}%{:<38} ║", args.target_acc * 100.0, "");
    println!("║ Learning Rate: {:<45} ║", args.lr);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_progress(
    iter: usize,
    loss_avg: &ExponentialAverage,
    acc_avg: &ExponentialAverage,
    scheduler: &CosineAnnealingLR,
    iter_times: &[Duration],
    verbose: bool,
) {
    let avg_time = if iter_times.is_empty() {
        0.0
    } else {
        iter_times.iter().sum::<Duration>().as_secs_f32() * 1000.0 / iter_times.len() as f32
    };

    if verbose {
        println!(
            "Iter {:4} | Loss: {:.4} | Acc: {:5.1}% | LR: {:.6} | {:.2}ms/iter",
            iter,
            loss_avg.value,
            acc_avg.value * 100.0,
            scheduler.get_lr(iter),
            avg_time
        );
    } else {
        println!(
            "Iter {:4} | Loss: {:.4} | Acc: {:5.1}%",
            iter,
            loss_avg.value,
            acc_avg.value * 100.0
        );
    }
}

fn print_summary(
    iters: usize,
    loss_avg: &ExponentialAverage,
    acc_avg: &ExponentialAverage,
    compile_time: Duration,
    train_time: Duration,
    iter_times: &[Duration],
) {
    let avg_iter = if iter_times.is_empty() {
        0.0
    } else {
        iter_times.iter().sum::<Duration>().as_micros() as f32 / iter_times.len() as f32
    };

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Training Summary                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Converged in {:>6} iterations                               ║", iters);
    println!("║ Final Loss:        {:>8.4}                                  ║", loss_avg.value);
    println!("║ Final Accuracy:    {:>7.2}%                                  ║", acc_avg.value * 100.0);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Compile Time:      {:>8.2}s                                  ║", compile_time.as_secs_f32());
    println!("║ Training Time:     {:>8.2}s                                  ║", train_time.as_secs_f32());
    println!("║ Avg Iter Time:     {:>8.0}µs                                 ║", avg_iter);
    println!("╚══════════════════════════════════════════════════════════════╝");
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
) {
    let remap = (input, target, loss, output, weights, new_weights);

    #[cfg(feature = "metal")]
    {
        if optimal {
            #[cfg(feature = "search")]
            {
                cx.compile(
                    (
                        GenericCompiler::default(),
                        luminal_metal::UnifiedMetalCompiler::<f32>::optimal(),
                    ),
                    remap,
                );
            }
            #[cfg(not(feature = "search"))]
            {
                eprintln!("Warning: --optimal requires 'search' feature. Using fast mode.");
                cx.compile(
                    (
                        GenericCompiler::default(),
                        luminal_metal::MetalCompiler::<f32>::default(),
                    ),
                    remap,
                );
            }
        } else {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_metal::MetalCompiler::<f32>::default(),
                ),
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
                    (
                        GenericCompiler::default(),
                        luminal_cuda::UnifiedCudaCompiler::<f32>::optimal(),
                    ),
                    remap,
                );
            }
            #[cfg(not(feature = "search"))]
            {
                eprintln!("Warning: --optimal requires 'search' feature. Using fast mode.");
                cx.compile(
                    (
                        GenericCompiler::default(),
                        luminal_cuda::CudaCompiler::<f32>::default(),
                    ),
                    remap,
                );
            }
        } else {
            cx.compile(
                (
                    GenericCompiler::default(),
                    luminal_cuda::CudaCompiler::<f32>::default(),
                ),
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

/// Generate a training example: two 4-bit numbers to add
fn make_problem(rng: &mut ThreadRng) -> ([f32; 8], [f32; 5]) {
    fn get_lower_bits(byte: u8, bits: usize, slice: &mut [f32]) {
        #[allow(clippy::needless_range_loop)]
        for i in 0..bits {
            slice[i] = if (byte >> i) & 1 == 1 { 1.0 } else { 0.0 };
        }
    }

    let (n1, n2): (u8, u8) = (rng.gen_range(0..16), rng.gen_range(0..16));
    let ans = n1.wrapping_add(n2);
    let mut p = [0.; 8];
    get_lower_bits(n1, 4, &mut p);
    get_lower_bits(n2, 4, &mut p[4..]);
    let mut a = [0.; 5];
    get_lower_bits(ans, 5, &mut a);
    (p, a)
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
            beta: 0.999,
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
