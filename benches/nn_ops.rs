//! Benchmarks for neural network operations
//!
//! Run with: cargo bench --bench nn_ops

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use luminal::nn::{Conv2D, Linear, MaxPool2D, RMSNorm};
use luminal::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_vec(size: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.random()).collect()
}

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");

    for (in_features, out_features) in [(64, 256), (256, 1024), (1024, 4096)] {
        let batch_size = 32;
        let input_data = random_vec(batch_size * in_features);
        let weight_data = random_vec(in_features * out_features);

        group.throughput(Throughput::Elements(
            (batch_size * in_features * out_features) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", in_features, out_features)),
            &(in_features, out_features),
            |b, &(in_f, out_f)| {
                b.iter(|| {
                    let mut cx = Graph::new();
                    let input = cx.tensor((batch_size, in_f)).set(input_data.clone());
                    let linear = Linear::new(in_f, out_f, false, &mut cx);
                    linear.weight.set(weight_data.clone());
                    let out = linear.forward(input).retrieve();
                    cx.execute();
                    out.data()
                })
            },
        );
    }

    group.finish();
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    for (h, w) in [(32, 32), (64, 64), (128, 128)] {
        let batch_size = 4;
        let in_channels = 16;
        let out_channels = 32;
        let kernel = 3;
        let input_data = random_vec(batch_size * in_channels * h * w);
        let weight_data = random_vec(out_channels * in_channels * kernel * kernel);

        group.throughput(Throughput::Elements((h * w) as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", h, w)),
            &(h, w),
            |b, &(height, width)| {
                b.iter(|| {
                    let mut cx = Graph::new();
                    let input = cx
                        .tensor((batch_size, in_channels, height, width))
                        .set(input_data.clone());
                    let conv = Conv2D::new(
                        in_channels,
                        out_channels,
                        (kernel, kernel),
                        (1, 1),
                        (1, 1),
                        false,
                        &mut cx,
                    );
                    conv.weight.set(weight_data.clone());
                    let out = conv.forward(input).retrieve();
                    cx.execute();
                    out.data()
                })
            },
        );
    }

    group.finish();
}

fn bench_maxpool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxpool2d");

    for size in [32, 64, 128, 256] {
        let batch_size = 4;
        let channels = 32;
        let input_data = random_vec(batch_size * channels * size * size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut cx = Graph::new();
                    let input = cx
                        .tensor((batch_size, channels, s, s))
                        .set(input_data.clone());
                    let pool = MaxPool2D::new((2, 2), (2, 2));
                    let out = pool.forward(input).retrieve();
                    cx.execute();
                    out.data()
                })
            },
        );
    }

    group.finish();
}

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    for dim in [256, 512, 1024, 2048, 4096] {
        let batch_size = 32;
        let input_data = random_vec(batch_size * dim);

        group.throughput(Throughput::Elements((batch_size * dim) as u64));

        group.bench_with_input(BenchmarkId::new("cpu", dim), &dim, |b, &d| {
            b.iter(|| {
                let mut cx = Graph::new();
                let input = cx.tensor((batch_size, d)).set(input_data.clone());
                let norm = RMSNorm::new(d, 1e-5, &mut cx);
                let out = norm.forward(input).retrieve();
                cx.execute();
                out.data()
            })
        });
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [128, 256, 512, 1024] {
        let a_data = random_vec(size * size);
        let b_data = random_vec(size * size);

        group.throughput(Throughput::Elements((2 * size * size * size) as u64)); // 2*N^3 FLOPs

        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", size, size)),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut cx = Graph::new();
                    let a = cx.tensor((s, s)).set(a_data.clone());
                    let b_tensor = cx.tensor((s, s)).set(b_data.clone());
                    let out = a.matmul(b_tensor).retrieve();
                    cx.execute();
                    out.data()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear,
    bench_conv2d,
    bench_maxpool2d,
    bench_rmsnorm,
    bench_matmul
);
criterion_main!(benches);
