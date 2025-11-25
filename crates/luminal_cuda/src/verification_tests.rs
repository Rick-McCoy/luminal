//! Verification tests for CUDA backend
//!
//! These tests verify that CUDA-compiled operations produce results
//! consistent with the CPU reference implementation.

#[cfg(test)]
mod tests {
    use luminal::prelude::*;
    use luminal_nn::{
        AvgPool2D, BatchNorm1d, BatchNorm2d, Conv2D, Dropout, GlobalAvgPool2D, GlobalMaxPool2D,
        Linear, MaxPool1D, MaxPool2D, RMSNorm, GRU, LSTM,
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::CudaCompiler;
    luminal::test_imports!();

    // Helper to run both CPU and CUDA versions and compare
    fn compare_with_cpu<F>(build_fn: F, tolerance: f32)
    where
        F: Fn(&mut Graph) -> GraphTensor,
    {
        // Run on CPU
        let mut cx_cpu = Graph::new();
        let out_cpu = build_fn(&mut cx_cpu).retrieve();
        cx_cpu.execute();
        let cpu_result = out_cpu.data();

        // Run on CUDA
        let mut cx_cuda = Graph::new();
        let mut out_cuda = build_fn(&mut cx_cuda).retrieve();
        cx_cuda.compile(CudaCompiler::<f32>::default(), &mut out_cuda);
        cx_cuda.execute();
        let cuda_result = out_cuda.data();

        // Compare
        assert_eq!(
            cpu_result.len(),
            cuda_result.len(),
            "Output lengths differ: CPU={}, CUDA={}",
            cpu_result.len(),
            cuda_result.len()
        );

        for (i, (cpu, cuda)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
            let diff = (cpu - cuda).abs();
            assert!(
                diff < tolerance,
                "Mismatch at index {}: CPU={}, CUDA={}, diff={}",
                i,
                cpu,
                cuda,
                diff
            );
        }
    }

    // ==================== RMSNorm Tests ====================

    #[test]
    fn test_rmsnorm_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 8)).set(input_data.clone());
                let norm = RMSNorm::new(8, 1e-5, cx);
                norm.forward(input)
            },
            1e-4,
        );
    }

    #[test]
    fn test_rmsnorm_batch_cuda() {
        let mut rng = StdRng::seed_from_u64(123);
        let input_data: Vec<f32> = (0..128).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 8, 8)).set(input_data.clone());
                let norm = RMSNorm::new(8, 1e-5, cx);
                norm.forward(input)
            },
            1e-4,
        );
    }

    // ==================== BatchNorm Tests ====================

    #[test]
    fn test_batchnorm1d_inference_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..24).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 6)).set(input_data.clone());
                let mut bn = BatchNorm1d::default_new(6, cx);
                bn.set_training(false);
                bn.running_mean.set(vec![0.0; 6]);
                bn.running_var.set(vec![1.0; 6]);
                bn.forward(input)
            },
            1e-5,
        );
    }

    #[test]
    fn test_batchnorm2d_inference_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..96).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 3, 4, 4)).set(input_data.clone());
                let mut bn = BatchNorm2d::default_new(3, cx);
                bn.set_training(false);
                bn.running_mean.set(vec![0.0; 3]);
                bn.running_var.set(vec![1.0; 3]);
                bn.forward(input)
            },
            1e-5,
        );
    }

    // ==================== Pooling Tests ====================

    #[test]
    fn test_maxpool2d_cuda() {
        let input_data: Vec<f32> = (0..64).map(|x| x as f32).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((1, 1, 8, 8)).set(input_data.clone());
                let pool = MaxPool2D::new((2, 2), (2, 2));
                pool.forward(input)
            },
            1e-6,
        );
    }

    #[test]
    fn test_maxpool2d_multichannel_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..192).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 3, 4, 8)).set(input_data.clone());
                let pool = MaxPool2D::new((2, 2), (2, 2));
                pool.forward(input)
            },
            1e-6,
        );
    }

    #[test]
    fn test_maxpool1d_cuda() {
        let input_data: Vec<f32> = (0..24).map(|x| x as f32).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 3, 4)).set(input_data.clone());
                let pool = MaxPool1D::new(2, 2);
                pool.forward(input)
            },
            1e-6,
        );
    }

    #[test]
    fn test_global_avg_pool_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..128).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 4, 4, 4)).set(input_data.clone());
                let pool = GlobalAvgPool2D::new(true);
                pool.forward(input)
            },
            1e-5,
        );
    }

    #[test]
    fn test_global_max_pool_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..128).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 4, 4, 4)).set(input_data.clone());
                let pool = GlobalMaxPool2D::new(true);
                pool.forward(input)
            },
            1e-6,
        );
    }

    #[test]
    fn test_avgpool2d_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..64).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((1, 1, 8, 8)).set(input_data.clone());
                let pool = AvgPool2D::new((2, 2), (2, 2));
                pool.forward(input)
            },
            1e-5,
        );
    }

    // ==================== Conv2D with Padding Tests ====================

    #[test]
    fn test_conv2d_with_padding_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..64).map(|_| rng.random()).collect();
        let weight_data: Vec<f32> = (0..18).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((1, 2, 4, 8)).set(input_data.clone());
                let conv = Conv2D::with_padding(2, 1, (3, 3), (1, 1), (1, 1), (1, 1), false, cx);
                conv.weight.set(weight_data.clone());
                conv.forward(input)
            },
            1e-4,
        );
    }

    #[test]
    fn test_conv2d_same_padding_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();
        let weight_data: Vec<f32> = (0..18).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((1, 2, 4, 4)).set(input_data.clone());
                let conv = Conv2D::same(2, 1, (3, 3), false, cx);
                conv.weight.set(weight_data.clone());
                conv.forward(input)
            },
            1e-4,
        );
    }

    // ==================== Dropout Tests ====================

    #[test]
    fn test_dropout_inference_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 8)).set(input_data.clone());
                let mut dropout = Dropout::new(0.5, (4, 8), cx);
                dropout.set_training(false);
                dropout.forward(input)
            },
            1e-6,
        );
    }

    // ==================== LSTM Tests ====================

    #[test]
    fn test_lstm_step_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..8).map(|_| rng.random()).collect();
        let weight_ih: Vec<f32> = (0..128).map(|_| rng.random::<f32>() * 0.1).collect();
        let weight_hh: Vec<f32> = (0..256).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 4)).set(input_data.clone());
                let lstm = LSTM::new(4, 8, true, cx);
                lstm.weight_ih.set(weight_ih.clone());
                lstm.weight_hh.set(weight_hh.clone());
                lstm.bias_ih.unwrap().set(vec![0.0; 32]);
                lstm.bias_hh.unwrap().set(vec![0.0; 32]);

                let (h0, c0) = lstm.init_states(2, cx);
                let (h1, _c1) = lstm.forward_step(input, h0, c0);
                h1
            },
            1e-4,
        );
    }

    // ==================== GRU Tests ====================

    #[test]
    fn test_gru_step_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..8).map(|_| rng.random()).collect();
        let weight_ih: Vec<f32> = (0..96).map(|_| rng.random::<f32>() * 0.1).collect();
        let weight_hh: Vec<f32> = (0..192).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((2, 4)).set(input_data.clone());
                let gru = GRU::new(4, 8, true, cx);
                gru.weight_ih.set(weight_ih.clone());
                gru.weight_hh.set(weight_hh.clone());
                gru.bias_ih.unwrap().set(vec![0.0; 24]);
                gru.bias_hh.unwrap().set(vec![0.0; 24]);

                let h0 = gru.init_state(2, cx);
                gru.forward_step(input, h0)
            },
            1e-4,
        );
    }

    // ==================== Linear Layer Tests ====================

    #[test]
    fn test_linear_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();
        let weight_data: Vec<f32> = (0..256).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 8)).set(input_data.clone());
                let linear = Linear::new(8, 32, false, cx);
                linear.weight.set(weight_data.clone());
                linear.forward(input)
            },
            1e-4,
        );
    }

    #[test]
    fn test_linear_with_bias_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();
        let weight_data: Vec<f32> = (0..256).map(|_| rng.random::<f32>() * 0.1).collect();
        let bias_data: Vec<f32> = (0..32).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 8)).set(input_data.clone());
                let linear = Linear::new(8, 32, true, cx);
                linear.weight.set(weight_data.clone());
                linear.bias.unwrap().set(bias_data.clone());
                linear.forward(input)
            },
            1e-4,
        );
    }

    // ==================== Combined Model Tests ====================

    #[test]
    fn test_mlp_cuda() {
        use luminal::module::Module;
        use luminal_nn::{ReLU, Swish};

        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..32).map(|_| rng.random()).collect();
        let w1: Vec<f32> = (0..256).map(|_| rng.random::<f32>() * 0.1).collect();
        let w2: Vec<f32> = (0..512).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((4, 8)).set(input_data.clone());

                let layer1 = Linear::new(8, 32, false, cx);
                layer1.weight.set(w1.clone());

                let layer2 = Linear::new(32, 16, false, cx);
                layer2.weight.set(w2.clone());

                let model = (layer1, Swish, layer2, ReLU);
                model.forward(input)
            },
            1e-3,
        );
    }

    #[test]
    fn test_conv_pool_pipeline_cuda() {
        let mut rng = StdRng::seed_from_u64(42);
        let input_data: Vec<f32> = (0..128).map(|_| rng.random()).collect();
        // Weight size: ch_out * ch_in * kernel_x * kernel_y = 4 * 2 * 3 * 3 = 72
        let weight_data: Vec<f32> = (0..72).map(|_| rng.random::<f32>() * 0.1).collect();

        compare_with_cpu(
            |cx| {
                let input = cx.tensor((1, 2, 8, 8)).set(input_data.clone());

                // Conv2D -> MaxPool -> GlobalAvgPool pipeline
                let conv = Conv2D::new(2, 4, (3, 3), (1, 1), (1, 1), false, cx);
                conv.weight.set(weight_data.clone());

                let conv_out = conv.forward(input);
                let pool = MaxPool2D::new((2, 2), (2, 2));
                let pooled = pool.forward(conv_out);

                let global_pool = GlobalAvgPool2D::new(true);
                global_pool.forward(pooled)
            },
            1e-3,
        );
    }
}
