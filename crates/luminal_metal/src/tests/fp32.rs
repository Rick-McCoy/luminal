use dfdx::prelude::{Module as DfdxModule, *};
use rand::{rngs::StdRng, Rng, SeedableRng};

use luminal::{module::Module, prelude::*};
use luminal_nn::{Conv1D, Linear, ReLU};

use crate::{binary_test, unary_test, MetalCompiler};
luminal::test_imports!();

unary_test!(|a| a.sin(), |a| a.sin(), test_sin, f32);
unary_test!(|a| a.sqrt(), |a| a.sqrt(), test_sqrt, f32);
unary_test!(|a| a.reciprocal(), |a| a.recip(), test_reciprocal, f32);
unary_test!(|a| a * a, |a| a.clone() * a, test_square, f32);
unary_test!(|a| a.log(), |a| a.ln(), test_log, f32);
unary_test!(|a| a.log2(), |a| a.ln() / 2_f32.ln(), test_log2, f32);
unary_test!(|a| a.exp2(), |a| (a * 2_f32.ln()).exp(), test_exp2, f32);
unary_test!(
    |a| a.softmax(0),
    |a| a.softmax::<DAxis<0>>(),
    test_softmax,
    f32
);
unary_test!(
    |a| a.mean_norm(0).std_norm(0, 1e-5),
    |a| a.normalize::<DAxis<0>>(1e-5),
    test_norm,
    f32
);

binary_test!(|a, b| a + b, |a, b| a + b, test_add, f32);
binary_test!(|a, b| a - b, |a, b| a - b, test_sub, f32);
binary_test!(|a, b| a * b, |a, b| a * b, test_mul, f32);
binary_test!(|a, b| a / b, |a, b| a / b, test_div, f32);
binary_test!(
    |a, b| a % b,
    |a, b| a.clone() - ((a / b.clone()).to_dtype::<i32>().to_dtype::<f32>() * b),
    test_mod,
    f32
);
binary_test!(|a, b| a.minimum(b), |a, b| a.minimum(b), test_min, f32);
binary_test!(|a, b| a.maximum(b), |a, b| a.maximum(b), test_max, f32);

#[test]
fn test_contiguous() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((3, 4)).set(data.clone());
    let mut b = a.permute((1, 0)).reshape((12, 1)).retrieve();
    cx.compile(MetalCompiler::<f32>::default(), &mut b);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<3>, DConst::<4>));
    let d_b = d_a.permute::<Rank2<4, 3>, _>().reshape::<Rank2<12, 1>>();

    assert_close(&b.data(), &d_b.as_vec());
}

// Reduction op tests

#[test]
fn test_sum() {
    let mut cx = Graph::new();
    let data = random_vec(4 * 4096);
    let a = cx.tensor((1, 4, 4096));
    a.set(data.clone());
    let mut b = a.sum(1).retrieve();
    let mut c = a.sum(0).retrieve();
    let mut d = a.sum(2).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<4>, DConst::<4096>));
    let d_b = d_a.clone().sum::<_, DAxis<1>>();
    let d_c = d_a.clone().sum::<_, DAxis<0>>();
    let d_d = d_a.sum::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_max() {
    let mut cx = Graph::new();
    let data = random_vec(12);
    let a = cx.tensor((2, 2, 3));
    a.set(data.clone());
    let mut b = a.max(1).retrieve();
    let mut c = a.max(0).retrieve();
    let mut d = a.max(2).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<2>, DConst::<2>, DConst::<3>));
    let d_b = d_a.clone().max::<_, DAxis<1>>();
    let d_c = d_a.clone().max::<_, DAxis<0>>();
    let d_d = d_a.max::<_, DAxis<2>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_mean() {
    let data = random_vec(40960);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 4096)).set(data.clone());
    let mut b = a.mean(2).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), &mut b);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<4096>));
    let d_b = d_a.mean::<_, DAxis<2>>();
    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_matmul_simple() {
    let mut cx = Graph::new();
    let a_data = random_vec(256 * 256);
    let b_data = random_vec(256 * 256);
    let a = cx.tensor((256, 256)).set(a_data.clone());
    let b = cx.tensor((256, 256)).set(b_data.clone());
    let mut c = a.matmul(b).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<256>, DConst::<256>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<256>, DConst::<256>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let a_data = random_vec(512 * 512);
    let b_data = random_vec(512 * 512);
    let a = cx.tensor((512, 512)).set(a_data.clone());
    let b = cx.tensor((512, 512)).set(b_data.clone());
    let mut c = a.matmul(b).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<512>, DConst::<512>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<512>, DConst::<512>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_batch_matmul() {
    let mut cx = Graph::new();
    let a = cx
        .tensor((2, 2, 3))
        .set(vec![1., 2., 3., 1., 2., 1., 1., 2., 3., 1., 2., 1.]);
    let b = cx
        .tensor((3, 4))
        .set(vec![1., 2., 3., 1., 1., 2., 1., 2., -1., -2., 1., 2.]);
    let mut c = a.matmul(b).retrieve();

    cx.compile(MetalCompiler::<f32>::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor([[[1., 2., 3.], [1., 2., 1.]], [[1., 2., 3.], [1., 2., 1.]]]);
    let d_b = d_dev.tensor([[1., 2., 3., 1.], [1., 2., 1., 2.], [-1., -2., 1., 2.]]);
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_matmul_transpose() {
    const M: usize = 1024; // Any
    const K: usize = 16; // >= 16
    const N: usize = 256; // >= 256, power of 2
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let a_data = random_vec_rng(M * K, &mut rng);
    let a = cx.tensor((M, K)).set(a_data.clone());
    let b_data = random_vec_rng(K * N, &mut rng);
    let b = cx.tensor((N, K)).set(b_data.clone());
    let a_t_data = random_vec_rng(K * M, &mut rng);
    let a_t = cx.tensor((K, M)).set(a_t_data.clone());
    let b_t_data = random_vec_rng(K * N, &mut rng);
    let b_t = cx.tensor((K, N)).set(b_t_data.clone());

    let mut a_b = a.matmul(b.permute((1, 0))).retrieve();
    let mut a_b_t = a.matmul(b_t).retrieve();
    let mut a_t_b = a_t.permute((1, 0)).matmul(b.permute((1, 0))).retrieve();
    let mut a_t_b_t = a_t.permute((1, 0)).matmul(b_t).retrieve();

    cx.compile(
        MetalCompiler::<f32>::default(),
        (&mut a_b, &mut a_b_t, &mut a_t_b, &mut a_t_b_t),
    );
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(a_data, (DConst::<M>, DConst::<K>));
    let d_b = d_dev.tensor_from_vec(b_data, (DConst::<N>, DConst::<K>));
    let d_a_t = d_dev.tensor_from_vec(a_t_data, (DConst::<K>, DConst::<M>));
    let d_b_t = d_dev.tensor_from_vec(b_t_data, (DConst::<K>, DConst::<N>));
    let d_a_b = d_a.clone().matmul(d_b.clone().permute());
    let d_a_b_t = d_a.matmul(d_b_t.clone());
    let d_a_t_b = d_a_t
        .clone()
        .permute::<_, dfdx::shapes::Axes2<1, 0>>()
        .matmul(d_b.permute());
    let d_a_t_b_t = d_a_t
        .permute::<_, dfdx::shapes::Axes2<1, 0>>()
        .matmul(d_b_t);

    assert_close(&a_b.data(), &d_a_b.as_vec());
    assert_close(&a_b_t.data(), &d_a_b_t.as_vec());
    assert_close(&a_t_b.data(), &d_a_t_b.as_vec());
    assert_close(&a_t_b_t.data(), &d_a_t_b_t.as_vec());
}

#[test]
fn test_relu_and_linear() {
    // Test single and batch, unoptimized and optimized
    let mut cx = Graph::new();
    let input_data = random_vec(32);
    let w1 = random_vec(32 * 64);
    let w2 = random_vec(32 * 64);
    let batch = cx.named_tensor("Batch", (2, 32)).set(random_vec(32 * 2));
    let a = cx.named_tensor("Single", 32).set(input_data.clone());

    let model = (
        Linear::new(32, 64, false, &mut cx),
        ReLU,
        Linear::new(64, 32, false, &mut cx),
    );
    model.0.weight.set(w1.clone());
    model.2.weight.set(w2.clone());
    let mut b = model.forward(a).retrieve();
    let mut batch_out = model.forward(batch).retrieve();
    cx.execute();

    let unoptimized_b = b.data();
    let unoptimized_batch_out = batch_out.data();
    b.drop();
    batch_out.drop();
    cx.compile(
        <(GenericCompiler, MetalCompiler<f32>)>::default(),
        (&mut b, &mut batch_out),
    );
    cx.execute();

    assert_close_precision(&unoptimized_b, &b.data(), 1e-2);
    assert_close_precision(&unoptimized_batch_out, &batch_out.data(), 1e-2);

    // Test against dfdx
    let dev = Cpu::default();
    let mut model = <(
        dfdx::nn::modules::builders::UnbiasedLinear<32, 64>,
        dfdx::nn::modules::builders::ReLU,
        dfdx::nn::modules::builders::UnbiasedLinear<64, 32>,
    )>::build_on_device(&dev);
    // Set weights
    model.0.weight = dev
        .tensor_from_vec(w1, (DConst::<32>, DConst::<64>))
        .permute();
    model.2.weight = dev
        .tensor_from_vec(w2, (DConst::<64>, DConst::<32>))
        .permute();
    let a = dev.tensor_from_vec(input_data, (DConst::<32>,));
    let out = model.forward(a);

    assert_close_precision(&unoptimized_b, &out.as_vec(), 1e-2);
}

#[test]
fn test_transformer_encoder_block() {
    let mut cx = Graph::new();
    let model = luminal_nn::TransformerEncoderBlock::new(3, 4, 1, &mut cx);
    model
        .attention
        .w_k
        .weight
        .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
    model
        .attention
        .w_q
        .weight
        .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
    model
        .attention
        .w_v
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
    model
        .attention
        .w_o
        .weight
        .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
    model
        .ff
        .0
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
    model
        .ff
        .2
        .weight
        .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

    let a = cx
        .tensor(('a', 3))
        .set_dyn(vec![-1., 2., 3., 3., 3., -1.], (2, 3));
    let mut b = model.forward(a).retrieve();

    cx.compile(<(GenericCompiler, MetalCompiler<f32>)>::default(), &mut b);
    cx.execute();

    let d_dev = Cpu::default();
    let mut d_model: dfdx::nn::modules::TransformerEncoderBlock<3, 1, 4, f32, Cpu> =
        d_dev.build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
    d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
    d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
    d_model.self_attn.w_o.weight = d_dev
        .tensor_from_vec(
            vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_k.weight = d_dev
        .tensor_from_vec(
            vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_q.weight = d_dev
        .tensor_from_vec(
            vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.self_attn.w_v.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
            (DConst::<3>, DConst::<3>),
        )
        .permute();
    d_model.ff.0 .0.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
            (DConst::<3>, DConst::<4>),
        )
        .permute();
    d_model.ff.0 .0.bias = d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (DConst::<4>,));
    d_model.ff.0 .2.weight = d_dev
        .tensor_from_vec(
            vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
            (DConst::<4>, DConst::<3>),
        )
        .permute();
    d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
    d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
    d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
    d_model.norm1.epsilon = 1e-5;
    d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
    d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
    d_model.norm2.epsilon = 1e-5;
    let d_a = d_dev.tensor_from_vec(vec![-1., 2., 3., 3., 3., -1.], (DConst::<2>, DConst::<3>));
    let d_b = d_model.forward(d_a);

    assert_close(&b.data(), &d_b.as_vec());
}

#[test]
fn test_pool_1d_dims() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor((4, 4)).set(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    // Stride 1
    let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    cx.execute();

    assert_exact(
        &out1.data(),
        &[
            1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13., 14.,
            15., 14., 15., 16.,
        ],
    );
}

#[test]
fn test_pool_2d() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor((4, 4)).set(vec![
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    ]);
    // 3x3 kernel
    let out1 = inp1
        // Pool first dim first by moving it to end
        .permute((1, 0))
        .pool_last_dim(3, 1, 1)
        // Now move other dim to end
        .permute((1, 2, 0))
        .pool_last_dim(3, 1, 1)
        // Now swap middle two dims
        .permute((0, 2, 1, 3))
        // Now merge both pooled dimensions
        .reshape((4, 3, 3))
        .retrieve();

    cx.execute();

    assert_exact(
        &out1.data(),
        &[
            1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00, 7.00,
            8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00, 14.00, 15.00,
            6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
        ],
    );
}

#[test]
fn test_pool_1d_dilation() {
    let mut cx = Graph::new();

    let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
    // Stride 1
    let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
    // Stride 2
    let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
    // Stride 3
    let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

    cx.execute();

    assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    assert_exact(&out3.data(), &[1., 3.]);
}

#[test]
fn test_conv2d() {
    // Test Conv2D with stride=kernel (no overlap), matching verification test pattern
    let mut rng = StdRng::seed_from_u64(42);

    const CH_IN: usize = 5;
    const CH_OUT: usize = 2;
    const KERNEL: (usize, usize) = (2, 2);
    const STRIDE: (usize, usize) = (2, 2);
    const DILATION: (usize, usize) = (1, 1);
    const DIMX_IN: usize = 16;
    const DIMY_IN: usize = 8;

    let input_data: Vec<f32> = (0..(CH_IN * DIMX_IN * DIMY_IN))
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let weight_data: Vec<f32> = (0..(CH_OUT * CH_IN * KERNEL.0 * KERNEL.1))
        .map(|_| rng.random::<f32>() * 0.2 - 0.1)
        .collect();

    // Run on CPU (no compiler)
    let mut cx_cpu = Graph::new();
    let inp_cpu = cx_cpu
        .tensor((CH_IN, DIMX_IN, DIMY_IN))
        .set(input_data.clone());
    let model_cpu =
        luminal_nn::Conv2D::new(CH_IN, CH_OUT, KERNEL, STRIDE, DILATION, false, &mut cx_cpu);
    model_cpu.weight.set(weight_data.clone());
    let out_cpu = model_cpu.forward(inp_cpu).retrieve();
    cx_cpu.execute();
    let cpu_result = out_cpu.data();

    // Run on Metal
    let mut cx_metal = Graph::new();
    let inp_metal = cx_metal
        .tensor((CH_IN, DIMX_IN, DIMY_IN))
        .set(input_data.clone());
    let model_metal = luminal_nn::Conv2D::new(
        CH_IN,
        CH_OUT,
        KERNEL,
        STRIDE,
        DILATION,
        false,
        &mut cx_metal,
    );
    model_metal.weight.set(weight_data.clone());
    let mut out_metal = model_metal.forward(inp_metal).retrieve();
    cx_metal.compile(MetalCompiler::<f32>::default(), &mut out_metal);
    cx_metal.execute();
    let metal_result = out_metal.data();

    // Compare
    assert_eq!(
        cpu_result.len(),
        metal_result.len(),
        "Output lengths differ: CPU={}, Metal={}",
        cpu_result.len(),
        metal_result.len()
    );
    assert_close(&metal_result, &cpu_result);
}

#[test]
fn test_conv1d_pad_stride() {
    let mut cx = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    const CH_IN: usize = 80;
    const CH_OUT: usize = 384;
    const KERNEL: usize = 3;
    const STRIDE: usize = 1;
    const PADDING: usize = 1;
    const DIM_IN: usize = 10;
    let kernel_data = random_vec_rng(KERNEL * CH_IN * CH_OUT, &mut rng);
    let input_data = random_vec_rng(CH_IN * DIM_IN, &mut rng);

    let model = Conv1D::new(CH_IN, CH_OUT, KERNEL, STRIDE, 1, PADDING, false, &mut cx);
    model.weight.set(kernel_data.clone());

    let inp1 = cx
        .tensor((1, CH_IN, 's'))
        .set_dyn(input_data.clone(), (1, CH_IN, DIM_IN));

    let mut out1 = model.forward(inp1).retrieve();
    cx.compile(crate::MetalCompiler::<f32>::default(), &mut out1);
    cx.execute();

    let input =
        candle_core::Tensor::from_vec(input_data, (1, CH_IN, DIM_IN), &candle_core::Device::Cpu)
            .unwrap();
    let kernel = candle_core::Tensor::from_vec(
        kernel_data,
        (CH_OUT, CH_IN, KERNEL),
        &candle_core::Device::Cpu,
    )
    .unwrap();
    let output = input.conv1d(&kernel, PADDING, STRIDE, 1, 1).unwrap();

    assert_close(
        &out1.data(),
        &output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
}
