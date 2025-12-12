use crate::prelude::*;

/// Rectified Linear Unit activation function
#[derive(Default)]
pub struct ReLU;

impl SerializeModule for ReLU {
    fn serialize(&self, _: &mut Serializer) {}
}

impl Module<GraphTensor> for ReLU {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        input.relu()
    }
}

/// Gaussian Error Linear Unit activation function
#[derive(Default)]
pub struct GeLU;

impl SerializeModule for GeLU {
    fn serialize(&self, _: &mut Serializer) {}
}

impl Module<GraphTensor> for GeLU {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        input.gelu()
    }
}

/// Sigmoid activation function
#[derive(Default)]
pub struct Sigmoid;

impl SerializeModule for Sigmoid {
    fn serialize(&self, _: &mut Serializer) {}
}

impl Module<GraphTensor> for Sigmoid {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        input.sigmoid()
    }
}

/// Swish activation function
#[derive(Default)]
pub struct Swish;

impl SerializeModule for Swish {
    fn serialize(&self, _: &mut Serializer) {}
}

impl Module<GraphTensor> for Swish {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        input.swish()
    }
}

/// Tanh activation function
#[derive(Default)]
pub struct Tanh;

impl SerializeModule for Tanh {
    fn serialize(&self, _: &mut Serializer) {}
}

impl Module<GraphTensor> for Tanh {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        input.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::ReLU;
    use crate::nn::Linear as LuminalLinear;
    use crate::prelude::{GenericCompiler, Graph, Module as LuminalModule};
    use crate::tests::assert_close;
    use dfdx::prelude::{Module as DfdxModule, *};

    #[test]
    fn test_relu_and_linear() {
        // Test single and batch, unoptimized and optimized
        let mut cx = Graph::new();
        let batch = cx.tensor((2, 3)).set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let a = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);

        let model = (
            LuminalLinear::new(3, 4, false, &mut cx),
            ReLU,
            LuminalLinear::new(4, 2, false, &mut cx),
        );
        model
            .0
            .weight
            .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
        model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
        let mut b = LuminalModule::forward(&model, a).retrieve();
        let mut batch_out = LuminalModule::forward(&model, batch).retrieve();

        cx.execute();

        let unoptimized_b = b.data();
        let unoptimized_batch_out = batch_out.data();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
        cx.execute();

        assert_close(&unoptimized_b, &b.data());
        assert_close(&unoptimized_batch_out, &batch_out.data());

        // Test against dfdx
        let dev = Cpu::default();
        let mut model = <(
            dfdx::nn::modules::builders::UnbiasedLinear<3, 4>,
            dfdx::nn::modules::builders::ReLU,
            dfdx::nn::modules::builders::UnbiasedLinear<4, 2>,
        )>::build_on_device(&dev);
        // Set weights
        model.0.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
                (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
            )
            .permute();
        model.2.weight = dev
            .tensor_from_vec(
                vec![1., 2., 3., 1., 2., 3., 1., 2.],
                (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<2>),
            )
            .permute();
        let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
        let d_batch = dev.tensor_from_vec(
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
        );
        let out = DfdxModule::forward(&model, a);
        let d_batch_out = DfdxModule::forward(&model, d_batch);

        assert_close(&unoptimized_b, &out.as_vec());
        assert_close(&unoptimized_batch_out, &d_batch_out.as_vec());
    }
}
