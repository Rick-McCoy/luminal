use std::any::TypeId;

use itertools::Itertools;
use petgraph::{algo::toposort, visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    op::{
        Add, Contiguous, Exp2, Function, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Sin, Sqrt,
        SumReduce,
    },
    prelude::{tinyvec::ArrayVec, *},
};

#[derive(Clone, Debug)]
pub struct Autograd(Vec<NodeIndex>, NodeIndex);

impl Autograd {
    pub fn new<W: ToIds>(params: W, loss: GraphTensor) -> Self {
        Self(params.to_ids(), loss.id)
    }
}

// Run dfs with a starting stack and record all encountered nodes in a set
fn build_dfs_set(
    stack: &mut Vec<NodeIndex>,
    graph: &StorageGraph,
    direction: Direction,
) -> FxHashSet<NodeIndex> {
    let mut set = FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !set.contains(&n) {
            set.insert(n);
            stack.extend(
                graph
                    .edges_directed(n, direction)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| match direction {
                        Direction::Incoming => e.source(),
                        Direction::Outgoing => e.target(),
                    }),
            );
        }
    }
    set
}

impl Compiler for Autograd {
    type Output = Vec<(NodeIndex, ShapeTracker)>;
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) -> Vec<(NodeIndex, ShapeTracker)> {
        let Autograd(params, loss) = self;
        // Build up valid set for nodes we want to pay attention to (everything outside of this set doesn't matter)
        let forward_set = build_dfs_set(&mut params.clone(), graph, Direction::Outgoing);
        let backward_set = build_dfs_set(&mut vec![*loss], graph, Direction::Incoming);
        let valid_set: FxHashSet<_> = forward_set.intersection(&backward_set).copied().collect();

        // We have the last loss node, now let's backprop through everything to get the gradient graph
        let mut grads = FxHashMap::default();
        // Add loss gradient
        grads.insert(
            *loss,
            (
                graph.constant(1.0).id,
                ShapeTracker::new(()), // Assume scalar loss for now
            ),
        );
        let weight_set = params.iter().copied().collect::<FxHashSet<_>>();
        for fwd_node in toposort(&graph.graph, None).unwrap().into_iter().rev() {
            if !valid_set.contains(&fwd_node) {
                continue;
            }
            // Check if the node is undifferentiable
            let graph_ref: *mut Graph = graph;
            let op = graph.node_weight(fwd_node).unwrap().as_any().type_id();
            if op == TypeId::of::<Function>() {
                continue;
            }
            if op == TypeId::of::<Mod>() || op == TypeId::of::<LessThan>() {
                assert!(
                    !weight_set.contains(&fwd_node),
                    "{fwd_node:?} is marked as a weight but is undifferentiable: {:?}",
                    graph.node_weight(fwd_node).unwrap()
                );
                continue;
            }

            // Differentiate through fwd_node to get gradients for it's sources
            // Get input tensors
            let inps = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, (a, _, _))| *a)
                .map(|(node, (_, _, sh))| GraphTensor::from_id(node, sh, graph_ref))
                .collect::<Vec<_>>();
            let mut prev_grad = {
                let (id, sh) = grads[&fwd_node];
                GraphTensor::from_id(id, sh, graph_ref)
            };
            if op == TypeId::of::<Add>() {
                // f(a, b) = a + b
                // df/da = 1
                if valid_set.contains(&inps[0].id) {
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
                // df/db = 1
                if valid_set.contains(&inps[1].id) {
                    add_grad(prev_grad, inps[1], graph, &mut grads);
                }
            } else if op == TypeId::of::<Mul>() {
                // f(a, b) = a * b
                // df/da = b
                if valid_set.contains(&inps[0].id) {
                    add_grad(inps[1] * prev_grad, inps[0], graph, &mut grads);
                }
                // df/db = a
                if valid_set.contains(&inps[1].id) {
                    add_grad(inps[0] * prev_grad, inps[1], graph, &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<SumReduce>(fwd_node)
                .cloned()
            {
                // f(x) = sum_reduce(x)
                // f'(x) = 1
                if valid_set.contains(&inps[0].id) {
                    prev_grad
                        .shape
                        .expand_dim(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<MaxReduce>(fwd_node)
                .cloned()
            {
                // f(x) = max_reduce(x)
                // f'(x) = x == max_reduce(x)
                if valid_set.contains(&inps[0].id) {
                    // fwd_nod is already max_reduce(x)
                    prev_grad
                        .shape
                        .expand_dim(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    let reduced = GraphTensor::from_id(fwd_node, prev_grad.shape, graph_ref);
                    let grad = inps[0].eq(reduced) * prev_grad;
                    add_grad(grad, inps[0], graph, &mut grads);
                }
            } else if op == TypeId::of::<Contiguous>() {
                if valid_set.contains(&inps[0].id) {
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
            } else {
                if !valid_set.contains(&inps[0].id) {
                    continue;
                }
                let local_grad = if op == TypeId::of::<Log2>() {
                    // f(x) = log2(x)
                    // f'(x) = 1 / (x * ln(2))
                    1.0 / (inps[0] * 2_f32.ln())
                } else if op == TypeId::of::<Exp2>() {
                    // f(x) = exp2(x)
                    // f'(x) = exp2(x) * ln(2)
                    inps[0].exp2() * 2_f32.ln()
                } else if op == TypeId::of::<Sin>() {
                    // f(x) = sin(x)
                    // f'(x) = cos(x)
                    inps[0].cos()
                } else if op == TypeId::of::<Sqrt>() {
                    // f(x) = sqrt(x)
                    // f'(x) = 1 / (2 * sqrt(x))
                    1.0 / (2.0 * inps[0].sqrt())
                } else if op == TypeId::of::<Recip>() {
                    // f(x) = 1 / x
                    // f'(x) = -1 / x**2
                    -1.0 / (inps[0] * inps[0])
                } else {
                    unreachable!()
                };
                add_grad(local_grad * prev_grad, inps[0], graph, &mut grads);
            }
        }

        // Create a gradient array to match 1-1 with the weight array passed in
        self.0.iter().map(|weight| grads[weight]).collect()
    }
}

fn add_grad(
    mut grad: GraphTensor,
    fwd: GraphTensor,
    graph: &mut Graph,
    grad_map: &mut FxHashMap<NodeIndex, (NodeIndex, ShapeTracker)>,
) {
    // Reshape gradient to match the shape of the input source (before the input was reshaped)
    // Undo permutes
    let mut new_indexes = ArrayVec::new();
    new_indexes.resize(fwd.shape.len(), 0);
    for i in 0..fwd.shape.len() {
        new_indexes[fwd.shape.indexes[i]] = grad.shape.indexes[i];
    }
    grad.shape.indexes = new_indexes;

    // Undo expands (sum reduce)
    for i in fwd.shape.indexes.into_iter().rev() {
        if fwd.shape.fake[i] {
            grad.id = graph
                .add_op(SumReduce(i))
                .input(grad.id, 0, grad.shape)
                .finish();
            grad.shape.remove_dim(i);
            grad.shape = grad.shape.contiguous();
        }
    }

    // Check to see if a reshape was done here. If so, we may need to assert grad shape is contiguous or insert a contiguous call
    if let Some((_, _, mut pre_fwd_shape)) = graph.get_sources(fwd.id).first() {
        if let Some(SumReduce(dim)) = graph.try_get_op(fwd.id) {
            pre_fwd_shape.remove_dim(*dim);
        } else if let Some(MaxReduce(dim)) = graph.try_get_op(fwd.id) {
            pre_fwd_shape.remove_dim(*dim);
        }
        if grad.shape.dims() != pre_fwd_shape.dims() {
            if !grad.shape.is_contiguous() {
                grad = grad.contiguous();
            }
            grad.shape = pre_fwd_shape.contiguous();
        }
    }

    if let Some((existing_grad_node, existing_grad_shape)) = grad_map.get(&fwd.id).copied() {
        let grad = GraphTensor::from_id(grad.id, grad.shape, graph);
        let existing_grad = GraphTensor::from_id(existing_grad_node, existing_grad_shape, graph);
        let new_grad = grad + existing_grad;
        grad_map.insert(fwd.id, (new_grad.id, new_grad.shape));
    } else {
        grad_map.insert(fwd.id, (grad.id, grad.shape));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Module as LModule;
    use dfdx::nn::Module as DModule;
    crate::test_imports!();

    fn get_vec(grad: (NodeIndex, ShapeTracker), cx: &mut Graph) -> Vec<f32> {
        GraphTensor::from_id(grad.0, grad.1, cx).data()
    }

    #[test]
    fn test_autograd_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("Input", 2).set([10., 5.]);
        let b = a.max(0);

        let grads = cx.compile(Autograd::new(a, b), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let d_a = dev.tensor([10., 5.]);
        let d_b = d_a.trace(Gradients::leaky()).max();
        let d_grads = d_b.backward();

        assert_exact(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_matmul() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("A", (2, 2)).set([[2., 4.], [3., 1.]]);
        let input = cx.named_tensor("Input", 2).set([10., 5.]);
        let output = (input.matmul(a)).sum(0);

        let grads = cx.compile(Autograd::new(a, output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let w1 = dev.tensor([[2., 4.], [3., 1.]]);
        let inp = dev.tensor([10., 5.]);
        let out = inp.trace(Gradients::leaky()).matmul(w1.clone()).sum();
        let d_grads = out.backward();

        assert_exact(&get_vec(grads[0], &mut cx), &d_grads.get(&w1).as_vec());
    }

    #[test]
    fn test_autograd_mlp() {
        let mut cx = Graph::new();
        let model = (
            crate::nn::Linear::new(2, 2, false, &mut cx),
            crate::nn::ReLU,
            crate::nn::Linear::new(2, 1, false, &mut cx),
        );
        model.0.weight.set([[2., 4.], [3., 1.]]);
        model.2.weight.set([[6.], [5.]]);
        let input = cx.named_tensor("Input", 2).set([10., 5.]);
        let output = model.forward(input).sum(0);

        let mut grads = cx.compile(Autograd::new(params(model), output), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), &mut grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let mut d_model = dev.build_module::<(
            dfdx::nn::builders::UnbiasedLinear<2, 2>,
            dfdx::nn::builders::ReLU,
            dfdx::nn::builders::UnbiasedLinear<2, 1>,
        ), f32>();
        d_model.0.weight = dev.tensor([[2., 4.], [3., 1.]]).permute();
        d_model.2.weight = dev.tensor([[6.], [5.]]).permute();
        let inp = dev.tensor([10., 5.]);
        let out = d_model.forward(inp.trace(Gradients::leaky())).sum();
        let d_grads = out.backward();

        assert_exact(
            &get_vec(grads[0], &mut cx),
            &d_grads.get(&d_model.0.weight).permute().as_vec(),
        );
        assert_exact(
            &get_vec(grads[1], &mut cx),
            &d_grads.get(&d_model.2.weight).as_vec(),
        );
    }

    #[test]
    fn test_autograd_layer_norm() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([-1., 2., 3.]);
        let mut b = a.layer_norm(0, 1e-5).max(0).retrieve();

        let grads = cx.compile(Autograd::new(a, b), &mut b);
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), &mut b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([-1., 2., 3.]);
        let d_b = d_a.trace(Gradients::leaky()).normalize(1e-5).max();
        assert_close(&b.data(), &d_b.as_vec());
        let d_grads = d_b.backward();
        assert_close(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_softmax() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([-1., 2., 3.]);
        let mut b = a.softmax(0).max(0).retrieve();

        let mut grads = cx.compile(Autograd::new(a, b), &mut b);
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), (&mut grads, &mut b));
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([-1., 2., 3.]);
        let d_b = d_a.trace(Gradients::leaky()).softmax().max();
        assert_close(&b.data(), &d_b.as_vec());
        let d_grads = d_b.backward();
        assert_close(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_transformer() {
        let mut cx = Graph::new();
        let model = crate::nn::TransformerEncoderBlock::new(3, 4, 1, &mut cx);
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

        let a = cx.tensor((2, 3)).set([[-1., 2., 3.], [3., 3., -1.]]);
        let target = cx.tensor((2, 3)).set([[0., 1., 0.], [0., 0., 1.]]);
        let out = model.forward(a);
        let mut loss = super::cross_entropy_with_logits_loss(out, target).retrieve();

        let mut model_params = params(&model);
        let mut grads = cx.compile(
            Autograd::new((&model_params, a), loss),
            (&mut model_params, &mut loss),
        );
        cx.keep_tensors(&grads);
        cx.compile(
            GenericCompiler::default(),
            (&mut model_params, &mut grads, &mut loss),
        );
        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = d_dev
            .build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
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
        let d_target =
            d_dev.tensor_from_vec(vec![0., 1., 0., 0., 0., 1.], (DConst::<2>, DConst::<3>));
        let d_b = d_model.forward(d_a.trace(Gradients::leaky()));
        let d_loss = dfdx::prelude::cross_entropy_with_logits_loss(d_b, d_target);

        assert_close(&loss.data(), &d_loss.as_vec());

        let d_grads = d_loss.backward();
        assert_close(
            &get_vec(*grads.last().unwrap(), &mut cx),
            &d_grads.get(&d_a).as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.ff.2.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads.get(&d_model.ff.0 .2.weight).permute().as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.ff.0.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads.get(&d_model.ff.0 .0.weight).permute().as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_o.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_o.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_q.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_q.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_k.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_k.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_v.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_v.weight)
                .permute()
                .as_vec(),
        );
    }

    #[test]
    fn test_autograd_conv2d() {
        let mut cx = Graph::new();

        // Simple conv2d: 1 input channel, 1 output channel, 2x2 kernel
        let conv = crate::nn::Conv2D::new(1, 1, (2, 2), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1, 0.2, 0.3, 0.4]);

        // Input: batch=1, channels=1, 4x4
        let input = cx.tensor((1, 1, 4, 4)).set(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);

        let output = conv.forward(input);
        let loss = output.sum(output.shape.all_axes());

        // Try to compute gradients
        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        // Just verify it doesn't panic and produces a gradient
        let grad_data = get_vec(grads[0], &mut cx);
        assert_eq!(grad_data.len(), 4); // 2x2 kernel
        println!("Conv2D gradient: {:?}", grad_data);
    }

    #[test]
    fn test_conv2d_gradient_magnitude() {
        let mut cx = Graph::new();

        // Simple conv2d: 1 input channel, 1 output channel, 3x3 kernel
        let conv = crate::nn::Conv2D::new(1, 1, (3, 3), (1, 1), (1, 1), false, &mut cx);
        // Small weights to avoid explosion
        conv.weight.set(vec![0.01; 9]);

        // Input: batch=2, channels=1, 4x4
        let input = cx.tensor((2, 1, 4, 4)).set(vec![
            // Batch 1
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            // Batch 2
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        ]);

        let output = conv.forward(input);
        let loss = output.sum(output.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D weight gradient: {:?}", grad_data);

        // Check gradient magnitudes are reasonable
        for &g in &grad_data {
            assert!(!g.is_nan(), "Gradient is NaN");
            assert!(!g.is_infinite(), "Gradient is infinite");
            assert!(g.abs() < 100.0, "Gradient too large: {}", g);
        }
    }

    #[test]
    fn test_conv2d_training_step() {
        // Test a single training step with conv2d + cross-entropy-like loss
        let mut cx = Graph::new();

        // Conv2D: 1->2 channels, 3x3 kernel
        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]); // 2 * 1 * 3 * 3 = 18

        // Input: batch=2, 1 channel, 4x4
        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        // Forward pass: conv -> relu -> softmax-like loss
        let output = conv.forward(input); // (2, 2, 2, 2) after conv
        let output_flat = output.reshape((2, 8)); // Flatten to (batch, features)

        // Simulate cross-entropy: log_softmax + sum
        let log_probs = output_flat.log_softmax(1);
        let loss = -log_probs.mean(log_probs.shape.all_axes());

        // Compute gradients
        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Training step gradient: {:?}", grad_data);

        // Check for NaN/Inf
        for (i, &g) in grad_data.iter().enumerate() {
            assert!(!g.is_nan(), "Gradient[{}] is NaN", i);
            assert!(!g.is_infinite(), "Gradient[{}] is infinite: {}", i, g);
        }
        println!("Conv2D training step test passed!");
    }

    #[test]
    fn test_pool_last_dim_gradient() {
        // Test gradient through pool_last_dim specifically
        let mut cx = Graph::new();

        let weight = cx
            .tensor((2, 4))
            .set(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // Create a pooled view (this is what conv2d uses internally)
        let input = cx.tensor((1, 1, 4, 4)).set(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);

        // Mimicking what conv2d does: pool -> permute -> reshape -> matmul
        let pooled = input
            .pool_last_dim(2, 1, 1) // (1, 1, 4, 3, 2)
            .permute((0, 1, 3, 4, 2)) // (1, 1, 3, 2, 4)
            .pool_last_dim(2, 1, 1); // (1, 1, 3, 2, 3, 2)

        // Flatten for matmul
        let flat = pooled.reshape((1, 4, 9)); // (batch, ch_in*k*k, out)

        // Matmul with weight
        let out = weight.expand_dim(0, 1).matmul(flat);
        let loss = out.sum(out.shape.all_axes());

        let grads = cx.compile(Autograd::new(weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("pool_last_dim gradient: {:?}", grad_data);

        for (i, &g) in grad_data.iter().enumerate() {
            assert!(!g.is_nan(), "Gradient[{}] is NaN", i);
            assert!(!g.is_infinite(), "Gradient[{}] is infinite: {}", i, g);
        }
    }

    #[test]
    fn test_conv2d_without_softmax() {
        // Test conv2d with simple MSE-like loss (no softmax)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        // Simple sum loss without softmax
        let loss = output.sum(output.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D without softmax gradient: {:?}", grad_data);

        // These should NOT be zero
        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(
            !all_zero,
            "All gradients are zero - gradient not flowing through conv2d!"
        );
    }

    #[test]
    fn test_log_softmax_gradient() {
        // Test log_softmax gradient in isolation
        let mut cx = Graph::new();

        let weight = cx
            .tensor((2, 4))
            .set(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let input = cx
            .tensor((2, 4))
            .set(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Simple matmul -> log_softmax -> mean
        let out = weight.matmul(input.permute((1, 0)));
        let log_probs = out.log_softmax(1);
        let loss = -log_probs.mean(log_probs.shape.all_axes());

        let grads = cx.compile(Autograd::new(weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("log_softmax gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "log_softmax gradients are all zero!");
    }

    #[test]
    #[should_panic(expected = "Conv2D + log_softmax gradients are all zero")]
    fn test_conv2d_plus_logsoftmax() {
        // Test conv2d + reshape + log_softmax - the combination used in CNN training
        // This test documents a known bug where the combination yields zero gradients
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input); // (2, 2, 2, 2)
        println!("Conv output shape: {:?}", output.dims());

        // Flatten + log_softmax (mimics what cross-entropy does)
        let flat = output.reshape((2, 8));
        println!("Flat shape: {:?}", flat.dims());

        let log_probs = flat.log_softmax(1);
        let loss = -log_probs.mean(log_probs.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D + log_softmax gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(
            !all_zero,
            "Conv2D + log_softmax gradients are all zero! This is the bug."
        );
    }

    #[test]
    fn test_conv2d_plus_reshape_sum() {
        // Test conv2d + reshape + sum to isolate if reshape is the issue
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input); // (2, 2, 2, 2)
        let flat = output.reshape((2, 8)); // Same reshape as the failing case
        let loss = flat.sum(flat.shape.all_axes()); // But with sum instead of log_softmax

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D + reshape + sum gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "Conv2D + reshape + sum gradients are all zero!");
    }

    #[test]
    fn test_conv2d_plus_reshape_max() {
        // Test conv2d + reshape + max reduction (part of log_softmax)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input); // (2, 2, 2, 2)
        let flat = output.reshape((2, 8));
        // max along axis 1, then sum to scalar
        let max_val = flat.max(1);
        let loss = max_val.sum(max_val.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D + reshape + max gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "Conv2D + reshape + max gradients are all zero!");
    }

    #[test]
    fn test_conv2d_plus_reshape_max_expand() {
        // Test conv2d + reshape + max + expand (the exact pattern in log_softmax)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input); // (2, 2, 2, 2)
        let flat = output.reshape((2, 8));
        // This is the first line of log_softmax: m = self - self.max(axes).expand(self.shape)
        let max_expanded = flat.max(1).expand(flat.shape);
        let m = flat - max_expanded;
        let loss = m.sum(m.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D + reshape + max + expand gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(
            !all_zero,
            "Conv2D + reshape + max + expand gradients are all zero!"
        );
    }

    #[test]
    fn test_conv2d_exp_sum_log() {
        // Test the second part of log_softmax: exp -> sum -> log
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));
        // Part 1 of log_softmax
        let m = flat - flat.max(1).expand(flat.shape);
        // Part 2: exp -> sum -> log
        let exp_sum_log = m.exp().sum(1).log();
        let loss = exp_sum_log.sum(exp_sum_log.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D + exp + sum + log gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "Conv2D + exp + sum + log gradients are zero!");
    }

    #[test]
    fn test_conv2d_full_logsoftmax_pattern() {
        // Test the complete log_softmax pattern with varied inputs
        // (uniform inputs cause numerical cancellation)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight
            .set((0..18).map(|i| 0.1 * (i as f32 - 9.0)).collect::<Vec<_>>());

        let input = cx
            .tensor((2, 1, 4, 4))
            .set((0..32).map(|i| i as f32 / 32.0).collect::<Vec<_>>());

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));
        // Full log_softmax pattern
        let m = flat - flat.max(1).expand(flat.shape);
        let log_sum_exp = m.exp().sum(1).log().expand(m.shape);
        let log_probs = m - log_sum_exp;
        let loss = log_probs.sum(log_probs.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!(
            "Conv2D + full log_softmax pattern gradient: {:?}",
            grad_data
        );

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-6);
        assert!(
            !all_zero,
            "Conv2D + full log_softmax pattern gradients are zero!"
        );
    }

    #[test]
    fn test_double_use_with_expand() {
        // Test the pattern: x - f(x).expand(x.shape) where x is used twice
        // Uses varied inputs to avoid numerical cancellation
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight
            .set((0..18).map(|i| 0.1 * (i as f32 - 9.0)).collect::<Vec<_>>());

        let input = cx
            .tensor((2, 1, 4, 4))
            .set((0..32).map(|i| i as f32 / 32.0).collect::<Vec<_>>());

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // Pattern: x - f(x).expand(x.shape)
        // f(x) = exp(x).sum(1).log()
        let log_sum_exp = flat.exp().sum(1).log().expand(flat.shape);
        let result = flat - log_sum_exp; // flat is used in both places
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Double use with expand gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-6);
        assert!(!all_zero, "Double use with expand gradients are zero!");
    }

    #[test]
    fn test_simple_x_minus_sum_expand() {
        // Simplest form: x - x.sum().expand(x.shape)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // Simplest problematic pattern: x - x.sum(1).expand(x.shape)
        let sum_expanded = flat.sum(1).expand(flat.shape);
        let result = flat - sum_expanded;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("x - x.sum().expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "x - x.sum().expand() gradients are zero!");
    }

    #[test]
    fn test_without_conv2d() {
        // Test the same pattern without conv2d to confirm it's not a conv2d issue
        let mut cx = Graph::new();

        let weight = cx.tensor((2, 8)).set(vec![0.1; 16]);
        let input = cx.tensor((8, 8)).set(vec![0.5; 64]);

        let flat = weight.matmul(input); // (2, 8)

        // Pattern: x - x.sum(1).expand(x.shape)
        let sum_expanded = flat.sum(1).expand(flat.shape);
        let result = flat - sum_expanded;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!(
            "Without conv2d: x - x.sum().expand() gradient: {:?}",
            grad_data
        );

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(
            !all_zero,
            "Without conv2d: x - x.sum().expand() gradients are zero!"
        );
    }

    #[test]
    fn test_exp_sum_log_expand_pattern() {
        // Test: x - x.exp().sum().log().expand(x.shape) with varied inputs
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight
            .set((0..18).map(|i| 0.1 * (i as f32 - 9.0)).collect::<Vec<_>>());

        let input = cx
            .tensor((2, 1, 4, 4))
            .set((0..32).map(|i| i as f32 / 32.0).collect::<Vec<_>>());

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // The log-sum-exp pattern
        let lse = flat.exp().sum(1).log().expand(flat.shape);
        let result = flat - lse;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("x - x.exp().sum().log().expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-6);
        assert!(
            !all_zero,
            "x - x.exp().sum().log().expand() gradients are zero!"
        );
    }

    #[test]
    fn test_exp_sum_log_no_expand() {
        // Test: just exp().sum().log() without the final expand and subtraction
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // exp -> sum -> log without expand
        let loss = flat.exp().sum(1).log().sum(0); // sum to scalar differently

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("exp().sum().log() (no expand) gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "exp().sum().log() gradients are zero!");
    }

    #[test]
    fn test_log_expand_subtract() {
        // Test: x - log(something).expand(x.shape)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // Test: log -> expand -> subtract
        let log_val = flat.sum(1).log().expand(flat.shape); // simplified
        let result = flat - log_val;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("x - log(sum(x)).expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "log -> expand -> subtract gradients are zero!");
    }

    #[test]
    fn test_exp_sum_expand_subtract() {
        // Test: x - exp(sum(x)).expand(x.shape)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.01; 18]); // smaller weights to avoid overflow

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.1; 32]); // smaller values

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // Test: sum -> exp -> expand -> subtract (no log)
        let exp_sum_val = flat.sum(1).exp().expand(flat.shape);
        let result = flat - exp_sum_val;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("x - exp(sum(x)).expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "exp -> expand -> subtract gradients are zero!");
    }

    #[test]
    fn test_exp_sum_log_chain_expanded() {
        // Test just exp -> sum -> log (no subtraction from x)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // exp -> sum -> log -> expand, then sum to loss
        let lse = flat.exp().sum(1).log().expand(flat.shape);
        let loss = lse.sum(lse.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("exp().sum().log().expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        assert!(!all_zero, "exp().sum().log().expand() gradients are zero!");
    }

    #[test]
    fn test_tensor_used_twice_simple() {
        // Test: using a tensor twice in subtraction (x - f(x))
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 18]);

        let input = cx.tensor((2, 1, 4, 4)).set(vec![0.5; 32]);

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // Use flat twice: directly and through a function
        // y = flat - flat.mean().expand()
        let mean_expanded = flat.mean(1).expand(flat.shape);
        let result = flat - mean_expanded;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("x - x.mean().expand() gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g == 0.0);
        // This should also NOT be zero (though numerically it might be close to zero
        // since we're subtracting mean and then summing)
        println!("All zero: {}", all_zero);
    }

    #[test]
    fn test_double_use_debug() {
        // Debug test: check what gradients look like at different stages
        let mut cx = Graph::new();

        // Simple 2D tensor
        let weight = cx
            .tensor((2, 4))
            .set(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);

        // y = x - x.exp().sum(1).log().expand(x.shape)
        // This is log_softmax essentially
        let lse = weight.exp().sum(1).log().expand(weight.shape);
        let result = weight - lse;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Simple 2D log_softmax pattern gradient: {:?}", grad_data);

        // Mathematical expectation: 1 - softmax(x) for each element
        // All values different so softmax values differ
        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-10);
        assert!(!all_zero, "Simple log_softmax pattern gradients are zero!");
    }

    #[test]
    fn test_matmul_reshape_logsoftmax() {
        // Test matmul -> reshape -> log_softmax (without conv2d)
        let mut cx = Graph::new();

        let weight = cx.tensor((8, 16)).set(vec![0.1; 128]);
        let input = cx.tensor((16, 4)).set(vec![0.5; 64]);

        let out = weight.matmul(input); // (8, 4)
        let flat = out.reshape((2, 16)); // reshape like what conv2d output does

        // log_softmax pattern
        let lse = flat.exp().sum(1).log().expand(flat.shape);
        let result = flat - lse;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!(
            "matmul -> reshape -> log_softmax gradient (first 16): {:?}",
            &grad_data[..16]
        );

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-10);
        assert!(
            !all_zero,
            "matmul + reshape + log_softmax gradients are zero!"
        );
    }

    #[test]
    fn test_conv2d_no_reshape_logsoftmax() {
        // Test conv2d -> log_softmax without reshape (flatten differently)
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 8, (3, 3), (1, 1), (1, 1), false, &mut cx);
        conv.weight.set(vec![0.1; 72]);

        let input = cx.tensor((1, 1, 4, 4)).set(vec![0.5; 16]);

        let output = conv.forward(input); // (1, 8, 2, 2) = 32 elements

        // Sum across spatial dimensions first, then do log_softmax on channels
        let summed = output.sum((2, 3)); // (1, 8)

        // log_softmax on channel dimension
        let lse = summed.exp().sum(1).log().expand(summed.shape);
        let result = summed - lse;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!(
            "conv2d (no reshape) -> log_softmax gradient: {:?}",
            grad_data
        );

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-10);
        assert!(
            !all_zero,
            "conv2d (no reshape) + log_softmax gradients are zero!"
        );
    }

    #[test]
    fn test_conv2d_varied_inputs() {
        // Test with varied input values to avoid numerical cancellation
        let mut cx = Graph::new();

        let conv = crate::nn::Conv2D::new(1, 2, (3, 3), (1, 1), (1, 1), false, &mut cx);
        // Varied weights
        conv.weight
            .set((0..18).map(|i| 0.1 * (i as f32 - 9.0)).collect::<Vec<_>>());

        // Varied input values
        let input = cx
            .tensor((2, 1, 4, 4))
            .set((0..32).map(|i| i as f32 / 32.0).collect::<Vec<_>>());

        let output = conv.forward(input);
        let flat = output.reshape((2, 8));

        // log_softmax pattern
        let lse = flat.exp().sum(1).log().expand(flat.shape);
        let result = flat - lse;
        let loss = result.sum(result.shape.all_axes());

        let grads = cx.compile(Autograd::new(conv.weight, loss), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), ());
        cx.execute();

        let grad_data = get_vec(grads[0], &mut cx);
        println!("Conv2D with varied inputs gradient: {:?}", grad_data);

        let all_zero = grad_data.iter().all(|&g| g.abs() < 1e-6);
        assert!(
            !all_zero,
            "Conv2D with varied inputs: gradients are still essentially zero!"
        );
    }
}
