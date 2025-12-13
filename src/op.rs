use std::{
    any::Any,
    borrow::BorrowMut,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::prelude::*;

use dyn_clone::{clone_trait_object, DynClone};
use rustc_hash::FxHashMap;

/// A tensor with data. The data can be anything that implements the Data trait
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Box<dyn Data>,
}

impl Tensor {
    pub fn new<T: Data>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }
    pub fn downcast_ref<T: Data>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref()
    }
    pub fn downcast_mut<T: Data>(&mut self) -> Option<&mut T> {
        self.data.as_any_mut().downcast_mut()
    }
    pub fn is<T: Data>(&self) -> bool {
        self.data.as_any().is::<T>()
    }
}

/// Some sort of data, for instance a Vec<f32> on CPU, CudaSlice<f32> on Nvidia GPUs, or metal::Buffer for Apple GPUs
pub trait Data: Any + Debug + DynClone {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

clone_trait_object!(Data);

impl Data for Vec<f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Either an owned or borrowed tensor that gets consumed by ops
pub enum InputTensor<'a> {
    /// An owned tensor
    Owned(Tensor),
    /// A borrowed tensor
    Borrowed(&'a Tensor),
}

impl<'a> InputTensor<'a> {
    /// Borrow the tensor
    pub fn borrowed(&'a self) -> &'a Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t,
        }
    }

    /// Unwrap or clone the tensor, depending on if it's owned or not
    pub fn cloned(self) -> Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t.clone(),
        }
    }
}

/// The main operator trait.
///
/// Defines an operator that takes in a vector of input tensors and shapes and produces a vector of output tensors
pub trait Operator: Debug + as_any::AsAny {
    /// Process the input tensors and produce output tensors
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>;
    /// Implement custom functionality
    #[allow(unused)]
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        None
    }
}

impl<T: Operator> Operator for Box<T> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self, key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self, inp)
    }
}
impl<T: Operator> Operator for Arc<Mutex<T>> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self.lock().unwrap().borrow_mut(), key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self.lock().unwrap().borrow_mut(), inp)
    }
}

/// An opaque function running on CPU that takes in Vec<f32> tensors and outputs Vec<f32> tensors
#[allow(clippy::type_complexity)]
pub struct Function(
    pub String,
    pub Box<dyn Fn(Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>>,
);

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Operator for Function {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        (self.1)(inp)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A constant value placed on the graph at runtime. Can either be an expression evaluated at runtime, or a constant float
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    Expression(Expression),
    Float(f32),
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        ConstantValue::Float(value)
    }
}
impl From<f64> for ConstantValue {
    fn from(value: f64) -> Self {
        ConstantValue::Float(value as f32)
    }
}
impl<T: Into<Expression>> From<T> for ConstantValue {
    fn from(value: T) -> Self {
        ConstantValue::Expression(value.into())
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq)]
pub struct Constant(pub ConstantValue, pub *const FxHashMap<char, usize>);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant(",)?;
        match &self.0 {
            ConstantValue::Expression(e) => e.fmt(f)?,
            ConstantValue::Float(fl) => fl.fmt(f)?,
        }
        write!(f, ")")
    }
}

impl Operator for Constant {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor::new(vec![match &self.0 {
            ConstantValue::Expression(e) => {
                e.exec_float(unsafe { self.1.as_ref().unwrap() }).unwrap() as f32
            }
            ConstantValue::Float(f) => *f,
        }])]
    }
}

/// Graph break for chunking search graphs
#[derive(Clone, PartialEq)]
pub struct GraphBreak;
impl Debug for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|(t, _)| t.cloned()).collect() // inefficient, but we don't care as this won't execute on the kernel
    }
}

// Unary Op (A -> A)

/// Ensure a tensor is contiguously layed out in memory. May involve copying
#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Copy data over to new tensor
        let inp_data = get_vec(&inp[0].0);
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl Operator for Log2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).log2();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2;
impl Operator for Exp2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).exp2();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin;
impl Operator for Sin {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sin();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip;
impl Operator for Recip {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).recip();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sqrt();
        }
        vec![Tensor::new(out_data)]
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add;
impl Operator for Add {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) + get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul;
impl Operator for Mul {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) * get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod;
impl Operator for Mod {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) % get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan;
impl Operator for LessThan {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = (get_index(lhs, &lexpr, &mut stack, i) < get_index(rhs, &rexpr, &mut stack, i))
                as i32 as f32;
        }
        vec![Tensor::new(out_data)]
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![0.0; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    result[i * back_size + j] += get_index(input, &expr, &mut stack, orig_index);
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![-f32::INFINITY; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    result[new_index] =
                        result[new_index].max(get_index(input, &expr, &mut stack, orig_index));
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap()
}

fn get_index(
    data: &[f32],
    (ind, val): &(Expression, Expression),
    stack: &mut Vec<i64>,
    index: usize,
) -> f32 {
    if val.exec_single_var_stack(index, stack) != 0 {
        let i = ind.exec_single_var_stack(index, stack);
        data[i]
    } else {
        0.0
    }
}

/// Pool elements along the last dimension, exposing windows as a new dimension.
///
/// This is a primitive operation with explicit gradient support. It transforms
/// input of shape `(..., N)` to output of shape `(..., num_windows, kernel_size)`.
///
/// For sliding windows with stride, each input element can contribute to multiple
/// output positions. The gradient correctly accumulates contributions via scatter-add.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolLastDim {
    /// Size of the pooling kernel
    pub kernel: usize,
    /// Stride between windows
    pub stride: usize,
    /// Dilation factor (spacing between kernel elements)
    pub dilation: usize,
}

impl PoolLastDim {
    pub fn new(kernel: usize, stride: usize, dilation: usize) -> Self {
        Self {
            kernel,
            stride,
            dilation,
        }
    }
}

impl Operator for PoolLastDim {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let input = get_vec(&inp[0].0);
        let shape = inp[0].1.shape_usize();
        let n_dims = shape.len();
        let input_last_dim = shape[n_dims - 1];

        // Effective kernel size with dilation
        let full_kernel = self.kernel + (self.kernel - 1) * (self.dilation - 1);

        // Number of output windows
        let num_windows = (input_last_dim - full_kernel) / self.stride + 1;

        // Output shape: (..., num_windows, kernel)
        let batch_size: usize = shape.iter().take(n_dims - 1).product::<usize>().max(1);
        let out_size = batch_size * num_windows * self.kernel;
        let mut out_data = vec![0.0; out_size];

        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        // For each batch element
        for b in 0..batch_size {
            let batch_offset = b * input_last_dim;
            let out_batch_offset = b * num_windows * self.kernel;

            // For each window
            for w in 0..num_windows {
                let window_start = w * self.stride;
                let out_window_offset = out_batch_offset + w * self.kernel;

                // For each element in the kernel
                for k in 0..self.kernel {
                    let input_idx = batch_offset + window_start + k * self.dilation;
                    let out_idx = out_window_offset + k;
                    out_data[out_idx] = get_index(input, &expr, &mut stack, input_idx);
                }
            }
        }

        vec![Tensor::new(out_data)]
    }
}

/// Gradient (backward) for PoolLastDim. Implements scatter-add.
///
/// Each input element can contribute to multiple output positions (overlapping windows).
/// The gradient accumulates contributions from all output positions that used each input.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolLastDimBackward {
    /// Size of the pooling kernel
    pub kernel: usize,
    /// Stride between windows
    pub stride: usize,
    /// Dilation factor
    pub dilation: usize,
    /// Original input last dimension size (needed to compute output shape)
    pub input_last_dim: usize,
}

impl PoolLastDimBackward {
    pub fn new(kernel: usize, stride: usize, dilation: usize, input_last_dim: usize) -> Self {
        Self {
            kernel,
            stride,
            dilation,
            input_last_dim,
        }
    }
}

impl Operator for PoolLastDimBackward {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Input is the upstream gradient with shape (..., num_windows, kernel)
        let grad_out = get_vec(&inp[0].0);
        let shape = inp[0].1.shape_usize();
        let n_dims = shape.len();

        // The gradient input has shape (..., num_windows, kernel)
        // We need to output gradient with shape (..., input_last_dim)
        // Handle case where n_dims might be < 2 (e.g., after reshape or contiguous)
        if n_dims < 2 {
            panic!(
                "PoolLastDimBackward expected input shape with at least 2 dimensions, got {:?}",
                shape
            );
        }
        let num_windows = shape[n_dims - 2];
        let kernel = shape[n_dims - 1];
        let batch_size: usize = shape.iter().take(n_dims - 2).product::<usize>().max(1);

        // Output gradient shape: (..., input_last_dim)
        let out_size = batch_size * self.input_last_dim;
        let mut grad_in = vec![0.0; out_size];

        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        // Scatter-add: for each output gradient element, add it to the corresponding input position
        for b in 0..batch_size {
            let grad_in_batch_offset = b * self.input_last_dim;
            let grad_out_batch_offset = b * num_windows * kernel;

            for w in 0..num_windows {
                let window_start = w * self.stride;
                let grad_out_window_offset = grad_out_batch_offset + w * kernel;

                for k in 0..kernel {
                    let input_pos = window_start + k * self.dilation;
                    if input_pos < self.input_last_dim {
                        let grad_out_idx = grad_out_window_offset + k;
                        let grad_in_idx = grad_in_batch_offset + input_pos;
                        grad_in[grad_in_idx] +=
                            get_index(grad_out, &expr, &mut stack, grad_out_idx);
                    }
                }
            }
        }

        vec![Tensor::new(grad_in)]
    }
}
