pub mod codegen;
pub mod debug;
pub mod egraph_debugger;
pub mod extract;
pub mod run;
pub mod translate;
pub mod utils;

#[cfg(all(test, feature = "metal"))]
mod e2e_tests;

#[cfg(feature = "cuda")]
use itertools::Itertools;
use luminal::prelude::*;
#[cfg(feature = "cuda")]
use luminal_cuda::CudaData;
#[cfg(feature = "metal")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLFunction};
#[cfg(feature = "cuda")]
use std::fs::File;
#[cfg(feature = "cuda")]
use std::io::Write;
use std::{collections::HashMap, fmt::Debug};

#[cfg(feature = "metal")]
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(feature = "metal")]
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
#[cfg(feature = "metal")]
pub type Function = Retained<ProtocolObject<dyn MTLFunction>>;
#[cfg(feature = "cuda")]
pub type Device = std::sync::Arc<luminal_cuda::CudaContext>;
#[cfg(feature = "cuda")]
pub type Buffer = cudarc::driver::CudaSlice<f32>;
// Stub types when no GPU backend is enabled
#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type Buffer = Vec<f32>;
#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type Device = ();

#[derive(Clone, PartialEq, Eq)]
pub enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub code: String,
    // launch params
    pub grid: (Expression, Expression, Expression),
    pub threadblock: (Expression, Expression, Expression),
    pub smem: Expression, // sizes of required shared memory buffers
    pub outputs: Vec<Expression>,
}

#[derive(Clone, Debug)]
pub enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { node: NodeIndex },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: String,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
    },
    Add,
    Mul,
    Max,
    Exp2,
    Log2,
    Recip,
    Sin,
    Neg,
    Sqrt,
    LessThan,
    Mod,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
    Custom(Kernel),
    Diff(String), // Diff a buffer
    Break,
    TCMatmul {
        a_k_stride: Expression,
        b_k_stride: Expression,
        a_inner_stride: Expression,
        b_inner_stride: Expression,
        c_inner_stride: Expression,
        k_outer_loops: Expression,
    },
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CompatKernel(Kernel, *mut Graph);

#[cfg(feature = "cuda")]
impl Operator for CompatKernel {
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use luminal_cuda::CudaData;
        let dyn_vars = &unsafe { self.1.as_ref().unwrap() }.dyn_map;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = cudarc::nvrtc::compile_ptx(&self.0.code).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let kernel = module.load_function("kernel_name").unwrap();
        let mut builder = stream.launch_builder(&kernel);

        // set inputs
        for input in inputs
            .iter()
            .map(|(b, _)| b.borrowed().downcast_ref::<CudaData<f32>>().unwrap())
        {
            builder.arg(&input.0);
        }

        // set output
        let mut out = self
            .0
            .outputs
            .iter()
            .map(|s| {
                stream
                    .alloc_zeros::<f32>(s.exec(dyn_vars).unwrap())
                    .unwrap()
            })
            .collect_vec();
        for o in &mut out {
            builder.arg(o);
        }

        // Set dispatch
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    self.0.grid.0.exec(dyn_vars).unwrap() as u32,
                    self.0.grid.1.exec(dyn_vars).unwrap() as u32,
                    self.0.grid.2.exec(dyn_vars).unwrap() as u32,
                ),
                block_dim: (
                    self.0.threadblock.0.exec(dyn_vars).unwrap() as u32,
                    self.0.threadblock.1.exec(dyn_vars).unwrap() as u32,
                    self.0.threadblock.2.exec(dyn_vars).unwrap() as u32,
                ),
                shared_mem_bytes: self.0.smem.exec(dyn_vars).unwrap() as u32,
            })
        }
        .unwrap();

        out.into_iter().map(|b| Tensor::new(CudaData(b))).collect()
    }
}

#[cfg(feature = "metal")]
impl Operator for CompatKernel {
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLBuffer as MTLBufferTrait, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
            MTLResourceOptions, MTLSize,
        };

        autoreleasepool(|_| {
            let dyn_vars = &unsafe { self.1.as_ref().unwrap() }.dyn_map;
            let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
            let queue = device.newCommandQueue().expect("No command queue");
            let command_buffer = queue.commandBuffer().expect("No command buffer");

            // Compile kernel
            let lib = device
                .newLibraryWithSource_options_error(&NSString::from_str(&self.0.code), None)
                .expect("Failed to compile Metal kernel");

            let function = lib
                .newFunctionWithName(&NSString::from_str("kernel_name"))
                .expect("Failed to find kernel_name function");

            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .expect("Failed to create pipeline state");

            let encoder = command_buffer
                .computeCommandEncoder()
                .expect("No compute encoder");
            encoder.setComputePipelineState(&pipeline);

            // Copy inputs to Metal buffers and set them
            let mut buffer_index = 0usize;
            let mut input_buffers = Vec::new();
            for (input, _shape) in &inputs {
                // Get input data as Vec<f32>
                let input_data = input.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                let metal_buf = unsafe {
                    use std::ffi::c_void;
                    use std::ptr::NonNull;
                    device
                        .newBufferWithBytes_length_options(
                            NonNull::new(input_data.as_ptr() as *mut c_void).unwrap(),
                            (input_data.len() * std::mem::size_of::<f32>()) as _,
                            MTLResourceOptions::StorageModeShared,
                        )
                        .expect("Failed to create input buffer")
                };
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&metal_buf), 0, buffer_index);
                }
                buffer_index += 1;
                input_buffers.push(metal_buf);
            }

            // Allocate and set output buffers
            let mut output_buffers = Vec::with_capacity(self.0.outputs.len());
            for size_expr in &self.0.outputs {
                let size = size_expr.exec(dyn_vars).unwrap();
                let buffer = device
                    .newBufferWithLength_options(
                        (size * std::mem::size_of::<f32>()) as _,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("Failed to allocate output buffer");
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buffer), 0, buffer_index);
                }
                buffer_index += 1;
                output_buffers.push((buffer, size));
            }

            // Dispatch
            let grid = MTLSize {
                width: self.0.grid.0.exec(dyn_vars).unwrap() as _,
                height: self.0.grid.1.exec(dyn_vars).unwrap() as _,
                depth: self.0.grid.2.exec(dyn_vars).unwrap() as _,
            };
            let threadgroup = MTLSize {
                width: self.0.threadblock.0.exec(dyn_vars).unwrap() as _,
                height: self.0.threadblock.1.exec(dyn_vars).unwrap() as _,
                depth: self.0.threadblock.2.exec(dyn_vars).unwrap() as _,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, threadgroup);
            encoder.endEncoding();

            command_buffer.commit();
            unsafe { command_buffer.waitUntilCompleted() };

            // Copy output data back to Vec<f32> and return as Tensors
            output_buffers
                .into_iter()
                .map(|(buf, size)| {
                    let mut data = vec![0f32; size];
                    unsafe {
                        let ptr = buf.contents().as_ptr() as *const f32;
                        std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);
                    }
                    Tensor::new(data)
                })
                .collect()
        })
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
impl Operator for CompatKernel {
    fn process(&mut self, _inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        panic!("CompatKernel requires 'cuda' or 'metal' feature to be enabled")
    }
}

pub fn custom_kernel(
    inputs: &[GraphTensor],
    kernel: Kernel,
    output_shape: impl ToShape,
    cx: &mut Graph,
) -> GraphTensor {
    let graph_ref: *mut Graph = cx;
    let mut kernel_op = cx.add_op(CompatKernel(kernel, graph_ref));
    for input in inputs {
        kernel_op = kernel_op.input(input.id, 0, input.shape);
    }
    let kernel_op = kernel_op.finish();
    GraphTensor::from_id(kernel_op, ShapeTracker::new(output_shape), cx)
}

#[derive(Debug)]
pub struct Diff {
    name: String,
}

#[cfg(feature = "cuda")]
impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Dump
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let buffer = inp[0].0.borrowed().downcast_ref::<CudaData<f32>>().unwrap();
        let data: Vec<f32> = stream.memcpy_dtov(&buffer.0).unwrap();
        let mut file = File::create(format!("{}.bin", self.name)).unwrap();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();
        vec![Tensor::new(buffer.clone())]
    }
}

#[cfg(feature = "metal")]
impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use std::fs::File;
        use std::io::Write;

        // Get the input data as Vec<f32>
        let data = inp[0]
            .0
            .borrowed()
            .downcast_ref::<Vec<f32>>()
            .expect("Input is not a Vec<f32>")
            .clone();

        // Write to file
        let mut file = File::create(format!("{}.bin", self.name)).unwrap();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();

        // Return the input data unchanged (pass-through)
        vec![Tensor::new(data)]
    }
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
impl Operator for Diff {
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        panic!("Diff requires 'cuda' or 'metal' feature to be enabled")
    }
}

pub trait GT2 {
    fn diff2(self, name: impl ToString) -> Self;
    fn graph_break(self) -> Self;
}

impl GT2 for GraphTensor {
    fn diff2(mut self, name: impl ToString) -> Self {
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        let id = self
            .graph()
            .add_op(Diff {
                name: name.to_string(),
            })
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }

    fn graph_break(mut self) -> Self {
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        let id = self
            .graph()
            .add_op(GraphBreak)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }
}

#[derive(Debug)]
pub struct GraphBreak;

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|i| i.0.cloned()).collect()
    }
}
