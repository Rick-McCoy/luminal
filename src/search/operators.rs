//! Search-related operators for custom kernels and debugging.
//!
//! These operators are used by the search infrastructure to execute
//! optimized kernels and debug graph execution.

use crate::prelude::*;

#[cfg(all(feature = "search", feature = "cuda"))]
use itertools::Itertools;
#[cfg(all(feature = "search", feature = "cuda"))]
use std::fs::File;
#[cfg(all(feature = "search", feature = "cuda"))]
use std::io::Write as IoWrite;

use super::types::Kernel;

/// CUDA buffer data wrapper (local definition to avoid dependency on luminal_cuda)
#[cfg(all(feature = "search", feature = "cuda"))]
#[derive(Debug)]
pub struct CudaData<T>(pub cudarc::driver::CudaSlice<T>);

#[cfg(all(feature = "search", feature = "cuda"))]
impl<T: cudarc::driver::DeviceRepr> Clone for CudaData<T> {
    fn clone(&self) -> Self {
        Self(self.0.try_clone().unwrap())
    }
}

#[cfg(all(feature = "search", feature = "cuda"))]
impl<T: cudarc::driver::DeviceRepr + Default + Clone + std::fmt::Debug + 'static> crate::op::Data
    for CudaData<T>
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Custom kernel operator for executing search-generated kernels.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CompatKernel(pub Kernel, pub *mut Graph);

#[cfg(all(feature = "search", feature = "cuda"))]
impl Operator for CompatKernel {
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let dyn_vars = &unsafe { self.1.as_ref().unwrap() }.dyn_map;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            &self.0.code,
            cudarc::nvrtc::CompileOptions {
                include_paths: vec!["/usr/include".into(), "/usr/local/cuda/include".into()],
                options: vec![
                    "--gpu-architecture=sm_75".into(),
                    "--relocatable-device-code=false".into(),
                    "--std=c++17".into(),
                ],
                ..Default::default()
            },
        )
        .unwrap();
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

#[cfg(all(feature = "search", feature = "metal"))]
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
            command_buffer.waitUntilCompleted();

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

#[cfg(not(all(feature = "search", any(feature = "cuda", feature = "metal"))))]
impl Operator for CompatKernel {
    fn process(&mut self, _inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        panic!("CompatKernel requires 'search' feature with 'cuda' or 'metal' feature to be enabled")
    }
}

/// Create a custom kernel operation in the graph.
#[cfg(feature = "search")]
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

/// Debug operator that dumps tensor data to a file.
#[derive(Debug)]
pub struct Diff {
    pub name: String,
}

#[cfg(all(feature = "search", feature = "cuda"))]
impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let buffer = inp[0]
            .0
            .borrowed()
            .downcast_ref::<CudaData<f32>>()
            .unwrap();
        let data: Vec<f32> = stream.clone_dtoh(&buffer.0).unwrap();
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

#[cfg(all(feature = "search", feature = "metal"))]
impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use std::fs::File;
        use std::io::Write;

        let data = inp[0]
            .0
            .borrowed()
            .downcast_ref::<Vec<f32>>()
            .expect("Input is not a Vec<f32>")
            .clone();

        let mut file = File::create(format!("{}.bin", self.name)).unwrap();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();

        vec![Tensor::new(data)]
    }
}

#[cfg(not(all(feature = "search", any(feature = "cuda", feature = "metal"))))]
impl Operator for Diff {
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        panic!("Diff requires 'search' feature with 'cuda' or 'metal' feature to be enabled")
    }
}

/// Extension trait for GraphTensor with search-related operations.
#[cfg(feature = "search")]
pub trait SearchGraphTensorExt {
    /// Dump tensor data to a file for debugging.
    fn diff2(self, name: impl ToString) -> Self;
    /// Insert a graph break point.
    fn graph_break(self) -> Self;
}

#[cfg(feature = "search")]
impl SearchGraphTensorExt for GraphTensor {
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

/// Graph break operator that passes data through unchanged.
#[derive(Debug)]
pub struct GraphBreak;

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|i| i.0.cloned()).collect()
    }
}
