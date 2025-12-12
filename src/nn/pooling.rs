use crate::prelude::*;

pub struct AvgPool2D {
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl AvgPool2D {
    pub fn new(kernel: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel, stride }
    }
}

impl SerializeModule for AvgPool2D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {
        // No parameters to serialize for average pooling
    }
}

impl AvgPool2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        // Input: (batch (optional), ch_in, dimx_in, dimy_in)
        let mut expanded = false;
        if input.shape.len() == 3 {
            // Expand batch
            input = input.expand_dim(0, 1);
            expanded = true;
        }
        let (batch, ch_in, dimx_in, dimy_in) = input.dims4();
        let dimx_out = ((dimx_in - self.kernel.0) / self.stride.0 + 1).simplify();
        let dimy_out = ((dimy_in - self.kernel.1) / self.stride.1 + 1).simplify();

        let output = input
            .pool_last_dim(self.kernel.1, self.stride.1, 1) // dilation = 1 for pooling
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(self.kernel.0, self.stride.0, 1)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((
                batch,
                ch_in,
                self.kernel.0 * self.kernel.1,
                dimx_out * dimy_out,
            ))
            .mean(2) // Average over the kernel dimension
            .reshape((batch, ch_in, dimx_out, dimy_out));

        if expanded {
            output.reshape((ch_in, dimx_out, dimy_out))
        } else {
            output
        }
    }
}

pub struct AdaptiveAvgPool2D {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl SerializeModule for AdaptiveAvgPool2D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {
        // No learnable parameters
    }
}

impl AdaptiveAvgPool2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        let mut expanded = false;
        // Handle missing batch dimension
        if input.shape.len() == 3 {
            input = input.expand_dim(0, 1);
            expanded = true;
        }

        // Extract dimensions
        let (batch, ch, h_in, w_in) = input.dims4();
        let (h_out, w_out) = self.output_size;

        let stride_h = (h_in / h_out).simplify();
        let stride_w = (w_in / w_out).simplify();
        let kernel_h = (h_in - (h_out - 1) * stride_h).simplify();
        let kernel_w = (w_in - (w_out - 1) * stride_w).simplify();

        // Two-stage pooling (Y then X), followed by averaging over the kernel window
        let mut output = input
            .pool_last_dim(kernel_w, stride_w, 1)
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(kernel_h, stride_h, 1)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((batch, ch, kernel_h * kernel_w, h_out * w_out))
            .mean(2)
            .reshape((batch, ch, h_out, w_out));

        // Remove batch dim if it was originally absent
        if expanded {
            output = output.reshape((ch, h_out, w_out));
        }

        output
    }
}

/// Max Pooling 2D
///
/// Applies max pooling over a 2D input (N, C, H, W).
/// Selects the maximum value in each kernel window.
pub struct MaxPool2D {
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl MaxPool2D {
    pub fn new(kernel: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel, stride }
    }

    /// Create with same kernel and stride (common case)
    pub fn same(size: usize) -> Self {
        Self {
            kernel: (size, size),
            stride: (size, size),
        }
    }
}

impl SerializeModule for MaxPool2D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {
        // No learnable parameters
    }
}

impl MaxPool2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        // Input: (batch (optional), ch_in, dimx_in, dimy_in)
        let mut expanded = false;
        if input.shape.len() == 3 {
            input = input.expand_dim(0, 1);
            expanded = true;
        }

        let (batch, ch_in, dimx_in, dimy_in) = input.dims4();
        let dimx_out = ((dimx_in - self.kernel.0) / self.stride.0 + 1).simplify();
        let dimy_out = ((dimy_in - self.kernel.1) / self.stride.1 + 1).simplify();

        // Pool to get windows, then take max
        let output = input
            .pool_last_dim(self.kernel.1, self.stride.1, 1)
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(self.kernel.0, self.stride.0, 1)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((
                batch,
                ch_in,
                self.kernel.0 * self.kernel.1,
                dimx_out * dimy_out,
            ))
            .max(2) // Max over the kernel dimension
            .reshape((batch, ch_in, dimx_out, dimy_out));

        if expanded {
            output.reshape((ch_in, dimx_out, dimy_out))
        } else {
            output
        }
    }
}

/// Global Max Pooling 2D
///
/// Reduces spatial dimensions to 1x1 by taking the max over H and W.
/// Output shape: (N, C, 1, 1) or (N, C) if squeeze=true
pub struct GlobalMaxPool2D {
    squeeze: bool,
}

impl GlobalMaxPool2D {
    pub fn new(squeeze: bool) -> Self {
        Self { squeeze }
    }
}

impl SerializeModule for GlobalMaxPool2D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {}
}

impl GlobalMaxPool2D {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        // Input: (N, C, H, W)
        let (batch, channels, height, width) = input.dims4();

        // Reshape to (N, C, H*W) and take max over last dim
        let reshaped = input.reshape((batch, channels, height * width));
        let pooled = reshaped.max(2);

        if self.squeeze {
            pooled
        } else {
            pooled.reshape((batch, channels, 1, 1))
        }
    }
}

/// Global Average Pooling 2D
///
/// Reduces spatial dimensions to 1x1 by taking the mean over H and W.
/// Output shape: (N, C, 1, 1) or (N, C) if squeeze=true
pub struct GlobalAvgPool2D {
    squeeze: bool,
}

impl GlobalAvgPool2D {
    pub fn new(squeeze: bool) -> Self {
        Self { squeeze }
    }
}

impl SerializeModule for GlobalAvgPool2D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {}
}

impl GlobalAvgPool2D {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        // Input: (N, C, H, W)
        let (batch, channels, height, width) = input.dims4();

        // Reshape to (N, C, H*W) and take mean over last dim
        let reshaped = input.reshape((batch, channels, height * width));
        let pooled = reshaped.mean(2);

        if self.squeeze {
            pooled
        } else {
            pooled.reshape((batch, channels, 1, 1))
        }
    }
}

/// Max Pooling 1D
///
/// Applies max pooling over a 1D input (N, C, L).
pub struct MaxPool1D {
    kernel: usize,
    stride: usize,
}

impl MaxPool1D {
    pub fn new(kernel: usize, stride: usize) -> Self {
        Self { kernel, stride }
    }

    pub fn same(size: usize) -> Self {
        Self {
            kernel: size,
            stride: size,
        }
    }
}

impl SerializeModule for MaxPool1D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {}
}

impl MaxPool1D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        // Input: (batch (optional), channels, length)
        let mut expanded = false;
        if input.shape.len() == 2 {
            input = input.expand_dim(0, 1);
            expanded = true;
        }

        let (batch, channels, length) = input.dims3();
        let out_length = ((length - self.kernel) / self.stride + 1).simplify();

        // Pool to get windows, then take max
        let output = input
            .pool_last_dim(self.kernel, self.stride, 1)
            .reshape((batch, channels, out_length, self.kernel))
            .max(3) // Max over kernel dimension
            .reshape((batch, channels, out_length));

        if expanded {
            output.reshape((channels, out_length))
        } else {
            output
        }
    }
}

/// Average Pooling 1D
pub struct AvgPool1D {
    kernel: usize,
    stride: usize,
}

impl AvgPool1D {
    pub fn new(kernel: usize, stride: usize) -> Self {
        Self { kernel, stride }
    }

    pub fn same(size: usize) -> Self {
        Self {
            kernel: size,
            stride: size,
        }
    }
}

impl SerializeModule for AvgPool1D {
    fn serialize(&self, _s: &mut crate::module::Serializer) {}
}

impl AvgPool1D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        let mut expanded = false;
        if input.shape.len() == 2 {
            input = input.expand_dim(0, 1);
            expanded = true;
        }

        let (batch, channels, length) = input.dims3();
        let out_length = ((length - self.kernel) / self.stride + 1).simplify();

        let output = input
            .pool_last_dim(self.kernel, self.stride, 1)
            .reshape((batch, channels, out_length, self.kernel))
            .mean(3)
            .reshape((batch, channels, out_length));

        if expanded {
            output.reshape((channels, out_length))
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let mut cx = Graph::new();

        // 1 batch, 1 channel, 4x4 input
        let input = cx.tensor((1, 1, 4, 4)).set(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);

        let pool = MaxPool2D::new((2, 2), (2, 2));
        let output = pool.forward(input).retrieve();

        cx.execute();

        // Expected: max of each 2x2 block
        // [6, 8]
        // [14, 16]
        let result = output.data();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 6.0);
        assert_eq!(result[1], 8.0);
        assert_eq!(result[2], 14.0);
        assert_eq!(result[3], 16.0);
    }

    #[test]
    fn test_maxpool1d() {
        let mut cx = Graph::new();

        let input = cx.tensor((1, 1, 6)).set(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);

        let pool = MaxPool1D::new(2, 2);
        let output = pool.forward(input).retrieve();

        cx.execute();

        // [max(1,3), max(2,5), max(4,6)] = [3, 5, 6]
        let result = output.data();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 6.0);
    }

    #[test]
    fn test_global_avg_pool() {
        let mut cx = Graph::new();

        let input = cx.tensor((1, 2, 2, 2)).set(vec![
            1.0, 2.0, 3.0, 4.0, // channel 0: mean = 2.5
            5.0, 6.0, 7.0, 8.0, // channel 1: mean = 6.5
        ]);

        let pool = GlobalAvgPool2D::new(true);
        let output = pool.forward(input).retrieve();

        cx.execute();

        let result = output.data();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.5).abs() < 1e-5);
        assert!((result[1] - 6.5).abs() < 1e-5);
    }

    #[test]
    fn test_global_max_pool() {
        let mut cx = Graph::new();

        let input = cx.tensor((1, 2, 2, 2)).set(vec![
            1.0, 2.0, 3.0, 4.0, // channel 0: max = 4
            5.0, 6.0, 7.0, 8.0, // channel 1: max = 8
        ]);

        let pool = GlobalMaxPool2D::new(true);
        let output = pool.forward(input).retrieve();

        cx.execute();

        let result = output.data();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 8.0);
    }
}
