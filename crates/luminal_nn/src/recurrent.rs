use luminal::prelude::*;
use std::ops::Neg;

/// Long Short-Term Memory (LSTM) layer
///
/// Processes sequential data with gating mechanisms for long-range dependencies.
/// Expects input of shape (batch, seq_len, input_size) and returns (batch, seq_len, hidden_size).
///
/// The LSTM uses the standard formulation:
/// - i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
/// - f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
/// - g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
/// - o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
/// - c_t = f_t * c_{t-1} + i_t * g_t
/// - h_t = o_t * tanh(c_t)
pub struct LSTM {
    pub weight_ih: GraphTensor,       // (4 * hidden_size, input_size)
    pub weight_hh: GraphTensor,       // (4 * hidden_size, hidden_size)
    pub bias_ih: Option<GraphTensor>, // (4 * hidden_size,)
    pub bias_hh: Option<GraphTensor>, // (4 * hidden_size,)
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight_ih: cx.named_tensor("LSTM Weight IH", (4 * hidden_size, input_size)),
            weight_hh: cx.named_tensor("LSTM Weight HH", (4 * hidden_size, hidden_size)),
            bias_ih: if bias {
                Some(cx.named_tensor("LSTM Bias IH", 4 * hidden_size))
            } else {
                None
            },
            bias_hh: if bias {
                Some(cx.named_tensor("LSTM Bias HH", 4 * hidden_size))
            } else {
                None
            },
            input_size,
            hidden_size,
        }
    }

    /// Initialize with Xavier/Glorot uniform initialization
    pub fn initialize(self) -> Self {
        use luminal::tests::random_vec_rng;
        use rand::rng;

        let mut rng = rng();

        // Xavier initialization
        let k = (1.0 / self.hidden_size as f32).sqrt();
        let scale = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| x * k * 2.0 - k).collect() };

        self.weight_ih.set(scale(random_vec_rng(
            4 * self.hidden_size * self.input_size,
            &mut rng,
        )));
        self.weight_hh.set(scale(random_vec_rng(
            4 * self.hidden_size * self.hidden_size,
            &mut rng,
        )));

        if let Some(b) = self.bias_ih {
            b.set(vec![0.0; 4 * self.hidden_size]);
        }
        if let Some(b) = self.bias_hh {
            b.set(vec![0.0; 4 * self.hidden_size]);
        }

        self
    }
}

impl SerializeModule for LSTM {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight_ih", self.weight_ih);
        s.tensor("weight_hh", self.weight_hh);
        if let Some(b) = self.bias_ih {
            s.tensor("bias_ih", b);
        }
        if let Some(b) = self.bias_hh {
            s.tensor("bias_hh", b);
        }
    }
}

impl LSTM {
    /// Forward pass for a single timestep
    ///
    /// Takes:
    /// - input: (batch, input_size)
    /// - h: hidden state (batch, hidden_size)
    /// - c: cell state (batch, hidden_size)
    ///
    /// Returns: (new_h, new_c)
    pub fn forward_step(
        &self,
        input: GraphTensor,
        h: GraphTensor,
        c: GraphTensor,
    ) -> (GraphTensor, GraphTensor) {
        // Compute gates
        // gates = input @ weight_ih.T + h @ weight_hh.T + biases
        let mut gates =
            input.matmul(self.weight_ih.permute((1, 0))) + h.matmul(self.weight_hh.permute((1, 0)));

        if let Some(b_ih) = self.bias_ih {
            gates = gates + b_ih.expand(gates.shape);
        }
        if let Some(b_hh) = self.bias_hh {
            gates = gates + b_hh.expand(gates.shape);
        }

        // Split gates into i, f, g, o
        let batch = input.dims()[0];

        // gates shape: (batch, 4 * hidden_size)
        // Split into 4 tensors of (batch, hidden_size)
        let i_gate = gates
            .reshape((batch, 4, self.hidden_size))
            .slice((.., 0..1, ..))
            .reshape((batch, self.hidden_size))
            .sigmoid();

        let f_gate = gates
            .reshape((batch, 4, self.hidden_size))
            .slice((.., 1..2, ..))
            .reshape((batch, self.hidden_size))
            .sigmoid();

        let g_gate = gates
            .reshape((batch, 4, self.hidden_size))
            .slice((.., 2..3, ..))
            .reshape((batch, self.hidden_size))
            .tanh();

        let o_gate = gates
            .reshape((batch, 4, self.hidden_size))
            .slice((.., 3..4, ..))
            .reshape((batch, self.hidden_size))
            .sigmoid();

        // Update cell and hidden state
        let new_c = f_gate * c + i_gate * g_gate;
        let new_h = o_gate * new_c.tanh();

        (new_h, new_c)
    }

    /// Create initial hidden and cell states (zeros)
    pub fn init_states(&self, batch_size: usize, cx: &mut Graph) -> (GraphTensor, GraphTensor) {
        let h0 = cx
            .named_tensor("LSTM h0", (batch_size, self.hidden_size))
            .set(vec![0.0; batch_size * self.hidden_size]);
        let c0 = cx
            .named_tensor("LSTM c0", (batch_size, self.hidden_size))
            .set(vec![0.0; batch_size * self.hidden_size]);
        (h0, c0)
    }
}

/// Gated Recurrent Unit (GRU) layer
///
/// A simpler alternative to LSTM with fewer parameters.
/// Expects input of shape (batch, seq_len, input_size).
///
/// The GRU uses:
/// - r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  (reset gate)
/// - z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  (update gate)
/// - n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  (new gate)
/// - h_t = (1 - z_t) * n_t + z_t * h_{t-1}
pub struct GRU {
    pub weight_ih: GraphTensor,       // (3 * hidden_size, input_size)
    pub weight_hh: GraphTensor,       // (3 * hidden_size, hidden_size)
    pub bias_ih: Option<GraphTensor>, // (3 * hidden_size,)
    pub bias_hh: Option<GraphTensor>, // (3 * hidden_size,)
    pub input_size: usize,
    pub hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight_ih: cx.named_tensor("GRU Weight IH", (3 * hidden_size, input_size)),
            weight_hh: cx.named_tensor("GRU Weight HH", (3 * hidden_size, hidden_size)),
            bias_ih: if bias {
                Some(cx.named_tensor("GRU Bias IH", 3 * hidden_size))
            } else {
                None
            },
            bias_hh: if bias {
                Some(cx.named_tensor("GRU Bias HH", 3 * hidden_size))
            } else {
                None
            },
            input_size,
            hidden_size,
        }
    }

    /// Initialize with Xavier/Glorot uniform initialization
    pub fn initialize(self) -> Self {
        use luminal::tests::random_vec_rng;
        use rand::rng;

        let mut rng = rng();

        let k = (1.0 / self.hidden_size as f32).sqrt();
        let scale = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| x * k * 2.0 - k).collect() };

        self.weight_ih.set(scale(random_vec_rng(
            3 * self.hidden_size * self.input_size,
            &mut rng,
        )));
        self.weight_hh.set(scale(random_vec_rng(
            3 * self.hidden_size * self.hidden_size,
            &mut rng,
        )));

        if let Some(b) = self.bias_ih {
            b.set(vec![0.0; 3 * self.hidden_size]);
        }
        if let Some(b) = self.bias_hh {
            b.set(vec![0.0; 3 * self.hidden_size]);
        }

        self
    }
}

impl SerializeModule for GRU {
    fn serialize(&self, s: &mut Serializer) {
        s.tensor("weight_ih", self.weight_ih);
        s.tensor("weight_hh", self.weight_hh);
        if let Some(b) = self.bias_ih {
            s.tensor("bias_ih", b);
        }
        if let Some(b) = self.bias_hh {
            s.tensor("bias_hh", b);
        }
    }
}

impl GRU {
    /// Forward pass for a single timestep
    ///
    /// Takes:
    /// - input: (batch, input_size)
    /// - h: hidden state (batch, hidden_size)
    ///
    /// Returns: new_h
    pub fn forward_step(&self, input: GraphTensor, h: GraphTensor) -> GraphTensor {
        let batch = input.dims()[0];

        // Compute input transformation
        let mut x_t = input.matmul(self.weight_ih.permute((1, 0)));
        if let Some(b) = self.bias_ih {
            x_t = x_t + b.expand(x_t.shape);
        }

        // Compute hidden transformation
        let mut h_t = h.matmul(self.weight_hh.permute((1, 0)));
        if let Some(b) = self.bias_hh {
            h_t = h_t + b.expand(h_t.shape);
        }

        // Split into reset, update, new gates
        let x_r = x_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 0..1, ..))
            .reshape((batch, self.hidden_size));
        let x_z = x_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 1..2, ..))
            .reshape((batch, self.hidden_size));
        let x_n = x_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 2..3, ..))
            .reshape((batch, self.hidden_size));

        let h_r = h_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 0..1, ..))
            .reshape((batch, self.hidden_size));
        let h_z = h_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 1..2, ..))
            .reshape((batch, self.hidden_size));
        let h_n = h_t
            .reshape((batch, 3, self.hidden_size))
            .slice((.., 2..3, ..))
            .reshape((batch, self.hidden_size));

        // Gates
        let r = (x_r + h_r).sigmoid(); // reset gate
        let z = (x_z + h_z).sigmoid(); // update gate
        let n = (x_n + r * h_n).tanh(); // new gate

        // Output: interpolate between old h and new candidate
        (z.neg() + 1.0) * n + z * h
    }

    /// Create initial hidden state (zeros)
    pub fn init_state(&self, batch_size: usize, cx: &mut Graph) -> GraphTensor {
        cx.named_tensor("GRU h0", (batch_size, self.hidden_size))
            .set(vec![0.0; batch_size * self.hidden_size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_step() {
        let mut cx = Graph::new();

        let lstm = LSTM::new(4, 8, true, &mut cx);
        lstm.weight_ih.set(vec![0.1; 4 * 8 * 4]);
        lstm.weight_hh.set(vec![0.1; 4 * 8 * 8]);
        lstm.bias_ih.unwrap().set(vec![0.0; 4 * 8]);
        lstm.bias_hh.unwrap().set(vec![0.0; 4 * 8]);

        let input = cx.tensor((2, 4)).set(vec![1.0; 8]);
        let (h0, c0) = lstm.init_states(2, &mut cx);

        let (h1, c1) = lstm.forward_step(input, h0, c0);
        h1.retrieve();
        c1.retrieve();

        cx.execute();

        // Check output shapes are correct
        assert_eq!(h1.data().len(), 2 * 8);
        assert_eq!(c1.data().len(), 2 * 8);

        // Values should be non-zero after processing
        assert!(h1.data().iter().all(|&x| x != 0.0));
    }

    #[test]
    fn test_gru_step() {
        let mut cx = Graph::new();

        let gru = GRU::new(4, 8, true, &mut cx);
        gru.weight_ih.set(vec![0.1; 3 * 8 * 4]);
        gru.weight_hh.set(vec![0.1; 3 * 8 * 8]);
        gru.bias_ih.unwrap().set(vec![0.0; 3 * 8]);
        gru.bias_hh.unwrap().set(vec![0.0; 3 * 8]);

        let input = cx.tensor((2, 4)).set(vec![1.0; 8]);
        let h0 = gru.init_state(2, &mut cx);

        let h1 = gru.forward_step(input, h0);
        h1.retrieve();

        cx.execute();

        assert_eq!(h1.data().len(), 2 * 8);
        assert!(h1.data().iter().all(|&x| x != 0.0));
    }
}
