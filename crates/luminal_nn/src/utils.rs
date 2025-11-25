//! Developer utilities for model inspection, serialization, and debugging.

use luminal::module::{param_dict, SerializeModule};
use luminal::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Model summary showing layer structure and parameter counts
#[derive(Debug)]
pub struct ModelSummary {
    pub layers: Vec<LayerInfo>,
    pub total_params: usize,
    pub trainable_params: usize,
}

/// Information about a single layer
#[derive(Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub param_count: usize,
    pub shape: Option<Vec<usize>>,
}

impl std::fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "┌─────────────────────────────────────────────────────────┐"
        )?;
        writeln!(
            f,
            "│                      Model Summary                       │"
        )?;
        writeln!(
            f,
            "├──────────────────────────────────┬──────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Layer                            │ Parameters           │"
        )?;
        writeln!(
            f,
            "├──────────────────────────────────┼──────────────────────┤"
        )?;

        for layer in &self.layers {
            let name = if layer.name.len() > 32 {
                format!("{}...", &layer.name[..29])
            } else {
                layer.name.clone()
            };
            let shape_str = layer
                .shape
                .as_ref()
                .map(|s| {
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("×")
                })
                .unwrap_or_default();
            let param_str = if shape_str.is_empty() {
                format!("{:>12}", format_number(layer.param_count))
            } else {
                format!("{:>12} ({})", format_number(layer.param_count), shape_str)
            };
            writeln!(f, "│ {:<32} │ {:<20} │", name, param_str)?;
        }

        writeln!(
            f,
            "├──────────────────────────────────┴──────────────────────┤"
        )?;
        writeln!(
            f,
            "│ Total parameters: {:>39} │",
            format_number(self.total_params)
        )?;
        writeln!(
            f,
            "│ Trainable parameters: {:>35} │",
            format_number(self.trainable_params)
        )?;
        writeln!(
            f,
            "└─────────────────────────────────────────────────────────┘"
        )?;
        Ok(())
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.2}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Generate a summary of the model's structure and parameters
pub fn model_summary(model: &impl SerializeModule, graph: &Graph) -> ModelSummary {
    let params = param_dict(model);

    // Sort by name for consistent ordering
    let sorted_params: BTreeMap<_, _> = params.into_iter().collect();

    let mut layers = Vec::new();
    let mut total_params = 0;

    for (name, node_id) in sorted_params {
        // Get tensor shape from graph
        let shape = graph.get_tensor_ref(node_id, 0).map(|tensor| {
            tensor
                .downcast_ref::<Vec<f32>>()
                .map(|v| vec![v.len()])
                .unwrap_or_default()
        });

        let param_count = shape.as_ref().map(|s| s.iter().product()).unwrap_or(0);
        total_params += param_count;

        layers.push(LayerInfo {
            name,
            param_count,
            shape,
        });
    }

    ModelSummary {
        layers,
        total_params,
        trainable_params: total_params, // For now, all params are trainable
    }
}

/// Count total parameters in a model
pub fn count_parameters(model: &impl SerializeModule) -> usize {
    param_dict(model).len()
}

// ============================================================================
// Checkpointing and Serialization
// ============================================================================

/// Save model weights to a binary file
///
/// Format: Simple binary format with weight names and data
pub fn save_weights<P: AsRef<Path>>(
    model: &impl SerializeModule,
    graph: &Graph,
    path: P,
) -> std::io::Result<()> {
    let params = param_dict(model);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write magic number and version
    writer.write_all(b"LUMI")?;
    writer.write_all(&1u32.to_le_bytes())?; // version 1

    // Write number of tensors
    let num_tensors = params.len() as u32;
    writer.write_all(&num_tensors.to_le_bytes())?;

    // Write each tensor
    for (name, node_id) in params {
        // Write name length and name
        let name_bytes = name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;

        // Get tensor data
        if let Some(tensor) = graph.get_tensor_ref(node_id, 0) {
            if let Some(data) = tensor.downcast_ref::<Vec<f32>>() {
                // Write data length and data
                writer.write_all(&(data.len() as u64).to_le_bytes())?;
                for val in data {
                    writer.write_all(&val.to_le_bytes())?;
                }
            } else {
                // Empty tensor
                writer.write_all(&0u64.to_le_bytes())?;
            }
        } else {
            // No tensor data available
            writer.write_all(&0u64.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Load model weights from a binary file
///
/// Returns a map of tensor names to their data
pub fn load_weights<P: AsRef<Path>>(path: P) -> std::io::Result<BTreeMap<String, Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"LUMI" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid weight file format",
        ));
    }

    // Read version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported weight file version: {}", version),
        ));
    }

    // Read number of tensors
    let mut num_tensors_bytes = [0u8; 4];
    reader.read_exact(&mut num_tensors_bytes)?;
    let num_tensors = u32::from_le_bytes(num_tensors_bytes);

    let mut weights = BTreeMap::new();

    for _ in 0..num_tensors {
        // Read name
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Read data
        let mut data_len_bytes = [0u8; 8];
        reader.read_exact(&mut data_len_bytes)?;
        let data_len = u64::from_le_bytes(data_len_bytes) as usize;

        let mut data = Vec::with_capacity(data_len);
        for _ in 0..data_len {
            let mut val_bytes = [0u8; 4];
            reader.read_exact(&mut val_bytes)?;
            data.push(f32::from_le_bytes(val_bytes));
        }

        weights.insert(name, data);
    }

    Ok(weights)
}

/// Apply loaded weights to a model
pub fn apply_weights(
    model: &impl SerializeModule,
    weights: &BTreeMap<String, Vec<f32>>,
    graph: &mut Graph,
) -> Result<usize, String> {
    let params = param_dict(model);
    let mut loaded = 0;
    let mut missing = Vec::new();

    for (name, node_id) in params {
        if let Some(data) = weights.get(&name) {
            // Get the Function op and set its data
            if let Some(op) = graph.graph.node_weight_mut(node_id) {
                if let Some(func) = op.as_any_mut().downcast_mut::<Function>() {
                    let data_clone = data.clone();
                    func.1 = Box::new(move |_| vec![Tensor::new(data_clone.clone())]);
                    loaded += 1;
                }
            }
        } else {
            missing.push(name);
        }
    }

    if !missing.is_empty() {
        return Err(format!(
            "Missing weights for {} parameters: {:?}",
            missing.len(),
            missing
        ));
    }

    Ok(loaded)
}

// ============================================================================
// Checkpoint Manager
// ============================================================================

/// Manages model checkpoints for training
pub struct CheckpointManager {
    /// Directory to save checkpoints
    pub save_dir: String,
    /// Maximum number of checkpoints to keep
    pub max_to_keep: usize,
    /// Saved checkpoint paths (oldest first)
    checkpoints: Vec<String>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(save_dir: impl Into<String>, max_to_keep: usize) -> Self {
        let save_dir = save_dir.into();
        std::fs::create_dir_all(&save_dir).ok();
        Self {
            save_dir,
            max_to_keep,
            checkpoints: Vec::new(),
        }
    }

    /// Save a checkpoint
    pub fn save(
        &mut self,
        model: &impl SerializeModule,
        graph: &Graph,
        step: usize,
    ) -> std::io::Result<String> {
        let path = format!("{}/checkpoint_{:08}.bin", self.save_dir, step);
        save_weights(model, graph, &path)?;

        self.checkpoints.push(path.clone());

        // Remove old checkpoints if we exceed max_to_keep
        while self.checkpoints.len() > self.max_to_keep {
            if let Some(old_path) = self.checkpoints.first() {
                std::fs::remove_file(old_path).ok();
                self.checkpoints.remove(0);
            }
        }

        Ok(path)
    }

    /// Load the latest checkpoint
    pub fn load_latest(&self) -> std::io::Result<Option<BTreeMap<String, Vec<f32>>>> {
        if let Some(path) = self.checkpoints.last() {
            Ok(Some(load_weights(path)?))
        } else {
            Ok(None)
        }
    }

    /// Get the latest checkpoint path
    pub fn latest_checkpoint(&self) -> Option<&str> {
        self.checkpoints.last().map(|s| s.as_str())
    }
}

// ============================================================================
// Debug Utilities
// ============================================================================

/// Print tensor statistics for debugging
pub fn tensor_stats(name: &str, data: &[f32]) {
    if data.is_empty() {
        println!("{}: empty tensor", name);
        return;
    }

    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();

    println!(
        "{}: shape=[{}], mean={:.6}, std={:.6}, min={:.6}, max={:.6}{}",
        name,
        data.len(),
        mean,
        std,
        min,
        max,
        if nan_count > 0 || inf_count > 0 {
            format!(" [WARN: {} NaN, {} Inf]", nan_count, inf_count)
        } else {
            String::new()
        }
    );
}

/// Check a tensor for NaN or Inf values
pub fn check_numerics(name: &str, data: &[f32]) -> Result<(), String> {
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();

    if nan_count > 0 || inf_count > 0 {
        Err(format!(
            "Tensor '{}' contains {} NaN and {} Inf values out of {} total",
            name,
            nan_count,
            inf_count,
            data.len()
        ))
    } else {
        Ok(())
    }
}

/// Validate gradient tensors for common issues
pub fn validate_gradients(name: &str, gradients: &[f32], threshold: f32) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for NaN/Inf
    let nan_count = gradients.iter().filter(|x| x.is_nan()).count();
    let inf_count = gradients.iter().filter(|x| x.is_infinite()).count();

    if nan_count > 0 {
        warnings.push(format!("{}: {} NaN gradient values", name, nan_count));
    }
    if inf_count > 0 {
        warnings.push(format!("{}: {} Inf gradient values", name, inf_count));
    }

    // Check for vanishing gradients
    let zero_count = gradients.iter().filter(|&&x| x.abs() < 1e-10).count();
    let zero_ratio = zero_count as f32 / gradients.len() as f32;
    if zero_ratio > 0.9 {
        warnings.push(format!(
            "{}: {:.1}% of gradients are near zero (vanishing gradients)",
            name,
            zero_ratio * 100.0
        ));
    }

    // Check for exploding gradients
    let max_grad = gradients
        .iter()
        .filter(|x| !x.is_nan() && !x.is_infinite())
        .fold(0.0f32, |a, &b| a.max(b.abs()));
    if max_grad > threshold {
        warnings.push(format!(
            "{}: max gradient magnitude {:.2} exceeds threshold {:.2} (exploding gradients)",
            name, max_grad, threshold
        ));
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.50K");
        assert_eq!(format_number(1_500_000), "1.50M");
        assert_eq!(format_number(1_500_000_000), "1.50B");
    }

    #[test]
    fn test_tensor_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        tensor_stats("test", &data); // Should print stats
    }

    #[test]
    fn test_check_numerics() {
        let good = vec![1.0, 2.0, 3.0];
        assert!(check_numerics("good", &good).is_ok());

        let bad = vec![1.0, f32::NAN, 3.0];
        assert!(check_numerics("bad", &bad).is_err());
    }

    #[test]
    fn test_validate_gradients() {
        let normal = vec![0.1, -0.2, 0.3];
        assert!(validate_gradients("normal", &normal, 100.0).is_empty());

        let vanishing = vec![1e-12, 1e-13, 1e-14];
        let warnings = validate_gradients("vanishing", &vanishing, 100.0);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_model_summary_display() {
        let summary = ModelSummary {
            layers: vec![
                LayerInfo {
                    name: "layer1.weight".to_string(),
                    param_count: 1024,
                    shape: Some(vec![32, 32]),
                },
                LayerInfo {
                    name: "layer2.weight".to_string(),
                    param_count: 4096,
                    shape: Some(vec![64, 64]),
                },
            ],
            total_params: 5120,
            trainable_params: 5120,
        };

        let output = format!("{}", summary);
        assert!(output.contains("Model Summary"));
        assert!(output.contains("layer1.weight"));
    }
}
