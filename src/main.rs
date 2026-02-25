use flate2::read::GzDecoder;
use ndarray::{Array1, Array2, Axis};
use plotters::prelude::*;
use rand::Rng;
use std::fs::File;
use std::io::{Read, Result};
use ndarray_npy::{write_npy, read_npy};

// MNIST data loader
struct MnistData {
    images: Array2<f32>,
    labels: Array1<u8>,
}

impl MnistData {
    fn load_images(path: &str) -> Result<Array2<f32>> {
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;

        // Parse MNIST image file format
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        assert_eq!(magic, 2051, "Invalid MNIST image file");

        let num_images = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
        let num_rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
        let num_cols =
            u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;

        let image_size = num_rows * num_cols;
        let mut images = Array2::<f32>::zeros((num_images, image_size));

        for i in 0..num_images {
            for j in 0..image_size {
                images[[i, j]] = buffer[16 + i * image_size + j] as f32 / 255.0;
            }
        }

        Ok(images)
    }

    fn load_labels(path: &str) -> Result<Array1<u8>> {
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer)?;

        // Parse MNIST label file format
        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        assert_eq!(magic, 2049, "Invalid MNIST label file");

        let num_labels = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
        let labels = Array1::from_vec(buffer[8..8 + num_labels].to_vec());

        Ok(labels)
    }

    fn load(images_path: &str, labels_path: &str) -> Result<Self> {
        Ok(MnistData {
            images: Self::load_images(images_path)?,
            labels: Self::load_labels(labels_path)?,
        })
    }
}

// Neural Network
struct NeuralNetwork {
    weights1: Array2<f32>,
    bias1: Array1<f32>,
    weights2: Array2<f32>,
    bias2: Array1<f32>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / input_size as f32).sqrt();
        let weights1 = Array2::from_shape_fn((input_size, hidden_size), |_| {
            rng.gen_range(-scale1..scale1)
        });

        let scale2 = (2.0 / hidden_size as f32).sqrt();
        let weights2 = Array2::from_shape_fn((hidden_size, output_size), |_| {
            rng.gen_range(-scale2..scale2)
        });

        NeuralNetwork {
            weights1,
            bias1: Array1::zeros(hidden_size),
            weights2,
            bias2: Array1::zeros(output_size),
        }
    }

    fn relu(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|a| a.max(0.0))
    }

    fn relu_derivative(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 })
    }

    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone();
        for mut row in result.axis_iter_mut(Axis(0)) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.mapv_inplace(|a| (a - max).exp());
            let sum: f32 = row.sum();
            row.mapv_inplace(|a| a / sum);
        }
        result
    }

    fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // Hidden layer
        let z1 = input.dot(&self.weights1) + &self.bias1;
        let a1 = Self::relu(&z1);

        // Output layer
        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let a2 = Self::softmax(&z2);

        (a1, a2)
    }

    fn train(
        &mut self,
        images: &Array2<f32>,
        labels: &Array1<u8>,
        epochs: usize,
        learning_rate: f32,
        batch_size: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let num_samples = images.nrows();
        let mut loss_history = Vec::new();
        let mut accuracy_history = Vec::new();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            for batch_start in (0..num_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_samples);
                let batch_images = images.slice(s![batch_start..batch_end, ..]).to_owned();
                let batch_labels = labels.slice(s![batch_start..batch_end]).to_owned();

                // Forward pass
                let (hidden, output) = self.forward(&batch_images);

                // Calculate loss and accuracy
                for (i, &label) in batch_labels.iter().enumerate() {
                    let pred = output
                        .row(i)
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap() as u8;

                    if pred == label {
                        correct += 1;
                    }

                    total_loss -= (output[[i, label as usize]] + 1e-10).ln();
                }

                // Backward pass
                let batch_size_f = (batch_end - batch_start) as f32;

                // Output layer gradient
                let mut output_grad = output.clone();
                for (i, &label) in batch_labels.iter().enumerate() {
                    output_grad[[i, label as usize]] -= 1.0;
                }
                output_grad /= batch_size_f;

                // Gradients for weights2 and bias2
                let grad_weights2 = hidden.t().dot(&output_grad);
                let grad_bias2 = output_grad.sum_axis(Axis(0));

                // Hidden layer gradient
                let hidden_grad = output_grad.dot(&self.weights2.t());
                let hidden_grad = hidden_grad * Self::relu_derivative(&hidden);

                // Gradients for weights1 and bias1
                let grad_weights1 = batch_images.t().dot(&hidden_grad);
                let grad_bias1 = hidden_grad.sum_axis(Axis(0));

                // Update weights
                self.weights1 = &self.weights1 - learning_rate * grad_weights1;
                self.bias1 = &self.bias1 - learning_rate * grad_bias1;
                self.weights2 = &self.weights2 - learning_rate * grad_weights2;
                self.bias2 = &self.bias2 - learning_rate * grad_bias2;
            }

            let avg_loss = total_loss / num_samples as f32;
            let accuracy = 100.0 * correct as f32 / num_samples as f32;

            loss_history.push(avg_loss);
            accuracy_history.push(accuracy);

            println!(
                "Epoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch + 1,
                epochs,
                avg_loss,
                accuracy
            );
        }

        (loss_history, accuracy_history)
    }

    fn test(&self, images: &Array2<f32>, labels: &Array1<u8>) -> f32 {
        let (_, output) = self.forward(images);
        let mut correct = 0;

        for (i, &label) in labels.iter().enumerate() {
            let pred = output
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap() as u8;

            if pred == label {
                correct += 1;
            }
        }

        100.0 * correct as f32 / labels.len() as f32
    }

    fn save(&self, path: &str) -> Result<()> {
        println!("Saving model to {}...", path);
        
        // Create directory if it doesn't exist
        std::fs::create_dir_all(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        // Save weights and biases as .npy files (numpy format)
        write_npy(format!("{}/weights1.npy", path), &self.weights1)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        write_npy(format!("{}/bias1.npy", path), &self.bias1)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        write_npy(format!("{}/weights2.npy", path), &self.weights2)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        write_npy(format!("{}/bias2.npy", path), &self.bias2)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        // Save model config as JSON
        let config = serde_json::json!({
            "model_type": "mnist_classifier",
            "architecture": {
                "input_size": self.weights1.nrows(),
                "hidden_size": self.weights1.ncols(),
                "output_size": self.weights2.ncols(),
            },
            "framework": "rust_from_scratch",
        });
        
        let config_file = File::create(format!("{}/config.json", path))?;
        serde_json::to_writer_pretty(config_file, &config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        println!("Model saved successfully!");
        Ok(())
    }

    #[allow(dead_code)]
    fn load(path: &str) -> Result<Self> {
        println!("Loading model from {}...", path);
        
        let weights1: Array2<f32> = read_npy(format!("{}/weights1.npy", path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let bias1: Array1<f32> = read_npy(format!("{}/bias1.npy", path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let weights2: Array2<f32> = read_npy(format!("{}/weights2.npy", path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let bias2: Array1<f32> = read_npy(format!("{}/bias2.npy", path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        println!("Model loaded successfully!");
        Ok(NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
        })
    }
}

// Import the s! macro for slicing
use ndarray::s;

fn main() {
    println!("MNIST Neural Network in Rust");
    println!("==============================\n");

    // Load MNIST data
    println!("Loading training data...");
    let train_data = MnistData::load(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz",
    )
    .expect("Failed to load training data. Please download MNIST dataset to 'data' folder.");

    println!("Loading test data...");
    let test_data = MnistData::load(
        "data/t10k-images-idx3-ubyte.gz",
        "data/t10k-labels-idx1-ubyte.gz",
    )
    .expect("Failed to load test data. Please download MNIST dataset to 'data' folder.");

    println!("Training samples: {}", train_data.images.nrows());
    println!("Test samples: {}\n", test_data.images.nrows());

    // Create neural network
    let input_size = 784; // 28x28 images
    let hidden_size = 128;
    let output_size = 10; // digits 0-9

    println!(
        "Creating neural network: {} -> {} -> {}\n",
        input_size, hidden_size, output_size
    );

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size);

    // Train the network
    println!("Starting training...\n");
    let epochs = 10;
    let learning_rate = 0.1;
    let batch_size = 32;

    let (loss_history, accuracy_history) = nn.train(
        &train_data.images,
        &train_data.labels,
        epochs,
        learning_rate,
        batch_size,
    );

    // Test the network
    println!("\nEvaluating on test set...");
    let test_accuracy = nn.test(&test_data.images, &test_data.labels);
    println!("Test Accuracy: {:.2}%", test_accuracy);

    // Save the trained model
    println!();
    nn.save("model").expect("Failed to save model");

    // Plot training progress
    println!("\nGenerating training graphs...");
    plot_training_progress(&loss_history, &accuracy_history).expect("Failed to create plots");
    println!("Saved: images/training_progress.png");
}

fn plot_training_progress(loss_history: &[f32], accuracy_history: &[f32]) -> Result<()> {
    // Create images directory if it doesn't exist
    std::fs::create_dir_all("images")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    
    let root = BitMapBackend::new("images/training_progress.png", (1200, 500)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    let (left, right) = root.split_horizontally(600);

    // Loss plot
    let max_loss = loss_history.iter().cloned().fold(0.0f32, f32::max);
    let mut loss_chart = ChartBuilder::on(&left)
        .caption("Training Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(1..loss_history.len() + 1, 0.0..(max_loss * 1.1))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    loss_chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    loss_chart
        .draw_series(LineSeries::new(
            loss_history
                .iter()
                .enumerate()
                .map(|(i, &loss)| (i + 1, loss)),
            &RED,
        ))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Accuracy plot
    let min_acc = accuracy_history.iter().cloned().fold(100.0f32, f32::min);
    let mut acc_chart = ChartBuilder::on(&right)
        .caption("Training Accuracy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(1..accuracy_history.len() + 1, (min_acc * 0.95)..100.0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    acc_chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy (%)")
        .draw()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    acc_chart
        .draw_series(LineSeries::new(
            accuracy_history
                .iter()
                .enumerate()
                .map(|(i, &acc)| (i + 1, acc)),
            &BLUE,
        ))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    root.present()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    Ok(())
}
