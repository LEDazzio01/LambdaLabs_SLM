# Small Language Model Project

This repository contains the implementation of a simple Transformer-based small language model, designed for educational purposes and experimentation. The project demonstrates the use of PyTorch for building and training language models.

## Features

- Transformer-based architecture with token and positional embeddings.
- Configurable model hyperparameters including embedding size, number of layers, and heads.
- Mixed precision training for efficient GPU utilization.
- Adaptive learning rate scheduling with `ReduceLROnPlateau`.
- Training and validation on dummy datasets.
- Gradient clipping to prevent exploding gradients.
- Saves model checkpoints for resuming training.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- tqdm

To install dependencies:
```bash
pip install torch tqdm
```

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Run the Model:**
   ```bash
   python Small_Language_Model.py
   ```

3. **Outputs:**
   - Training progress with batch-wise loss.
   - Validation loss and accuracy at the end of training.
   - Model checkpoint saved as `simple_language_model.pt`.

## Model Architecture

- **Embeddings:**
  - Token embedding: Maps vocabulary tokens to dense vector representations.
  - Positional embedding: Adds positional information to token embeddings.

- **Transformer Encoder:**
  - Multiple self-attention layers.
  - Configurable number of layers and heads.

- **Output Layer:**
  - Fully connected layer to map embeddings to vocabulary size.

## Hyperparameters

| Parameter       | Value   |
|-----------------|---------|
| Vocabulary Size | 10,000  |
| Embedding Dim   | 128     |
| Number of Heads | 4       |
| Layers          | 2       |
| Hidden Dim      | 512     |
| Max Seq Length  | 128     |
| Batch Size      | 32      |
| Learning Rate   | 1e-3    |
| Epochs          | 5       |

## Known Issues

- Validation accuracy remains low due to dummy dataset.
- Requires substantial GPU resources for larger datasets or experiments.

## Future Improvements

- Implement a real-world dataset.
- Add support for CPU-only training for resource-constrained environments.
- Explore alternative optimizers like AdamW.
- Integrate metrics like perplexity for deeper evaluation.

## Acknowledgments

This project was developed as a learning tool to understand Transformer architectures and PyTorch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




