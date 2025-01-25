# Supervised Machine Learning (SLM) Project

## Overview
This project focuses on training a custom Transformer-based small language model (SLM) using a supervised machine learning approach. The model is designed to learn language patterns from a subset of the OpenWebText dataset, with the goal of producing a robust language representation that can generate meaningful text sequences. This repository contains the training code, dataset preparation scripts, and instructions for replicating the project.

---

## Features
- **Transformer-Based Architecture**: A custom implementation of a Transformer model with:
  - 6 layers
  - 512 embedding dimensions
  - 8 attention heads
  - 2048 hidden units
- **Custom Dataset**: Training data tokenized and prepared from the OpenWebText dataset.
- **Mixed Precision Training**: Efficient training with PyTorch's `torch.cuda.amp` for faster computations and reduced memory usage.
- **Scalable Hardware Usage**: Designed for high-performance training on multi-GPU setups, tested on an 8 x A100 instance.

---

## Dataset
The dataset used for this project is a 10 million token subset of the OpenWebText dataset. Text is tokenized using the `bert-base-uncased` tokenizer from the Hugging Face Transformers library.

### Dataset Preparation
1. The dataset is downloaded using the `datasets` library.
2. Text data is tokenized in batches for efficiency.
3. Tokenized data is saved as `tokenized_openwebtext.pt` for reuse.

---

## Model Architecture
The language model is implemented using PyTorch and includes the following components:
- **Token Embeddings**: Encodes input tokens into a dense vector representation.
- **Positional Embeddings**: Adds positional information to token embeddings.
- **Transformer Encoder**: Composed of multiple layers with self-attention and feed-forward sublayers.
- **Output Layer**: Projects the hidden representations back into the vocabulary size for token prediction.

---

## Training Setup
- **Hardware**: Training performed on an 8 x A100 GPU instance.
- **Batch Size**: 64 (configurable based on memory availability).
- **Optimizer**: AdamW with a learning rate of 5e-5.
- **Loss Function**: CrossEntropyLoss applied to token predictions.
- **Epochs**: 10
- **Mixed Precision**: Enabled using `torch.cuda.amp.GradScaler` for faster and memory-efficient training.

### Training Loop
1. Data is loaded in batches using a custom `TextDataset` and PyTorch's `DataLoader`.
2. Each batch processes input and target sequences to calculate loss.
3. Backpropagation is performed with mixed precision scaling for efficiency.
4. Model checkpoints are saved periodically.

---

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Datasets library

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/slm-project.git
   cd slm-project
   ```

Run the Jupyter Notebook:
```bash
jupyter notebook Small_Language_Model.ipynb
```

---

## Results
- Training logs include loss values per batch and per epoch.
- Final model checkpoint saved as `small_language_model.pt`.

### Example Logs
```
Epoch 1, Batch 0, Loss: 5.2345
Epoch 1, Batch 10, Loss: 4.8762
...
Epoch 1/10, Loss: 4.5123
```

---

## Future Work
- Expand the dataset size for better model generalization.
- Implement additional regularization techniques to prevent overfitting.
- Fine-tune the trained model on domain-specific datasets for specialized tasks.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments
Special thanks to:
- The OpenWebText team for providing the dataset.
- The PyTorch and Hugging Face communities for their excellent libraries.
- Lambda Labs for providing the computing resources.

---

## Contact
For questions or feedback, feel free to reach out:
- **Email**: your.email@example.com
- **GitHub**: [your-username](https://github.com/your-username)



