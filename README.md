# Small Language Model

This repository contains the implementation of a small Transformer-based language model designed for educational and experimentation purposes. The model is implemented in PyTorch and supports training, validation, and saving/loading models.

---

## **Features**

- Transformer-based architecture with configurable parameters.
- Supports mixed-precision training for efficient GPU usage.
- Customizable dataset and training hyperparameters.
- Saves trained models locally or to Google Cloud Storage (GCS).
- Includes a dummy dataset generator for quick testing.

---

## **Model Architecture**

The model consists of:

1. **Token Embeddings**: Maps vocabulary indices to dense vectors.
2. **Positional Embeddings**: Adds positional information to token embeddings.
3. **Transformer Encoder Layers**: Self-attention mechanism to process sequences.
4. **Output Layer**: Maps hidden states to vocabulary logits for next-token prediction.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **Training the Model**

Run the script with the following command:
```bash
python small_language_model.py [OPTIONS]
```

### **Command-Line Arguments**
The script supports the following command-line arguments:

| Argument            | Type    | Default  | Description |
|---------------------|---------|----------|-------------|
| `--vocab_size`      | int     | `10000`  | Size of the vocabulary. |
| `--embed_dim`       | int     | `128`    | Dimensionality of token embeddings. |
| `--num_heads`       | int     | `4`      | Number of attention heads in each Transformer layer. |
| `--num_layers`      | int     | `2`      | Number of Transformer encoder layers. |
| `--hidden_dim`      | int     | `512`    | Hidden size of the feedforward network in Transformer layers. |
| `--max_seq_len`     | int     | `128`    | Maximum sequence length. |
| `--batch_size`      | int     | `32`     | Number of samples per batch. |
| `--learning_rate`   | float   | `1e-3`   | Learning rate for the optimizer. |
| `--epochs`          | int     | `5`      | Number of training epochs. |
| `--seed`            | int     | `42`     | Seed for reproducibility. |
| `--model_dir`       | str     | `.`      | Path to save the trained model. Supports GCS paths. |

### **Example Command**

```bash
python small_language_model.py --vocab_size 5000 --embed_dim 256 --num_heads 8 --num_layers 4 --batch_size 64 --epochs 10 --model_dir ./output
```

---

## **Output**

- **Model Checkpoint**: The trained model is saved as a `.pt` file in the specified `--model_dir`.
- If `--model_dir` points to a GCS bucket (e.g., `gs://bucket_name/`), the model will be uploaded to GCS.

---

## **Dummy Dataset**

The script uses a dummy dataset generated with:
```python
torch.randint(0, vocab_size, (10000,))
```

If you want to replace this with a real dataset, preprocess your text, tokenize it, and provide the tokenized sequences as input.

---

## **Validation Metrics**

During validation, the script computes:

1. **Validation Loss**: Average loss on the validation set.
2. **Accuracy**: Percentage of correctly predicted tokens.

---

## **Extending the Model**

### **Custom Dataset**
Replace the dummy dataset in the `train_and_evaluate` function with a real dataset. For example:

1. Preprocess your text data.
2. Tokenize the text using a tokenizer (e.g., Hugging Face's `tokenizers` library).
3. Convert tokenized text into `torch.Tensor` and use it with the `TextDataset` class.

### **Additional Features**
You can extend the model by:
- Adding dropout or layer normalization.
- Fine-tuning on specific tasks like text classification or machine translation.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contributions**

Contributions are welcome! Feel free to fork the repository and submit a pull request with your changes.

---

## **Contact**

For questions or feedback, please open an issue in the repository or reach out via LinkedIn.





