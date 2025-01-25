
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Define a simple Transformer-based language model
class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_len):
        super(SmallLanguageModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits

# Dataset preparation
class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.cat([tokens, torch.zeros(block_size, dtype=torch.long)])
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        input_ids = self.tokens[idx : idx + self.block_size]
        target_ids = self.tokens[idx + 1 : idx + self.block_size + 1]
        return input_ids, target_ids

# Hyperparameters
vocab_size = 10000
embed_dim = 128
num_heads = 4
num_layers = 2
hidden_dim = 512
max_seq_len = 128
batch_size = 32
learning_rate = 1e-3
epochs = 5
block_size = max_seq_len - 1

# Dummy dataset for simplicity
print("Creating dummy dataset...", flush=True)
tokens = torch.randint(0, vocab_size, (10000,))
print("Dummy dataset created.", flush=True)

# Split dataset into training and validation sets
train_size = int(0.8 * len(tokens))
train_tokens, val_tokens = tokens[:train_size], tokens[train_size:]
train_dataset = TextDataset(train_tokens, block_size=block_size)
val_dataset = TextDataset(val_tokens, block_size=block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

# Model, loss function, and optimizer
model = SmallLanguageModel(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_len).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# Training loop
print("Starting training...")
scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for batch_idx, (input_ids, target_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}", flush=True)

    scheduler.step(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f}s. Avg Loss: {epoch_loss / len(train_loader):.4f}")

# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch + 1  # Save current epoch
}, "simple_language_model.pt")
print("Model training complete and saved to 'simple_language_model.pt'.")

# Validation step
print("Starting validation...")
model.eval()
with torch.no_grad():
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        val_loss += loss.item()

        # Accuracy calculation
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == target_ids).sum().item()
        total += target_ids.numel()

        if batch_idx % 10 == 0:
            print(f"Validation Batch {batch_idx}/{len(val_loader)} Loss: {loss.item():.4f}", flush=True)

    accuracy = correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}", flush=True)
    print(f"Validation Accuracy: {accuracy:.4f}", flush=True)
