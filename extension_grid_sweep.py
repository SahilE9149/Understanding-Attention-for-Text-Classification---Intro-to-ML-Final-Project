"""
Extension 1 Experiment Runner: Modern Encoder Comparison Grid.
Tests LSTM, Affine, GRU, CNN on SST + IMDB with varying τ scales.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import csv
import os
from improved_attn_model_extended import ExtendedAttentionModel
from run_experiments import SeqDataset, collate_batch, load_seq_dataloaders

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameter grid for Extension 1
ENCODERS = ["lstm", "affine", "gru", "cnn"]
DATASETS = ["sst", "imdb"]
SCALING_FACTORS = [1, 10, 100]  # τ values: reduced for efficiency
L2_LAMBDA = 0.0  # No L2 for baseline comparison
NUM_CLASSES = {"sst": 2, "imdb": 2}
VOCAB_SIZE = 10000
EMBED_DIM = 100
HIDDEN_DIM = 100
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.01
EARLY_STOPPING_PATIENCE = 3


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits, labels) + model.l2_reg()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, eval_loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(eval_loader), correct / total


def compute_correlation(model, eval_loader, device):
    """Compute polarity-attention correlation r (Pearson)."""
    from scipy.stats import pearsonr
    
    model.eval()
    polarities = []
    attentions = []
    
    with torch.no_grad():
        for input_ids, labels in eval_loader:
            input_ids = input_ids.to(device)
            logits, attn, polarity = model(input_ids, return_attention=True)
            
            # Flatten for correlation
            attentions.extend(attn.cpu().numpy().flatten())
            polarities.extend(polarity[:, :, 1].cpu().numpy().flatten())  # Class 1 polarity
    
    if len(polarities) > 1:
        r, _ = pearsonr(polarities, attentions)
        return r
    return 0.0


def train_and_eval(dataset, encoder, scaling, num_classes=2):
    """
    Train and evaluate one configuration.
    Returns: (test_accuracy, polarity_attention_correlation, training_time_sec)
    """
    print(f"\n{'='*60}")
    print(f"Config: {dataset.upper()}, {encoder.upper()}, τ={scaling}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, dev_loader, test_loader, vocab_size, pad_idx = load_seq_dataloaders(
    dataset_name=dataset,
    batch_size=BATCH_SIZE,
    max_len=200,
)
    # Initialize model
    model = ExtendedAttentionModel(
    vocab_size=vocab_size,          # use real vocab size
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=num_classes,
    encoder_type=encoder,
    scaling=scaling,
    l2_lambda=L2_LAMBDA,
    dropout=DROPOUT,
).to(DEVICE)
    
    # Optimizer and loss
    optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_dev_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Dev Loss: {dev_loss:.4f}, Acc: {dev_acc:.4f}")
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    
    elapsed_time = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    correlation_r = compute_correlation(model, test_loader, DEVICE)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Polarity-Attention Correlation r: {correlation_r:.4f}")
    print(f"Training Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
    
    return test_acc, correlation_r, elapsed_time


def run_full_grid():
    """Run complete experimental grid and save results."""
    results = []
    results_file = "extension_results.csv"
    
    # Write CSV header
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "encoder", "scaling_tau", "test_accuracy",
            "correlation_r", "training_time_sec", "encoder_display"
        ])
    
    # Experiment grid: 4 encoders × 2 datasets × 3 scales = 24 configs
    total_configs = len(ENCODERS) * len(DATASETS) * len(SCALING_FACTORS)
    config_num = 0
    
    for dataset in DATASETS:
        for encoder in ENCODERS:
            for scaling in SCALING_FACTORS:
                config_num += 1
                print(f"\n[{config_num}/{total_configs}]", end="")
                
                try:
                    test_acc, corr_r, train_time = train_and_eval(
                        dataset=dataset,
                        encoder=encoder,
                        scaling=scaling,
                        num_classes=NUM_CLASSES[dataset]
                    )
                    
                    # Save result
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            dataset, encoder, scaling, f"{test_acc:.4f}",
                            f"{corr_r:.4f}", f"{train_time:.1f}",
                            encoder.upper()
                        ])
                    
                except Exception as e:
                    print(f"Error in config {dataset}, {encoder}, {scaling}: {e}")
    
    print(f"\n\n✅ All experiments complete! Results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    results_file = run_full_grid()
    print(f"\nGenerated: {results_file}")
