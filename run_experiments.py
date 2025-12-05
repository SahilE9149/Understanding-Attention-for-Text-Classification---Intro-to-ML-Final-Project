import time
import pickle
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from improved_attn_model import ImprovedAttentionModel

# =========================
# Dataset utilities
# =========================

class SeqDataset(Dataset):
    """Wraps preprocessed sequence pickle files from data_processor.py"""
    def __init__(self, seq_path):
        with open(seq_path, "rb") as f:
            self.data = list(pickle.load(f))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch(batch, pad_idx=0, max_len=200):
    """Collate batch with padding to max_len"""
    xs, ys = zip(*batch)
    xs_trim = [x[:max_len] for x in xs]
    lengths = [len(x) for x in xs_trim]
    maxL = max(lengths) if lengths else 1
    
    padded = torch.full((len(xs_trim), maxL), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs_trim):
        padded[i, :len(x)] = x
    
    ys = torch.stack(ys)
    return padded, ys

def load_seq_dataloaders(dataset_name, batch_size=32, max_len=200):
    """Load train/dev/test data for SST, IMDB, or 20News"""
    root = "data/"
    name = dataset_name.lower()
    
    if name == "sst":
        base = root + "sst/"
    elif name == "imdb":
        base = root + "imdb/"
    elif name in ["20news", "20news-i", "20news_i"]:
        base = root + "20news/"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_pkl = base + "train_seq.pkl"
    dev_pkl = base + "dev_seq_.pkl"
    test_pkl = base + "test_seq.pkl"
    
    train_ds = SeqDataset(train_pkl)
    dev_ds = SeqDataset(dev_pkl)
    test_ds = SeqDataset(test_pkl)
    
    pad_idx = 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_idx, max_len)
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx, max_len)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx, max_len)
    )
    
    # Load vocab size
    try:
        with open(base + "vocab/local_dict.pkl", "rb") as f:
            vocab, word2id, id2word = pickle.load(f)
            vocab_size = len(vocab)
    except FileNotFoundError:
        vocab_size = 300000
    
    return train_loader, dev_loader, test_loader, vocab_size, pad_idx

# =========================
# Training / Evaluation
# =========================

def compute_corr(attn_list, pol_list):
    """Pearson correlation between attention and polarity"""
    attn_flat = [a.contiguous().view(-1) for a in attn_list]
    pol_flat = [p.contiguous().view(-1) for p in pol_list]
    
    attn_all = torch.cat(attn_flat).cpu().numpy()
    pol_all = torch.cat(pol_flat).cpu().numpy()
    
    if attn_all.size < 2:
        return 0.0
    
    return float(np.corrcoef(attn_all, pol_all)[0, 1])

def train_and_eval(
    dataset="sst",
    encoder_type="affine",
    scaling=10.0,
    l2_lambda=0.0,
    epochs=30,
    batch_size=32,
    max_len=200,
):
    """
    Train and evaluate model on given dataset/encoder/hyperparams
    
    Args:
        dataset: "sst", "imdb", or "20news"
        encoder_type: "lstm" or "affine"
        scaling: λ for attention scaling (0.001 to 10000)
        l2_lambda: L2 regularization on attention (0, 0.0001, 0.001, 0.01)
        epochs: number of training epochs
        batch_size: batch size
        max_len: max sequence length
    
    Returns:
        dict with keys: dataset, encoder, scaling, l2_lambda, test_acc, r_corr, train_time_sec, peak_mem_mb
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data
    train_iter, dev_iter, test_iter, vocab_size, pad_idx = load_seq_dataloaders(
        dataset_name=dataset,
        batch_size=batch_size,
        max_len=max_len
    )
    
    # 2. Determine task type
    is_multiclass = dataset.lower().startswith("20news")
    num_classes = 20 if is_multiclass else 1
    
    # 3. Build model
    model = ImprovedAttentionModel(
        vocab_size=vocab_size,
        embed_dim=100,
        hidden_dim=100,
        num_classes=num_classes,
        encoder_type=encoder_type,
        scaling_factor=scaling,
        l2_lambda=l2_lambda,
        padding_idx=pad_idx
    ).to(device)
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss() if is_multiclass else nn.BCELoss()
    
    process = psutil.Process()
    best_val_acc = 0.0
    best_state = None
    patience = 5
    no_improve = 0
    
    start = time.time()
    
    # 4. Training loop
    for epoch in range(epochs):
        model.train()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            
            # Normalize labels for multi-class
            if is_multiclass:
                y_min = y.min().item()
                y_max = y.max().item()
                if y_min < 0 or y_max >= num_classes:
                    y = y - y_min
            
            optimizer.zero_grad()
            prob, attn, pol = model(x)
            
            if is_multiclass:
                loss = criterion(prob, y) + model.l2_reg()
            else:
                loss = criterion(prob.squeeze(), y.float()) + model.l2_reg()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 5. Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dev_iter:
                x = x.to(device)
                y = y.to(device)
                
                if is_multiclass:
                    y_min = y.min().item()
                    y_max = y.max().item()
                    if y_min < 0 or y_max >= num_classes:
                        y = y - y_min
                
                prob, _, _ = model(x)
                
                if is_multiclass:
                    pred = prob.argmax(dim=1)
                else:
                    pred = (prob.squeeze() >= 0.5).long()
                
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total if total > 0 else 0.0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
        
        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc*100:.2f}%")
    
    train_time = time.time() - start
    peak_mem = process.memory_info().rss / (1024 ** 2)  # MB
    
    # 6. Test with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    correct, total = 0, 0
    attn_list, pol_list = [], []
    
    with torch.no_grad():
        for x, y in test_iter:
            x = x.to(device)
            y = y.to(device)
            
            if is_multiclass:
                y_min = y.min().item()
                y_max = y.max().item()
                if y_min < 0 or y_max >= num_classes:
                    y = y - y_min
            
            prob, attn, pol = model(x)
            
            if is_multiclass:
                pred = prob.argmax(dim=1)
            else:
                pred = (prob.squeeze() >= 0.5).long()
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            attn_list.append(attn.cpu())
            pol_list.append(pol.cpu())
    
    test_acc = correct / total if total > 0 else 0.0
    r_corr = compute_corr(attn_list, pol_list)
    
    result = {
        "dataset": dataset,
        "encoder": encoder_type,
        "scaling": scaling,
        "l2_lambda": l2_lambda,
        "test_acc": test_acc,
        "r_corr": r_corr,
        "train_time_sec": train_time,
        "peak_mem_mb": peak_mem,
    }
    
    return result

def run_full_grid(
    datasets=("sst", "imdb", "20news"),
    encoders=("lstm", "affine"),
    scalings=(0.001, 1.0, 10.0, 20.0, 50.0, 100.0, 10000.0),
    l2_small=(0.0,),
):
    """
    Run full grid over all combinations.
    
    For LSTM: no L2 regularization (l2_small is ignored)
    For Affine on SST/20News: sweep L2 values
    
    Usage:
        # Full baseline grid (no L2)
        df = run_full_grid(l2_small=(0.0,))
        
        # Affine + L2 on small datasets
        df = run_full_grid(
            datasets=("sst", "20news"),
            encoders=("affine",),
            scalings=(10.0, 20.0),
            l2_small=(0.0, 0.0001, 0.001, 0.01)
        )
    """
    results = []
    
    for ds in datasets:
        for enc in encoders:
            for sc in scalings:
                # For affine on SST/20News, sweep L2
                if enc == "affine" and ds.lower() in ["sst", "20news", "20news-i"]:
                    for lam in l2_small:
                        print(f"\n== {ds} | {enc} | λ={sc} | L2={lam} ==")
                        res = train_and_eval(
                            dataset=ds,
                            encoder_type=enc,
                            scaling=sc,
                            l2_lambda=lam,
                            epochs=30,
                        )
                        print(res)
                        results.append(res)
                else:
                    # For LSTM or other combinations, use L2=0
                    print(f"\n== {ds} | {enc} | λ={sc} ==")
                    res = train_and_eval(
                        dataset=ds,
                        encoder_type=enc,
                        scaling=sc,
                        l2_lambda=0.0,
                        epochs=30,
                    )
                    print(res)
                    results.append(res)
    
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # Quick test on SST with Affine
    print("Quick test: SST with Affine encoder")
    res = train_and_eval(
        dataset="sst",
        encoder_type="affine",
        scaling=10.0,
        l2_lambda=0.0,
        epochs=5,
        batch_size=32,
    )
    print("\nQuick run result:", res)
    
    # To run full grid:
    # df = run_full_grid(l2_small=(0.0,))
    # df.to_csv("results_all_datasets.csv", index=False)