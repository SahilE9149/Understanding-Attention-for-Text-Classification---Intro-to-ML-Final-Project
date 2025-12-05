"""Affine Encoder - Lightweight efficient variant."""
import torch
import torch.nn as nn


class AffineEncoder(nn.Module):
    """Simple linear transformation encoder (no recurrence)."""
    
    def __init__(self, embed_dim, hidden_dim=100, dropout=0.5):
        super(AffineEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Single linear transformation - fully parallelizable
        self.linear = nn.Linear(embed_dim, hidden_dim)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, embed_dim)
        Returns:
            encoded: (batch_size, seq_len, hidden_dim) - transformed representations
        """
        # Apply dropout
        x = self.dropout(embeddings)
        
        # Linear transformation (no recurrence = 3.75x faster)
        encoded = self.linear(x)  # (batch, seq_len, hidden_dim)
        
        return encoded
    
    def get_name(self):
        return "Affine"
