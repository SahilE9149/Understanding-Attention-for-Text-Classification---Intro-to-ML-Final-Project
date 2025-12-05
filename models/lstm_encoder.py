"""LSTM Encoder - Baseline from original paper."""
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Single-layer LSTM encoder for sequence encoding."""
    
    def __init__(self, embed_dim, hidden_dim=100, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # LSTM processes sequences recurrently
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, embed_dim)
        Returns:
            encoded: (batch_size, seq_len, hidden_dim) - sequence representations
        """
        # Apply dropout
        x = self.dropout(embeddings)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        
        return lstm_out
    
    def get_name(self):
        return "LSTM"
