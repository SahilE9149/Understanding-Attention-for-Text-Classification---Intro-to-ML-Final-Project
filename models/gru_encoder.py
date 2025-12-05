"""GRU Encoder - Gated recurrent unit (lighter than LSTM)."""
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """GRU encoder - simpler recurrent unit than LSTM, fewer parameters."""
    
    def __init__(self, embed_dim, hidden_dim=100, dropout=0.5):
        super(GRUEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # GRU: 3 gates (reset, update) vs LSTM 4 gates (input, forget, output, cell)
        # ~30% fewer parameters than LSTM
        self.gru = nn.GRU(
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
            encoded: (batch_size, seq_len, hidden_dim)
        """
        # Apply dropout
        x = self.dropout(embeddings)
        
        # GRU encoding
        gru_out, h_n = self.gru(x)  # gru_out: (batch, seq_len, hidden_dim)
        
        return gru_out
    
    def get_name(self):
        return "GRU"
