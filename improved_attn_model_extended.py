"""Extended attention model with pluggable encoders for Extension 1."""
import torch
import torch.nn as nn
from models.lstm_encoder import LSTMEncoder
from models.affine_encoder import AffineEncoder
from models.gru_encoder import GRUEncoder
from models.cnn_encoder import CNNEncoder


class ExtendedAttentionModel(nn.Module):
    """
    Attention-based text classifier with pluggable encoder architecture.
    Supports LSTM, Affine, GRU, CNN encoders for comparative analysis.
    """
    
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=100, num_classes=2,
                 encoder_type="affine", scaling=10.0, l2_lambda=0.0, dropout=0.5):
        super(ExtendedAttentionModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.encoder_type = encoder_type
        self.scaling = scaling
        self.l2_lambda = l2_lambda
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Encoder factory - select based on encoder_type
        if encoder_type.lower() == "lstm":
            self.encoder = LSTMEncoder(embed_dim, hidden_dim, dropout)
        elif encoder_type.lower() == "affine":
            self.encoder = AffineEncoder(embed_dim, hidden_dim, dropout)
        elif encoder_type.lower() == "gru":
            self.encoder = GRUEncoder(embed_dim, hidden_dim, dropout)
        elif encoder_type.lower() == "cnn":
            self.encoder = CNNEncoder(embed_dim, hidden_dim, dropout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Attention parameters (same across all encoders)
        self.V = nn.Parameter(torch.randn(hidden_dim))
        self.W = nn.Parameter(torch.randn(hidden_dim, num_classes))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize parameters
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.V, -0.1, 0.1)
        nn.init.uniform_(self.W, -0.1, 0.1)
    
    def forward(self, input_ids, return_attention=False):
        """
        Args:
            input_ids: (batch_size, seq_len)
            return_attention: bool, return attention weights if True
        Returns:
            logits: (batch_size, num_classes)
            attention: (batch_size, seq_len) [optional]
            polarity: (batch_size, seq_len) [optional]
        """
        # Embedding
        embeddings = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Encode with selected encoder
        encoded = self.encoder(embeddings)  # (batch, seq_len, hidden_dim)
        
        # Polarity scores: s_j = h_j^T W
        polarity = torch.matmul(encoded, self.W)  # (batch, seq_len, num_classes)
        
        # Attention scores: a_j = V^T h_j / sqrt(scaling)
        attn_scores = torch.matmul(encoded, self.V)  # (batch, seq_len)
        attn_scores = attn_scores / (self.scaling ** 0.5)  # Scale by sqrt(τ)
        
        # Softmax attention
        attention = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum: h = Σ a_j h_j
        h_weighted = torch.sum(
            attention.unsqueeze(2) * encoded,
            dim=1
        )  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(h_weighted)  # (batch, num_classes)
        
        if return_attention:
            return logits, attention, polarity
        return logits
    
    def l2_reg(self):
        """L2 regularization on attention parameters V and W."""
        if self.l2_lambda == 0:
            return 0.0
        return self.l2_lambda * (torch.norm(self.V) ** 2 + torch.norm(self.W) ** 2)
    
    def get_encoder_name(self):
        return self.encoder.get_name()
    
    
