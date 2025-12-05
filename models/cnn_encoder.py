"""CNN Encoder - Convolutional encoder (fastest for sequences)."""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """CNN encoder using 1D convolutions - fastest encoding option."""
    
    def __init__(self, embed_dim, hidden_dim=100, dropout=0.5, num_filters=50):
        super(CNNEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.dropout = nn.Dropout(dropout)
        
        # Multiple kernel sizes to capture n-grams
        kernel_sizes = [3, 4, 5]
        
        # 1D convolutions (much faster than recurrent)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k//2  # same padding
            )
            for k in kernel_sizes
        ])
        
        # Project concatenated conv outputs to hidden_dim
        total_conv_out = len(kernel_sizes) * num_filters
        self.projection = nn.Linear(total_conv_out, hidden_dim)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch_size, seq_len, embed_dim)
        Returns:
            encoded: (batch_size, seq_len, hidden_dim)
        """
        # Apply dropout
        x = self.dropout(embeddings)  # (batch, seq_len, embed_dim)
        
        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions with ReLU
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, num_filters, seq_len)
            conv_outputs.append(conv_out)
        
        # Concatenate all conv outputs: (batch, total_filters, seq_len)
        x = torch.cat(conv_outputs, dim=1)
        
        # Transpose back: (batch, seq_len, total_filters)
        x = x.transpose(1, 2)
        
        # Project to hidden_dim
        encoded = self.projection(x)  # (batch, seq_len, hidden_dim)
        
        return encoded
    
    def get_name(self):
        return "CNN"
