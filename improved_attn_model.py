# improved_attn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImprovedAttentionModel(nn.Module):
    """
    Drop-in replacement for the author's Attention model:
    - same embeddings, attention, and output shape
    - supports encoder_type = 'lstm' (original) or 'affine'
    - adds L2 regularization on V,W
    - supports both binary (num_classes=1) and multi-class (num_classes>1)
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=100,
                 num_classes=1, encoder_type='lstm',
                 scaling_factor=10.0, l2_lambda=0.0, padding_idx=0):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.encoder_type = encoder_type
        self.scaling_factor = scaling_factor
        self.l2_lambda = l2_lambda

        # same as author's embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # original encoder
        if encoder_type == "lstm":
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                                batch_first=True, bidirectional=False)
            self.proj = nn.Linear(hidden_dim, embed_dim)

        # affine encoder: h_j = e_j
        self.use_lstm = (encoder_type == "lstm")

        # attention parameters (same names as author)
        self.V = nn.Parameter(torch.empty(embed_dim))
        self.W = nn.Parameter(torch.empty(embed_dim))
        nn.init.uniform_(self.V, -0.1, 0.1)
        nn.init.uniform_(self.W, -0.1, 0.1)

        # classifier head for multi-class
        if num_classes > 1:
            self.classifier = nn.Linear(embed_dim, num_classes)
            nn.init.uniform_(self.classifier.weight, -0.1, 0.1)
            nn.init.uniform_(self.classifier.bias, -0.1, 0.1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (B, T, d)
        emb = self.dropout(emb)

        if self.use_lstm:
            h_seq, _ = self.lstm(emb)  # (B, T, hidden_dim)
            h = self.proj(h_seq)  # (B, T, d) to match V,W dim
        else:
            h = emb  # affine encoder

        # attention logits a_j = h_j^T V / sqrt(scaling_factor)
        logits = torch.matmul(h, self.V) / math.sqrt(self.scaling_factor)
        attn = F.softmax(logits, dim=1)  # (B, T)

        # context = Σ a_j h_j
        ctx = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, d)

        # polarity scores per token: s_j = h_j^T W
        polarity = torch.matmul(h, self.W)  # (B, T)

        # classification: binary or multi-class
        if self.num_classes == 1:
            # binary: sigmoid output
            logit = torch.matmul(ctx, self.W)  # (B,)
            prob = torch.sigmoid(logit)  # (B,)
        else:
            # multi-class: logits from classifier head
            prob = self.classifier(ctx)  # (B, num_classes)

        return prob, attn, polarity

    def l2_reg(self):
        return self.l2_lambda * (torch.norm(self.V) ** 2 + torch.norm(self.W) ** 2)
