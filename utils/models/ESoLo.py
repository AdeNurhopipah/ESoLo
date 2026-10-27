"""
ESoLo (EEG Source Localisation) neural network.

The architecture consists of:
    1. ChannelAttention:
       Learns channel-wise attention weights from average and maximum temporal
       descriptors to emphasize informative EEG sensors.

    2. Temporal feature extractor:
       A one-dimensional convolution followed by residual convolutional blocks
       extracts hierarchical temporal features and progressively reduces the
       temporal resolution.

    3. Bidirectional LSTM:
       Models forward and backward temporal dependencies in the extracted EEG
       representation.

    4. Temporal attention pooling:
       Learns the contribution of each time point and aggregates the LSTM
       sequence into a single feature representation.

    5. Source prediction head:
       Maps the attended temporal representation to logits across all
       candidate source locations.

Input:
    EEG epochs with shape (batch, n_channels, n_times).

Output:
    Source logits with shape (batch, n_sources).

Optional outputs:
    The latent embedding, channel-attention weights, and temporal-attention
    weights can be returned for model interpretation.
"""

import torch
import torch.nn as nn


# ============================================================
# RESIDUAL BLOCK
# ============================================================
class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        dropout=0.2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=5,
                stride=stride,
                padding=2,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(out_ch),
        )

        if in_ch == out_ch and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm1d(out_ch),
            )

        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        output = self.block(x)

        return self.activation(
            output + residual
        )


# ============================================================
# CHANNEL ATTENTION
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(
        self,
        n_ch,
        reduction=4,
    ):
        super().__init__()

        hidden_size = max(
            n_ch // reduction,
            8,
        )

        self.attention = nn.Sequential(
            nn.Linear(n_ch, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_ch),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channel, time)

        avg_descriptor = torch.mean(
            x,
            dim=2,
        )

        max_descriptor = torch.amax(
            torch.abs(x),
            dim=2,
        )

        attention_logits = (
            self.attention(avg_descriptor)
            + self.attention(max_descriptor)
        )

        attention_weights = (
            2.0
            * self.sigmoid(attention_logits)
        )

        attended_x = x * (
            1.0
            + attention_weights.unsqueeze(-1)
        )

        return (
            attended_x,
            attention_weights,
        )


# ============================================================
# TEMPORAL ATTENTION
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(
        self,
        input_size,
        attention_size=64,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                input_size,
                attention_size,
            ),
            nn.Tanh(),
            nn.Linear(
                attention_size,
                1,
            ),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch, time, feature)

        attention_logits = self.attention(x)

        attention_weights = self.softmax(
            attention_logits
        )

        pooled = torch.sum(
            x * attention_weights,
            dim=1,
        )

        return (
            pooled,
            attention_weights.squeeze(-1),
        )


# ============================================================
# ESOLO
# ============================================================
class ESoLo(nn.Module):
    def __init__(
        self,
        n_channels,
        n_sources,
        hidden_size=64,
        attention_size=64,
        channel_reduction=4,
        dropout=0.2,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.channel_attention = ChannelAttention(
            n_ch=n_channels,
            reduction=channel_reduction,
        )

        self.features = nn.Sequential(
            nn.Conv1d(
                n_channels,
                128,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            ResidualBlock1D(
                128,
                128,
                dropout=dropout,
            ),

            ResidualBlock1D(
                128,
                64,
                stride=2,
                dropout=dropout,
            ),

            ResidualBlock1D(
                64,
                64,
                dropout=dropout,
            ),

            ResidualBlock1D(
                64,
                32,
                stride=2,
                dropout=dropout,
            ),
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.temporal_attention = AttentionPooling(
            input_size=hidden_size * 2,
            attention_size=attention_size,
        )

        self.embedding = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.source_head = nn.Linear(
            64,
            n_sources,
        )

    

    def forward(
        self,
        x,
        return_attention=False,
    ):
        x, channel_attention = (
            self.channel_attention(x)
        )

        x = self.features(x)

        x = x.permute(
            0,
            2,
            1,
        )

        x, _ = self.lstm(x)

        pooled, temporal_attention = (
            self.temporal_attention(x)
        )

        embedding = self.embedding(
            pooled
        )

        source_logits = self.source_head(
            embedding
        )

        if return_attention:
            return {
                "output": source_logits,
                "source_logits": source_logits,
                "embedding": embedding,
                "channel_attention": channel_attention,
                "temporal_attention": temporal_attention,
            }

        return source_logits

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
