import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, activation=nn.ReLU, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.mlp(x) + x)
