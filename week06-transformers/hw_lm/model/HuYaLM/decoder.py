import torch
import torch.nn as nn

from hw_lm.model.HuYaLM.attention import MultiheadAttention
from hw_lm.model.HuYaLM.ffn import FFN


class DecoderLayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_dim,
                 activation=nn.ReLU,
                 dropout=0.0,
                 attn_use_prelayer_norm=True):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.mha = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_prelayer_norm=attn_use_prelayer_norm
        )
        self.ffn = FFN(
            embed_dim=embed_dim,
            feedforward_dim=feedforward_dim,
            activation=activation,
            dropout=dropout
        )

    def forward(self, x, padding_mask, causal_mask):
        return self.ffn(self.mha(x, padding_mask=padding_mask, causal_mask=causal_mask)) 


