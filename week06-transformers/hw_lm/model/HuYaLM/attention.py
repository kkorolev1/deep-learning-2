import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attention_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, padding_mask, causal_mask):
        """
        Args:
            query: torch.Tensor (..., L, D)
            key: torch.Tensor (..., L, D)
            value: torch.Tensor (..., L, D)
        Returns:
            res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
            attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

        L is the length of sequence, D is the embedding dimension
        """
        original_shape = query.shape
        if len(query.shape) == 2:
            query = query.unsqueeze(0).unsqueeze(0)
            key = key.unsqueeze(0).unsqueeze(0)
            value = value.unsqueeze(0).unsqueeze(0)
        dot_products = torch.einsum(
            "bhld,bhdm->bhlm", query, key.transpose(-2, -1)) / self.temperature
        # Apply causal mask
        dot_products = dot_products + causal_mask
        # Apply padding mask
        # padding_mask (B, L) -> (B, 1, 1, L) -> repeat over rows and over heads
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand(
            -1, query.shape[1], query.shape[2], -1
        )
        dot_products = dot_products.masked_fill_(padding_mask, -torch.inf)
        attention = self.dropout(torch.softmax(dot_products, dim=-1))
        res = torch.einsum("bhlm,bhmd->bhld", attention, value)
        if len(original_shape) == 2:
            res = res.squeeze(0).squeeze(0)
            attention = attention.squeeze(0).squeeze(0)
        return res
    
class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, use_prelayer_norm=True):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(
            temperature=self.head_dim ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.use_prelayer_norm = use_prelayer_norm
        if use_prelayer_norm:
            self.prelayer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, padding_mask, causal_mask):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        batch_size, length = x.shape[0], x.shape[1]
        if self.use_prelayer_norm:
            x = self.prelayer_norm(x)
        query = self.q_proj(x).reshape(batch_size, length,
                                       self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.k_proj(x).reshape(batch_size, length,
                                     self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.v_proj(x).reshape(batch_size, length,
                                       self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        outputs = self.attention(
            query, key, value,
            padding_mask=padding_mask,
            causal_mask=causal_mask
        )
        outputs = self.o_proj(outputs.permute(0, 2, 1, 3).reshape(
            batch_size, length, self.embed_dim))
        outputs = self.layer_norm(self.dropout(outputs) + x)
        return outputs