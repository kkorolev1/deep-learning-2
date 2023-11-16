import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        seq_pos = torch.arange(0, max_len, dtype=torch.float)
        embed_pos = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) / d_model * math.log(10000))
        arg = torch.outer(seq_pos, embed_pos)
        
        pe = torch.zeros((1, max_len, d_model), dtype=torch.float) # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition
        pe[..., 0::2] = torch.sin(arg)
        pe[..., 1::2] = torch.cos(arg)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class HuYaLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=5000):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, padding_mask, **args):
        mask = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1], device=input_ids.device)
        src = self.pos_encoding(self.embedding(input_ids))
        seq = self.model(
            src=src,
            mask=mask,
            src_key_padding_mask=padding_mask,
            is_causal=True
        )
        return {
            "logits": self.head(seq)
        }
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)