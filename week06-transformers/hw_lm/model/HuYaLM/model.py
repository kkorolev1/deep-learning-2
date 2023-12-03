import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from hw_lm.model.HuYaLM.utils import PositionalEncoding, TokenEmbedding
from hw_lm.model.HuYaLM.decoder import DecoderLayer

class HuYaLM(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 feedforward_dim,
                 num_layers,
                 vocab_size,
                 dropout=0.1,
                 attn_use_prelayer_norm=True,
                 activation=nn.GELU,
                 max_len=320):
        super().__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                activation=activation,
                dropout=dropout,
                attn_use_prelayer_norm=attn_use_prelayer_norm
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, padding_mask, **args):
        causal_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1], device=input_ids.device)
        x = self.pos_encoding(self.embedding(input_ids))
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, padding_mask=padding_mask, causal_mask=causal_mask)
        logits = self.head(x)
        return {
            "logits": logits
        }
    
    def _get_next_token_argmax(self, logits):
        return logits.argmax(dim=-1).unsqueeze(0)
    
    def _get_next_token_nucleus(self, probas, p=0.9):
        sorted_probas, indices = torch.sort(probas, dim=-1, descending=True)
        cum_sum_probas = torch.cumsum(sorted_probas, dim=-1)
        nucleus = cum_sum_probas < p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        sorted_probas[~nucleus] = 0.0
        sorted_probas /= sorted_probas.sum()
        dist = torch.distributions.Categorical(probs=sorted_probas.squeeze(0))
        next_token_id = dist.sample().item()
        return indices[:, next_token_id].unsqueeze(0)
        
    @torch.no_grad()
    def inference(self, text, tokenizer, device, max_length, temperature=1, mode="argmax", p=0.9):
        input_ids = torch.tensor(
            [tokenizer.processor.bos_id()] + tokenizer.encode(text),
        device=device).unsqueeze(0)
        
        for i in range(max_length):
            padding_mask = torch.zeros_like(input_ids, dtype=bool, device=device)
            logits = self.forward(input_ids, padding_mask)["logits"][:, -1, :]
            if mode == "argmax":
                next_token_id = self._get_next_token_argmax(logits)
            elif mode == "nucleus":
                probas = F.softmax(logits, dim=-1) ** temperature
                probas /= probas.sum()
                next_token_id = self._get_next_token_nucleus(probas, p=p)
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            if next_token_id.item() == tokenizer.processor.eos_id():
                    break
        return tokenizer.decode(input_ids.tolist())
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
