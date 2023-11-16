import torch.nn as nn

class HuYaLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, input_ids, padding_mask, **args):
        mask = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1], device=input_ids.device)
        input_ids = self.embedding(input_ids)
        return self.model(
            src=input_ids,
            mask=mask,
            src_key_padding_mask=padding_mask,
            is_causal=True
        )