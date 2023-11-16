import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

    def forward(self, input_ids, logits, **kwargs):
        return F.cross_entropy(logits[:, :-1].transpose(1, 2), input_ids[:, 1:], ignore_index=self.pad_id)
