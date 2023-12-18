import numpy as np
import torch
import torch.nn.functional as F


def collate(batch):
    return {
        "image": torch.cat([item[0].unsqueeze(0) for item in batch])
    }