import numpy as np
import torch


def softmax(x) -> np.ndarray:
    """Applies the softmax function to a numpy array"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().flatten()

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
