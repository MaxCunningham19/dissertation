import numpy as np
import torch


def softmax(x) -> np.ndarray:
    """Applies the softmax function to a numpy array"""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
