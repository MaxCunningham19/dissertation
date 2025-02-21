import numpy as np
import torch


def extract_kwargs(kwargs: list[str]):
    kwargs_dict = {}
    if kwargs:
        for pair in kwargs:
            key, value = pair.split("=")

            if "," in value:
                values = value.split(",")
                true_vals = [None] * len(values)
                for i, v in enumerate(value):
                    true_vals[i] = convert_to_num_if_possible(v)

                value = true_vals
            else:
                value = convert_to_num_if_possible(value)

            kwargs_dict[key] = value
    return kwargs_dict


def convert_to_num_if_possible(value: str):
    try:
        if "." in value:
            value = float(value)
        else:
            value = int(value)
    except:
        pass
    return value


def softmax(x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
