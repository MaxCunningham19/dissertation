import numpy as np
import torch


def extract_kwargs(kwargs: list[str]):
    """Extracts the kwargs from the list of strings .e.g key=value"""
    kwargs_dict = {}
    if kwargs:
        for pair in kwargs:
            key, value = pair.split("=")

            if "," in value:
                values = value.split(",")
                true_vals = [None] * len(values)
                for i, v in enumerate(values):
                    true_vals[i] = convert_to_num_if_possible(v)

                value = true_vals
            else:
                value = convert_to_num_if_possible(value)

            if value == "True":
                value = True
            elif value == "False":
                value = False

            kwargs_dict[key] = value
    return kwargs_dict


def kwargs_to_string(kwargs: list[str] | None):
    """Converts a list of kwargs to a string that can be used as a filename"""
    if kwargs is None or not isinstance(kwargs, list) or len(kwargs) == 0 or not all(isinstance(item, str) for item in kwargs):
        return ""

    string = ""
    for i, pair in enumerate(kwargs):
        pair = pair.replace(" ", "").replace("/", "_").replace(".", "_").replace("=", "_")
        string += f"{pair}"
        if i < len(kwargs) - 1:
            string += "__"
    return string


def convert_to_num_if_possible(value: str):
    """Converts a string to a number if possible"""
    try:
        if "." in value:
            value = float(value)
        else:
            value = int(value)
    except:
        pass
    return value


def softmax(x) -> np.ndarray:
    """Applies the softmax function to a numpy array"""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
