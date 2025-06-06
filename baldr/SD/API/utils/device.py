import torch

def get_device():
    if torch.cuda.is_available():
        return  "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"
