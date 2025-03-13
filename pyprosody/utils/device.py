import torch

def get_optimal_device() -> torch.device:
    """
    Determines the optimal available device for PyTorch operations.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")