import torch


def get_device(prefer: str | None = None) -> torch.device:
    """
    Pick device: prefer "cuda", "mps", "cpu", or None for auto (CUDA > MPS > CPU).
    If prefer is set and available, use it; otherwise fall back to auto.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
