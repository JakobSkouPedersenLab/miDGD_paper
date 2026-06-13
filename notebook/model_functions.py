from pathlib import Path

import torch


def find_repo_root(start=None):
    """Find the repository root from either the repo or notebook directory."""
    path = Path.cwd() if start is None else Path(start)
    path = path.resolve()
    for candidate in (path, *path.parents):
        if (candidate / "models").is_dir() and (candidate / "notebook").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find repository root containing both models/ and notebook/."
    )


def load_trained_model(model_filename, device="cpu"):
    """Load one of the frozen manuscript checkpoints from models/."""
    model_path = find_repo_root() / "models" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_exclude_rep(model):
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "rep" not in name
    )
