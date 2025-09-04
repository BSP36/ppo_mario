import torch
from torch.optim.lr_scheduler import LambdaLR

def make_linear_decay_scheduler(
    optimizer: torch.optim.Optimizer, 
    total_updates: int, 
    warmup_ratio: float = 0.0, 
    min_lr_ratio: float = 0.0
):
    """
    Creates a linear learning rate scheduler with optional warmup and minimum learning rate.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
        total_updates (int): Total number of training updates (steps).
        warmup_ratio (float, optional): Fraction of total_updates used for linear warmup. Defaults to 0.0.
        min_lr_ratio (float, optional): Minimum learning rate as a fraction of the initial LR. Defaults to 0.0.

    Returns:
        LambdaLR: PyTorch learning rate scheduler.
    """
    warmup_steps = int(total_updates * warmup_ratio)

    def lr_lambda(current_update):
        # Linear warmup
        if warmup_steps > 0 and current_update < warmup_steps:
            return (current_update + 1) / max(1, warmup_steps)
        # Linear decay
        progress = (current_update - warmup_steps) / max(1, (total_updates - warmup_steps))
        factor = 1.0 - progress  # Decays from 1.0 to 0.0
        # Ensure LR does not go below min_lr_ratio
        return max(min_lr_ratio, factor * (1.0 - min_lr_ratio) + min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)