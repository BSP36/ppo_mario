import torch
import torch.nn as nn
from .block import ConvNeXtBlock

def clip_loss(
    log_pi_old: torch.Tensor,
    log_pi: torch.Tensor,
    advantage: torch.Tensor,
    eps: float
) -> torch.Tensor:
    """
    Computes the PPO clipped surrogate objective (CLIP loss).

    Args:
        log_pi_old (torch.Tensor): Log probabilities of actions under the old policy.
        log_pi (torch.Tensor): Log probabilities of actions under the current policy.
        advantage (torch.Tensor): Estimated advantage for each action.
        eps (float): Clipping parameter (epsilon) for PPO.

    Returns:
        torch.Tensor: Scalar loss value (to minimize).
    """
    # Calculate the probability ratio (current policy / old policy)
    ratio = (log_pi - log_pi_old).exp()
    # Compute the clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
    surrogate1 = ratio * advantage
    surrogate2 = clipped_ratio * advantage
    # Take the minimum of the unclipped and clipped objectives
    loss = -torch.min(surrogate1, surrogate2).mean()
    return loss


class ActorCritic(nn.Module):
    """
    Actor-Critic model using ConvNeXt blocks for feature extraction.

    Args:
        in_channels (int): Number of input channels (e.g., 4 for stacked frames).
        num_outputs (int): Number of output actions (policy logits or value).
        image_size (int): Height/width of the input image (assumed square).
        num_repeat (int, optional): Number of ConvNeXt blocks per stage. Defaults to 2.
        base_channels (int, optional): Number of channels for the first conv layer. Defaults to 32.
    """
    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        image_size: int,
        num_repeat: int = 2,
        base_channels: int = 32,
    ):
        super(ActorCritic, self).__init__()
        self.blocks = nn.ModuleList()
        curr_ch = base_channels

        # Initial convolution: downsample input spatially by 4
        self.blocks.append(
            nn.Conv2d(in_channels, curr_ch, kernel_size=7, stride=4, padding=3)
        )
        image_size = image_size // 4

        # Stacked stages: each halves spatial size and doubles channels
        for _ in range(3):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(curr_ch, curr_ch * 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                )
            )
            curr_ch *= 2
            image_size = image_size // 2
            for _ in range(num_repeat - 1):
                self.blocks.append(ConvNeXtBlock(curr_ch, curr_ch))

        # Global average pooling to flatten spatial dimensions
        self.pooling = nn.AvgPool2d(kernel_size=image_size, stride=1)

        # Fully connected head for output
        self.head = nn.Sequential(
            nn.Linear(curr_ch, 512),
            nn.GELU(),
            nn.Linear(512, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.blocks:
            x = layer(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.head(x)