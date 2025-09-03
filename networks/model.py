import torch
import torch.nn as nn
from .block import ConvNeXtBlock, Downsample, LayerNorm

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
        num_actions (int): Number of output actions (policy logits or value).
        image_size (int): Height/width of the input image (assumed square).
        base_channels (int, optional): Number of channels for the first conv layer. Defaults to 32.
        num_stages (int, optional): Number of stages in the network. Defaults to 3.
        num_repeat (int, optional): Number of ConvNeXt blocks per stage. Defaults to 2.
        
    """
    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        base_channels: int = 32,
        num_stages: int = 3,
        num_repeat: int = 2,
    ):
        super(ActorCritic, self).__init__()
        self.blocks = nn.ModuleList()
        curr_ch = base_channels

        # Initial convolution: downsample input spatially by 4
        self.stem_norm = LayerNorm(in_channels, eps=1e-6)
        self.stem = nn.Conv2d(in_channels, curr_ch, kernel_size=4, stride=4, padding=0, bias=False)

        # Stacked stages: each halves spatial size and doubles channels
        for l in range(num_stages):
            for _ in range(num_repeat):
                self.blocks.append(ConvNeXtBlock(curr_ch, curr_ch))

            if l != num_stages - 1:
                self.blocks.append(Downsample(curr_ch, curr_ch * 2))
                curr_ch *= 2

        # Global average pooling to flatten spatial dimensions
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # Shared pre-head norm
        self.pre_head_norm = nn.LayerNorm(curr_ch, eps=1e-6)
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(curr_ch, 512),
            nn.GELU(),
            nn.Linear(512, num_actions),
        )
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(curr_ch, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )
        # Initialize weights for all layers and custom-initialize actor/critic output layers
        self.apply(self._init_weights)
        with torch.no_grad():
            # Scale down the last actor layer weights for stable policy initialization
            last_actor = self._find_last_linear(self.actor)
            last_actor.weight.mul_(0.01)
            if last_actor.bias is not None:
                last_actor.bias.zero_()
            # Zero-initialize the last critic layer weights and bias for value head
            last_critic = self._find_last_linear(self.critic)
            last_critic.weight.zero_()
            if last_critic.bias is not None:
                last_critic.bias.zero_()

    @staticmethod
    def _init_weights(m):
        """Custom weight initialization for Linear and Conv2d layers."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _find_last_linear(module: nn.Module):
        """Find the last nn.Linear layer in a module."""
        last = None
        for m in module.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def forward(self, x: torch.Tensor):
        """
        Forward pass for ActorCritic network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy logits and value estimate.
        """
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.stem(x)

        for layer in self.blocks:
            x = layer(x)

        x = self.pooling(x).view(x.size(0), -1)
        x = self.pre_head_norm(x)
        return self.actor(x), self.critic(x)