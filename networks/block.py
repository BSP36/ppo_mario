"""
ConvNeXt Module

References:
    - Original repository: https://github.com/facebookresearch/ConvNeXt
    - Paper: https://arxiv.org/abs/2201.03545

This module implements building blocks for the ConvNeXt architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block consisting of depthwise convolution, layer normalization,
    pointwise linear layers, activation, optional layer scaling, and stochastic depth.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        stride (int, optional): Stride for depthwise convolution. Defaults to 1.
        expansion (int, optional): Expansion ratio for hidden dimension in MLP. Defaults to 4.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameter gamma. Defaults to 1e-6.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int = 1,
        expansion: int = 4,
        drop_path: float = 0.,
        layer_scale_init_value: float = 1e-6
    ):
        
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=in_dim
        ) # depthwise conv
        self.norm = LayerNorm(out_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(out_dim, expansion * out_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * out_dim, out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.donw_size = stride != 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ConvNeXtBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        x_skip = x
        x = self.dwconv(x)
        if self.donw_size:
            x_skip = x  # Update skip connection if downsampling
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x_skip + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    """
    Custom LayerNorm module supporting both 'channels_last' (default) and 'channels_first' data formats.

    Args:
        normalized_shape (int): Number of features/channels to normalize.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-6.
        data_format (str, optional): Either 'channels_last' (default, e.g., (N, H, W, C)) or 'channels_first' (e.g., (N, C, H, W)).
    """
    def __init__(self, normalized_shape: int, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError("data_format must be 'channels_last' or 'channels_first'")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # Standard layer normalization over the last dimension (channels)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Manual normalization over the channel dimension for (N, C, H, W) tensors
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
