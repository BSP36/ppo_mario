import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, expansion=4, drop_path=0., layer_scale_init_value=1e-6):
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

    def forward(self, x):
        x_skip = x
        x = self.dwconv(x)
        if self.donw_size:
            x_skip = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = x_skip + self.drop_path(x)
        return x



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
