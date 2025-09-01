import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import ConvNeXtBlock

def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
    
    m = torch.min(ratio * advantage, clipped)
    return -m.mean()

# class ActorCritic(nn.Module):
#     def __init__(self, in_channels, num_outputs, image_size, num_repeat=2, base_channels=32):
#         super(ActorCritic, self).__init__()
#         self.blocks = nn.ModuleList()
#         curr_ch = base_channels
#         self.blocks.append(nn.Conv2d(in_channels, curr_ch, 7, stride=4, padding=3)) # size / 4
#         image_size = image_size // 4
#         # blocks
#         for _ in range(3): # 16-64, 8-128, 4-256
#             self.blocks.append(ConvNeXtBlock(curr_ch, curr_ch*2, stride=2))
#             curr_ch *= 2
#             image_size = image_size // 2
#             for _ in range(num_repeat-1):
#                 self.blocks.append(ConvNeXtBlock(curr_ch, curr_ch))
#         # 
#         self.pooling = nn.AvgPool2d(kernel_size=image_size, stride=1)
#         # 
#         self.head = nn.Sequential(
#             nn.Linear(curr_ch, 512),
#             nn.GELU(),
#             nn.Linear(512, num_outputs)
#         )

#     def forward(self, x):
#         for layer in self.blocks:
#             x = layer(x)
#         x = self.pooling(x).view(x.size(0), -1)
#         return self.head(x)


class ActorCritic(nn.Module):
    def __init__(self, in_channels, num_outputs, image_size, num_repeat=2, base_channels=32):
        super(ActorCritic, self).__init__()
        self.blocks = nn.ModuleList()
        curr_ch = base_channels
        self.blocks.append(nn.Conv2d(in_channels, curr_ch, 7, stride=4, padding=3)) # size / 4
        image_size = image_size // 4
        # blocks
        for _ in range(3): # 16-64, 8-128, 4-256
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(curr_ch, curr_ch * 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                )
            )
            curr_ch *= 2
            image_size = image_size // 2
            for _ in range(num_repeat-1):
                self.blocks.append(ConvNeXtBlock(curr_ch, curr_ch))
        # 
        self.pooling = nn.AvgPool2d(kernel_size=image_size, stride=1)
        # 
        self.head = nn.Sequential(
            nn.Linear(curr_ch, 512),
            nn.GELU(),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.pooling(x).view(x.size(0), -1)
        return self.head(x)