from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_


class Decoder(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            hidden_size=256,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(Decoder, self).__init__()
        self.norm = norm_layer(embed_dim)

        self.conv_0 = nn.Conv2d(embed_dim, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=1)

        self.syncbn_fc_0 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_1 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_2 = nn.BatchNorm2d(hidden_size)
        self.syncbn_fc_3 = nn.BatchNorm2d(hidden_size)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, edge_feat):
        x = rearrange(x, "b (h w) c -> b c h w", h=22)

        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        x = self.conv_1(x)
        x = self.syncbn_fc_1(x)
        x = F.relu(x, inplace=False)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        x = self.conv_2(x)
        x = self.syncbn_fc_2(x)
        x = F.relu(x, inplace=False)
        x = torch.cat([x, edge_feat], dim=1)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        x = self.conv_3(x)
        x = self.syncbn_fc_3(x)
        x = F.relu(x, inplace=False)
        x = self.conv_4(x)
        x = F.interpolate(x, size=x.shape[-1] * 2, mode="bilinear", align_corners=False)

        return x
