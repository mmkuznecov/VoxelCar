"""Monocular occupancy prediction model.

Architecture overview:

    Image (3, H, W)
       │   2D CNN encoder with stride-2 downsampling
       ▼
    Feature map (C, h', w')
       │   bilinear resize to (D_x, D_y)  ← crude "BEV lift"
       ▼
    BEV feature (C, D_x, D_y)
       │   conv refinement
       │   1×1 conv to D_z channels
       ▼
    (D_z, D_x, D_y)
       │   permute to (D_x, D_y, D_z)
       ▼
    Occupancy logits

The encoder shrinks the image; the bilinear resize is a deliberately simple
"lift" from image plane to ego-BEV grid. A proper inverse-perspective lift
with camera intrinsics would be more accurate, but for a single fixed camera
the network can learn the mapping implicitly. This is a POC-grade choice.

Output convention matches the GT tensor shape (D_x, D_y, D_z) produced by
``src.ego``: D_x = forward, D_y = right, D_z = up.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class OccNet(nn.Module):
    """Image → ego-frame occupancy logits.

    Parameters
    ----------
    d_x, d_y, d_z : int
        Target ego-grid dimensions.
    channels : tuple[int, ...]
        Channel widths for the stride-2 encoder blocks. Default is a modest
        ~2.5 M-parameter network.
    dropout : float
        Applied inside the 3D head.
    """

    def __init__(self, d_x, d_y, d_z, channels=(32, 64, 128, 256, 256), dropout=0.1):
        super().__init__()
        self.d_x = int(d_x)
        self.d_y = int(d_y)
        self.d_z = int(d_z)

        ins = [3] + list(channels[:-1])
        outs = list(channels)
        self.encoder = nn.Sequential(
            *[_conv_block(ic, oc, stride=2) for ic, oc in zip(ins, outs)]
        )

        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(128, self.d_z, kernel_size=1),
        )

    def forward(self, x):
        # x : (B, 3, H, W), float in [0, 1]
        f = self.encoder(x)  # (B, C, h', w')
        f = F.interpolate(
            f, size=(self.d_x, self.d_y), mode="bilinear", align_corners=False
        )  # (B, C, Dx, Dy)
        logits = self.head(f)  # (B, Dz, Dx, Dy)
        return logits.permute(0, 2, 3, 1).contiguous()  # (B, Dx, Dy, Dz)

    @staticmethod
    def num_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
