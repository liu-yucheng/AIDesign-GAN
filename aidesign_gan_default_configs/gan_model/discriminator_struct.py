# D (Discriminator)
# CNN (Convolutional Neural Network)
# Resize convolution

import torch
from torch import nn

self = self
ir = self.config["image_resolution"]
ic = self.config["image_channel_count"]
lr = self.config["label_resolution"]
lc = self.config["label_channel_count"]
fm = self.config["feature_map_count"]

# NOTE:
# nn.Conv2d positional params: in_channels, out_channels, kernel_size, stride, padding
# nn.Upsample positional params: size
# nn.LeakyReLU positional params: negative_slope, inplace
# nn.BatchNorm2d positional params: num_features

_Conv2d = nn.Conv2d
_Upsample = nn.Upsample
_LeakyReLU = nn.LeakyReLU
_BatchNorm2d = nn.BatchNorm2d
_Sigmoid = nn.Sigmoid
_compile = torch.compile

self.model = nn.Sequential(
    # Layer group 1. input group
    _Conv2d(ic, fm, 5, 1, 2, bias=False),
    _Upsample(int(ir // 2), mode="bicubic", align_corners=False),
    _LeakyReLU(0.2, True),
    # 2.
    _Conv2d(fm, int(3 * fm), 3, 1, 1, bias=False),
    _Upsample(int(ir // 4), mode="bilinear", align_corners=False),
    _BatchNorm2d(int(3 * fm)),
    _LeakyReLU(0.2, True),
    # 3.
    _Conv2d(int(3 * fm), int(5 * fm), 3, 1, 1, bias=False),
    _Upsample(int(ir // 8), mode="bilinear", align_corners=False),
    _BatchNorm2d(int(5 * fm)),
    _LeakyReLU(0.2, True),
    # 4.
    _Conv2d(int(5 * fm), int(7 * fm), 3, 1, 1, bias=False),
    _Upsample(4, mode="bilinear", align_corners=False),
    _BatchNorm2d(int(7 * fm)),
    _LeakyReLU(0.2, True),
    # 5. output group
    _Conv2d(int(7 * fm), lc, 3, 1, 1, bias=False),
    _Upsample(lr, mode="bicubic", align_corners=False),
    _Sigmoid()
)

# Currently, torch.compile does not support Windows because OpenAI's triton does not support Windows.
# self.model = _compile(self.model)
