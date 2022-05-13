# G (Generator)
# CNN (Convolutional Neural Network)
# Resize transposed convolution

from torch import nn

self = self
zr = self.config["noise_resolution"]
zc = self.config["noise_channel_count"]
ir = self.config["image_resolution"]
ic = self.config["image_channel_count"]
fm = self.config["feature_map_count"]

# NOTE:
# nn.ConvTranspose2d positional params: in_channels, out_channels, kernel_size, stride, padding
# nn.Upsample positional params: size
# nn.ReLU positional params: inplace
# nn.BatchNorm2d positional params: num_features

_ConvTranspose2d = nn.ConvTranspose2d
_Upsample = nn.Upsample
_ReLU = nn.ReLU
_BatchNorm2d = nn.BatchNorm2d
_Tanh = nn.Tanh

self.model = nn.Sequential(
    # Layer group 1. input group
    _Upsample(4, mode="bicubic", align_corners=False),
    _ConvTranspose2d(zc, int(7 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(7 * fm)),
    _ReLU(True),
    # 2.
    _Upsample(int(ir // 8), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(7 * fm), int(5 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(5 * fm)),
    _ReLU(True),
    # 3.
    _Upsample(int(ir // 4), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(5 * fm), int(3 * fm), 3, 1, 1, bias=False),
    _BatchNorm2d(int(3 * fm)),
    _ReLU(True),
    # 4.
    _Upsample(int(ir // 2), mode="bilinear", align_corners=False),
    _ConvTranspose2d(int(3 * fm), fm, 3, 1, 1, bias=False),
    _BatchNorm2d(fm),
    _ReLU(True),
    # 5. output group
    _Upsample(ir, mode="bicubic", align_corners=False),
    _ConvTranspose2d(fm, ic, 5, 1, 2, bias=False),
    _Tanh()
)
