"""Network definitions for the Lineart (Informative Drawings) and HED models.

Adapted from huggingface/controlnet_aux (Apache-2.0), which in turn adapted
lllyasviel/ControlNet annotators. See THIRD_PARTY_NOTICES.md. Only the torch modules live here; loading,
pre- and post-processing are in linework.models.registry / linework.detect.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------- Lineart ----
norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LineartGenerator(nn.Module):
    """Generator from 'Learning to generate line drawings that convey geometry
    and semantics' (Chan, Durand, Isola 2022). Output: 1 = white paper, 0 = ink."""

    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), norm_layer(64), nn.ReLU(inplace=True)
        )
        model1 = []
        in_f, out_f = 64, 128
        for _ in range(2):
            model1 += [nn.Conv2d(in_f, out_f, 3, stride=2, padding=1), norm_layer(out_f), nn.ReLU(inplace=True)]
            in_f, out_f = out_f, out_f * 2
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*[ResidualBlock(in_f) for _ in range(n_residual_blocks)])
        model3 = []
        out_f = in_f // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f, out_f = out_f, out_f // 2
        self.model3 = nn.Sequential(*model3)
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


# -------------------------------------------------------------------- HED ----
class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(input_channel, output_channel, 3, 1, 1))
        for _ in range(1, layer_number):
            self.convs.append(nn.Conv2d(output_channel, output_channel, 3, 1, 1))
        self.projection = nn.Conv2d(output_channel, 1, 1, 1, 0)

    def forward(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=2, stride=2)
        for conv in self.convs:
            h = torch.nn.functional.relu(conv(h))
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    """lllyasviel's Apache-2.0 HED re-implementation (RGB input, 0..255)."""

    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

    def forward(self, x):
        h = x - self.norm
        h, p1 = self.block1(h)
        h, p2 = self.block2(h, down_sampling=True)
        h, p3 = self.block3(h, down_sampling=True)
        h, p4 = self.block4(h, down_sampling=True)
        h, p5 = self.block5(h, down_sampling=True)
        return p1, p2, p3, p4, p5
