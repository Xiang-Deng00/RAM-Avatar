import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import get_embedder, timestep_embedding


# norm_layer = lambda ch: nn.GroupNorm(4, ch)
# norm_layer = lambda ch: nn.BatchNorm2d(ch)
norm_layer = lambda ch: nn.InstanceNorm2d(ch)

acti_layer = lambda *args, **kwargs: nn.ReLU(*args, **kwargs)
# acti_layer = lambda *args, **kwargs: nn.SiLU(*args, **kwargs)


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            acti_layer(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            acti_layer(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            acti_layer(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            acti_layer(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            acti_layer(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            acti_layer(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SingleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_ratio=2):
        super(InConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            norm_layer(out_channels),
            acti_layer(inplace=True)
        ]
        if downsample_ratio > 1:
            for _ in range(int(math.log2(downsample_ratio))):
                layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
                )
                layers.append(norm_layer(out_channels))
                layers.append(acti_layer(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = acti_layer(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UNetEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_hidden):
        super(UNetEncoder, self).__init__()
        self.inc = SingleConv(n_channels_in, n_channels_hidden[0])
        down_modules = list()
        for ic, oc in zip(n_channels_hidden[:-1], n_channels_hidden[1:]):
            down_modules.append(Down(ic, oc))
        self.down_modules = nn.ModuleList(down_modules)

    def forward(self, x):
        out = list()
        x = self.inc(x)
        out.append(x)

        for mod in self.down_modules:
            x = mod(x)
            out.append(x)

        return out


class UNetDecoder(nn.Module):
    def __init__(self, n_channels_hidden, bilinear=True):
        super(UNetDecoder, self).__init__()
        up_modules = list()
        n_channels_hidden_ = n_channels_hidden[::-1]
        for ic, oc in zip(n_channels_hidden_[:-1], n_channels_hidden_[1:]):
            up_modules.append(Up(ic+oc, oc, bilinear))
        self.up_modules = nn.ModuleList(up_modules)

    def forward(self, xs):
        out = list()
        i = -2
        x = xs[-1]
        for mod in self.up_modules:
            x = mod(x, xs[i])
            out.append(x)
            i -= 1

        return out


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks=0,
            attention_resolutions=None,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.n_hiddens = tuple(model_channels * m for m in channel_mult)
        self.encoder = UNetEncoder(self.n_hiddens[0]//2, self.n_hiddens)
        self.decoder = UNetDecoder(self.n_hiddens, bilinear=True)
        self.inc = InConv(self.in_channels, self.n_hiddens[0]//2, downsample_ratio=1)
        self.outc = OutConv(self.n_hiddens[0], self.out_channels)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.encoder.apply(convert_module_to_f16)
        self.decoder.apply(convert_module_to_f16)
        self.outc.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.encoder.apply(convert_module_to_f32)
        self.decoder.apply(convert_module_to_f32)
        self.outc.apply(convert_module_to_f32)

    def forward(self, x, y=None):
        x_in = self.inc(x)
        hiddens = self.encoder(x_in)
        ys = self.decoder(hiddens)
        out = self.outc(ys[-1])
        return out


