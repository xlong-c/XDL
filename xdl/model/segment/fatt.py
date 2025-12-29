from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


class layer1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)


class GlobalFilter(nn.Module):
    def __init__(self, dim, dim2, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h // 2 + 1, w, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.conv = nn.Conv2d(dim, dim2, 1, bias=False)

    def forward(self, x, spatial_size=None):
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, dim=(1, 2), norm="ortho")
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x1 = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out) + x1
        return out


def autopad(
    k: Union[int, Tuple[int, int]], p: Union[int, Tuple[int, int], None] = None, d: int = 1
) -> Union[int, Tuple[int, int]]:  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            # k 是长度为2的元组 (height, width)
            k = (d * (k[0] - 1) + 1, d * (k[1] - 1) + 1)
    if p is None:
        p = k // 2 if isinstance(k, int) else (k[0] // 2, k[1] // 2)  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[int, Tuple[int, int]] = 1,
        s: Union[int, Tuple[int, int]] = 1,
        p: Union[int, Tuple[int, int], None] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ):
        """Initialize Conv layer with given arguments including activation.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size (int or tuple)
            s: Stride (int or tuple)
            p: Padding (int or tuple)
            g: Groups
            d: Dilation
            act: Activation function
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PConv(nn.Module):
    """Pinwheel-shaped Convolution using the Asymmetric Padding method."""

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class FATT(nn.Module):
    def __init__(self, dim=32, dims=(64, 128, 320, 512)):
        super().__init__()
        channels = dims
        self.size_fea = (88, 44, 22, 11)
        self.pinwheel1 = PConv(channels[0], dim, 4, 1)
        self.pinwheel2 = PConv(channels[1], dim, 4, 1)
        self.pinwheel3 = PConv(channels[2], dim, 4, 1)
        self.pinwheel4 = PConv(channels[3], dim, 4, 1)
        self.gf1 = GlobalFilter(3, dim, self.size_fea[0], self.size_fea[0])
        self.gf2 = GlobalFilter(3, dim, self.size_fea[1], self.size_fea[1])
        self.gf3 = GlobalFilter(3, dim, self.size_fea[2], self.size_fea[2])
        self.avp = nn.AdaptiveAvgPool2d(1)

        self.cbam1 = nn.Sequential(
            CBAM(32, ratio=4, kernel_size=3), CBAM(32, ratio=4, kernel_size=3)
        )
        self.cbam2 = nn.Sequential(
            CBAM(32, ratio=4, kernel_size=3), CBAM(32, ratio=4, kernel_size=3)
        )
        self.cbam3 = nn.Sequential(
            CBAM(32, ratio=4, kernel_size=7), CBAM(32, ratio=4, kernel_size=7)
        )

        self.ca1 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 1, bias=False),
            nn.Sigmoid(),
        )
        self.ca2 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 1, bias=False),
            nn.Sigmoid(),
        )
        self.ca3 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 1, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1, bias=False),
        )

    def forward(self, x):
        x1, x2, x3, x4 = x
        out_size = [x1.shape[2] * 4, x1.shape[3] * 4]
        x1 = F.interpolate(
            self.pinwheel1(x1), size=self.size_fea[0], mode="bilinear", align_corners=True
        )  # 88
        x2 = F.interpolate(
            self.pinwheel2(x2), size=self.size_fea[1], mode="bilinear", align_corners=True
        )  # 44
        x3 = F.interpolate(
            self.pinwheel3(x3), size=self.size_fea[2], mode="bilinear", align_corners=True
        )  # 22
        x4 = F.interpolate(
            self.pinwheel4(x4), size=self.size_fea[3], mode="bilinear", align_corners=True
        )  # 11

        fea = torch.cat(
            [
                torch.mean(x1, dim=1, keepdim=True),
                F.interpolate(
                    torch.mean(x2, dim=1, keepdim=True),
                    x1.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    torch.mean(x3, dim=1, keepdim=True),
                    x1.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            ],
            dim=1,
        )
        can = torch.cat([self.avp(x1), self.avp(x2), self.avp(x3)], dim=1)

        xx = F.interpolate(x4, x3.shape[2:], mode="bilinear", align_corners=False)
        # gf的输入是x1,x2,x3的平均化特征维度是[B,1,H,W],转换为对于的频域特征,在gf内部使用相同大小的权重进行点乘,然后转换回到空域最后的结果是[B,1,H,W]的
        xx = self.cbam1(
            x3 * self.gf3(F.interpolate(fea, x3.shape[2:], mode="bilinear", align_corners=False))
            + xx
        ) * self.ca1(can)
        # ca1的输入是来自x1,x2,x3的特征图的全局平均池化,最后的结果是一个[B,C*3,1,1]的通道特征向量
        xx = F.interpolate(xx, x2.shape[2:], mode="bilinear", align_corners=False)
        xx = self.cbam2(
            x2 * self.gf2(F.interpolate(fea, x2.shape[2:], mode="bilinear", align_corners=False))
            + xx
        ) * self.ca2(can)

        xx = F.interpolate(xx, x1.shape[2:], mode="bilinear", align_corners=False)
        xx = self.cbam3(
            x1 * self.gf1(F.interpolate(fea, x1.shape[2:], mode="bilinear", align_corners=False))
            + xx
        ) * self.ca3(can)
        xx = self.out(xx)
        xx = F.interpolate(xx, out_size, mode="bilinear", align_corners=False)
        return xx
        # 64,128,320,512
