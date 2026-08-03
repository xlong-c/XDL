"""RRDBNet — ESRGAN 主干网络（纯架构，无外部依赖）

来源: BasicSR (xinntao/BasicSR), AESOP-SR 项目使用
许可: Apache 2.0

包含:
- ResidualDenseBlock: 残差密集块 (RDB)
- RRDB: 残差嵌套残差密集块 (Residual-in-Residual Dense Block)
- RRDBNet: ESRGAN 生成器主干 (支持 1x/2x/4x 缩放)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


# ---------------------------------------------------------------------------
# 工具函数 (从 arch_util 内联)
# ---------------------------------------------------------------------------

@torch.no_grad()
def default_init_weights(module_list, scale=1.0, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    if hh % scale != 0:
        x = x[:, :, :-(hh % scale), :]
    if hw % scale != 0:
        x = x[:, :, :, :-(hw % scale)]
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# ---------------------------------------------------------------------------
# 模型组件
# ---------------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    """残差密集块 (Residual Dense Block) — ESRGAN 核心构建块。

    5 层卷积，前 4 层逐层 concat 增长通道，第 5 层压缩回原始通道。
    残差缩放系数 0.2 用于稳定训练。
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """残差嵌套残差密集块 (Residual-in-Residual Dense Block)。

    3 个 RDB 嵌套在一个残差连接中，缩放系数 0.2。
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN 生成器主干网络。

    由 RRDB 组成的超分辨率网络，支持 1x/2x/4x/8x 缩放。
    使用 pixel-unshuffle 在输入端降低空间分辨率以支持 1x/2x 缩放。

    Args:
        num_in_ch: 输入通道数
        num_out_ch: 输出通道数
        scale: 缩放因子 (1, 2, 4, 8)
        num_feat: 特征通道数，默认 64
        num_block: RRDB 块数，默认 23
        num_grow_ch: 密集连接增长率，默认 32
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 上采样
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
