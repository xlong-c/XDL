"""AESOP AutoEncoder — 自编码监督感知超分辨率损失模型

来源: AESOP (CVPR 2025), 2minkyulee/AESOP-SR
许可: Apache 2.0

核心思路: 用 L_pix 预训练的 Auto-Encoder 剥离感知方差 (VE),
在 AE 输出空间计算 fidelity loss, 突破感知-失真权衡。

包含:
- AutoEncoder_RRDBNet: 标准 AE (Encoder: Conv+PixUnshuffle+RRDB, Decoder: RRDBNet)
- ProbabilisticAutoEncoder_RRDBNet: 概率 AE (额外输出 sigma)
- BicubicDown_RRDBNet: 消融用 (bicubic 下采样替代 encoder)
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from xdl.model.lowlevel.rrdb_arch import RRDBNet, RRDB, make_layer


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

@torch.no_grad()
def _default_init_weights(module_list, scale: float = 1.0, bias_fill: float = 0, **kwargs):
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


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class AutoEncoder_RRDBNet(nn.Module):
    """AESOP 核心 Auto-Encoder — 用于计算 L_AESOP loss。

    结构:
      Encoder: Conv(3→feat//16) → PixelUnshuffle(4x) → 2×RRDB → Conv(feat→3)
      Decoder: RRDBNet (标准 ESRGAN 生成器)

    AE bottleneck 尺寸 = LR 尺寸, 强制丢弃感知方差 (VE),
    只保留保真度偏差 (SE)。

    Args:
        num_in_ch: 输入通道数 (默认 3)
        num_out_ch: 输出通道数 (默认 3)
        num_feat: 特征通道数 (默认 64)
        num_block: Decoder RRDB 块数 (默认 23)
        num_grow_ch: 密集连接增长率 (默认 32)
        scale: 超分缩放因子 (默认 4)
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()

        # Decoder — 标准 RRDBNet 生成器
        self.decoder = RRDBNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )

        # Encoder
        self.conv_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat // 16, 3, 1, 1),
            nn.Conv2d(num_feat // 16, num_feat // 16, 3, 1, 1),
        )
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.PixelUnshuffle(2),
        )
        self.body = make_layer(RRDB, num_basic_block=2, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.Conv2d(num_feat, num_in_ch, 3, 1, 1),
        )

        self.encoder = nn.Sequential(
            self.conv_first,
            self.down,
            self.body,
            self.conv_last,
        )

        self.dec_is_frozen = False
        self.enc_is_frozen = False
        _default_init_weights([self.conv_first, self.conv_last], 0.1)

    def freeze_encoder(self):
        if not self.enc_is_frozen:
            self.enc_is_frozen = True
            for param in self.encoder.parameters():
                param.requires_grad = False

    def freeze_decoder(self):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            for param in self.decoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        if self.enc_is_frozen:
            self.enc_is_frozen = False
            for param in self.encoder.parameters():
                param.requires_grad = True

    def unfreeze_decoder(self):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            for param in self.decoder.parameters():
                param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bottleneck = self.encoder(x)
        out = self.decoder(bottleneck)
        if return_bottleneck:
            return out, bottleneck
        return out


class ProbabilisticAutoEncoder_RRDBNet(AutoEncoder_RRDBNet):
    """概率 Auto-Encoder — 额外输出 sigma (不确定性)。

    在 Decoder 的 body 和 conv_first 上注册 forward hook,
    提取特征后通过 sigma_branch 输出方差图。
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
            scale=scale,
        )

        # Sigma 分支 — 从 Decoder 中间特征预测方差
        self.sigma_branch = nn.Sequential(
            RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch),
            nn.Conv2d(num_feat, 3, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(3, 3, 3, 1, 1),
        )

        self._body_feat = None
        self._conv_first_feat = None
        self._body_hook = self.decoder.body.register_forward_hook(self._hook_body)
        self._conv_first_hook = self.decoder.body[0].register_forward_hook(self._hook_conv_first)

    def _hook_body(self, module, input, output):
        self._body_feat = output.clone()

    def _hook_conv_first(self, module, input, output):
        self._conv_first_feat = output.clone()

    def forward(
        self,
        x: torch.Tensor,
        return_bottleneck: bool = False,
        return_sigma: bool = False,
    ):
        bottleneck = self.encoder(x)
        out = self.decoder(bottleneck)

        feat = self._body_feat + self._conv_first_feat
        sigma = self.sigma_branch(feat)

        output = [out]
        if return_bottleneck:
            output.append(bottleneck)
        if return_sigma:
            output.append(sigma)

        return output[0] if len(output) == 1 else tuple(output)
