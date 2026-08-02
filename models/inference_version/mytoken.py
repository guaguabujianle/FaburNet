import torch
import torch.nn as nn
from models.conv import DWConv
import torch.nn.functional as F


class SPAM(nn.Module):
    def __init__(self, inp, ratio=1):
        super(SPAM, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.excite = nn.Sequential(
            DWConv(inp),
            nn.Sigmoid()
        )

    def sge(self, x):
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w
        ge = self.excite(x_gather)
        return ge

    def forward(self, x):
        att = self.sge(x)
        out = att * x
        return out


class Token(nn.Module):
    def __init__(self, channels):
        super(Token, self).__init__()
        self.low_freq_extractor = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.high_freq_extractor = DWConv(channels)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.fusion = SPAM(channels)

    def forward(self, x):
        low_freq = self.down(x)
        high_freq = x - F.interpolate(low_freq, size=x.size()[-2:], mode='bilinear', align_corners=True)
        low_freq = F.interpolate(self.low_freq_extractor(low_freq), size=x.size()[-2:], mode='bilinear',
                                 align_corners=True)
        high_freq = self.high_freq_extractor(high_freq)
        fused = low_freq + high_freq
        fused = self.fusion(fused)
        return fused
