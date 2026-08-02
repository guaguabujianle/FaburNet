import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models.inference_version.conv import DepthwiseSeparableConv

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,H,W):
        B, _, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.view(B, C, H, W)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.out_norm = norm_layer(dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = self.out_norm(x)
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape  # B,C,H,W
        x = x.flatten(2).transpose(1, 2) # B,H*W,C
        x = self.norm(x)

        return x, H, W

class Hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, nclass):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, skip_channels, 3, padding=1, dilation=1, bias=False)
        self.relu = Hswish()
        self.bn = nn.BatchNorm2d(skip_channels)

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2*skip_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_skip):
        x = self.relu(F.interpolate(self.bn(self.conv(x)), scale_factor=(2, 2), mode='bilinear'))

        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_fusion(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride,padding):
        super(EncoderBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels,out_channels, kernel_size, stride, padding)
        self.relu = Hswish()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(F.max_pool2d(self.bn(self.conv(x)), 2, 2))


class AoPNet(nn.Module):
    def __init__(self, num_classes=3, img_size=256,
                 embed_dims=[128, 160, 256],mlp_ratio=1.0,drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], deep_supervision=True,**kwargs):
        super().__init__()
        self.ds = deep_supervision
        self.spatial_encoder1 = EncoderBlock(3, 16, 3, 1, 1)
        self.spatial_encoder2 = EncoderBlock(16, 32, 3, 1, 1)
        self.spatial_encoder3 = EncoderBlock(32, 128, 3, 1, 1)

        self.relu = Hswish()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.channel_encoder1 = nn.ModuleList([MLPBlock(
            dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,)])

        self.channel_encoder2 = nn.ModuleList([MLPBlock(
            dim=embed_dims[2], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,)])

        self.channel_decoder1 = nn.ModuleList([MLPBlock(
            dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,)])

        self.channel_decoder2 = nn.ModuleList([MLPBlock(
            dim=embed_dims[0], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer,)])

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.spatial_decoder1 = DecoderBlock(256, 160, 160, 3)
        self.spatial_decoder2 = DecoderBlock(160, 128, 128, 3)
        self.spatial_decoder3 = DecoderBlock(128, 32, 32, 3)
        self.spatial_decoder4 = DecoderBlock(32, 16, 16, 3)

        self.out_conv = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        out = self.spatial_encoder1(x)
        t1 = out # 128,128
        out = self.spatial_encoder2(out)
        t2 = out # 64,64
        out = self.spatial_encoder3(out)
        t3 = out  # B,128,32,32
        
        # similar 3
        out, H, W = self.patch_embed1(out)

        # similar 4
        for i, blk in enumerate(self.channel_encoder1):
            out = blk(out, H, W)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out  # B,160,16,16

        # similar 5
        out, H, W = self.patch_embed2(out)

        # similar 6
        for i, blk in enumerate(self.channel_encoder2):
            out = blk(out, H, W)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out= self.spatial_decoder1(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        # similar 8
        for i, blk in enumerate(self.channel_decoder1):
            out = blk(out, H, W)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out= self.spatial_decoder2(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        # similar 10
        for i, blk in enumerate(self.channel_decoder2):
            out = blk(out, H, W)

        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.spatial_decoder3(out, t2)

        out= self.spatial_decoder4(out, t1)
        out = self.relu(F.interpolate(self.out_conv(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)



if __name__ == '__main__':
    from thop import clever_format, profile
    net = AoPNet()
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(net, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    out = net(input)