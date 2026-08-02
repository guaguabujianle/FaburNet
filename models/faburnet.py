from models.mytoken import Token
from models.aopnet import AoPNet
from models.norm import Norm
import torch
from torch import nn
import torch.nn.functional as F


class Hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

def compute_entropy_map(logits):
    prob = F.softmax(logits, dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # (B, H, W)
    entropy = entropy / torch.log(torch.tensor(prob.size(1), dtype=torch.float32))  # normalize to [0,1]
    return entropy  # (B, H, W)


class StructureAwareUncertaintyEnhance(nn.Module):
    def __init__(self, in_channels, smooth=True, init_laplace=True):
        super().__init__()
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False)
        self.smooth = smooth
        

        self.fuse_reduce = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                Norm(in_channels),
                nn.ReLU(inplace=True)
            )

        if smooth:
            self.refine = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                Norm(1),
                nn.ReLU(inplace=True)
            )
        if init_laplace:
            laplace_kernel = torch.tensor([[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                self.edge_conv.weight.copy_(laplace_kernel)

    def forward(self, entropy_map, shallow_feat,x):
        x_cat = torch.cat([shallow_feat, x], dim=1)
        x_cat = self.fuse_reduce(x_cat)
        edge_map = torch.sigmoid(self.edge_conv(x_cat))  # B×1×H×W
        if self.smooth:
            edge_map = self.refine(edge_map)
        refined_map = entropy_map * edge_map  # 空间增强 mask
        return refined_map
    

# ========== DecoderBlock with Token + SAUE ========== #
class DecoderBlockWithToken(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, nclass):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, skip_channels, 3, padding=1, bias=False)
        self.relu = Hswish()
        self.bn = Norm(skip_channels)

        self.token = Token(out_channels)

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2 * skip_channels, out_channels, 3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

        self.coarse_pred = nn.Conv2d(out_channels, nclass, 1, bias=False)
        if in_channels in [256,160]:
            self.alpha = nn.Parameter(torch.tensor(1.2))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1))

        # 结构感知不确定性增强模块

        self.saue = StructureAwareUncertaintyEnhance(
            in_channels=skip_channels,
            smooth=True,
            init_laplace=True
        )


    def forward(self, x, x_skip):
        x = self.relu(F.interpolate(self.bn(self.conv(x)), scale_factor=2, mode='bilinear', align_corners=False))
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv_fusion(x)
        x = self.token(x) + x
        
        pred_coarse = self.coarse_pred(x)  # (B, C, H, W)
        entropy_map = compute_entropy_map(pred_coarse.detach()).unsqueeze(1)  # (B, 1, H, W)
        
        refined_map = self.saue(entropy_map, x_skip.detach(),x.detach())  # (B, 1, H, W)
        x = x * (1 + self.alpha*refined_map)

        weighted_map = refined_map.detach().clone()

        return x, weighted_map, pred_coarse # refined_map


class FaburNet(AoPNet):
    def __init__(self, use_token_decoder=(True, True, True, True)):
        super().__init__()

        for i, use_token in enumerate(use_token_decoder):
            name = f"spatial_decoder{i+1}"
            block = getattr(self, name)
            if use_token:
                new_block = DecoderBlockWithToken(
                    block.conv.in_channels,
                    block.conv.out_channels,
                    block.conv_fusion[0].out_channels,
                    block.ds_block.out_channels
                )
                setattr(self, name, new_block)
            else:
                setattr(self, name, block)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


    

if __name__ == '__main__':
    from thop import clever_format, profile
    model = FaburNet()
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops, params)
    output = model(input)


