import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp)
        self.pointwise = nn.Conv2d(inp, outp, kernel_size=1, stride=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)



