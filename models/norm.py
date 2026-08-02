import torch
from torch import nn
import torch.nn.functional as F

class Norm(nn.Module):
    """BatchNorm in training and stateless InstanceNorm in evaluation."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Norm expected a 4D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Norm expected {self.num_features} channels, got {x.shape[1]}"
            )

        if self.training:
            return F.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=self.weight,
                bias=self.bias,
                training=True,
                momentum=0.0,
                eps=self.eps,
            )
        return F.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=self.weight,
            bias=self.bias,
            use_input_stats=True,
            momentum=0.0,
            eps=self.eps,
        )

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"affine={self.affine}"
        )
