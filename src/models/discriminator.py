import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple, cast


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, GradientReversalFunction.apply(x, self.alpha))


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 480) -> None:
        super(DomainDiscriminator, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )
        self.ad_layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        if alpha is not None:
            x = cast(torch.Tensor, GradientReversalFunction.apply(x, alpha))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        x = self.ad_layer3(x)
        x = self.classifier(x)
        return x
