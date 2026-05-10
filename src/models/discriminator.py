import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels=480):
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

    def forward(self, x, alpha=None):
        if alpha is not None:
            x = GradientReversalFunction.apply(x, alpha)
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        x = self.ad_layer3(x)
        x = self.classifier(x)
        return x
