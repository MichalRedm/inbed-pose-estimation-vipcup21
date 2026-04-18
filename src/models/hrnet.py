import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HRNet(nn.Module):
    def __init__(self, config):
        super(HRNet, self).__init__()
        # Simplified HRNet for educational/MVP purposes
        # Based on w32 configuration
        num_joints = config.get("num_joints", 14)
        in_channels = config.get("in_channels", 3)

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))

        # High-Resolution Head (Simplified Transition to Heatmap)
        self.final_layer = nn.Conv2d(64, num_joints, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x = self.final_layer(x)
        return x


def get_pose_net(config, is_train=True):
    model = HRNet(config)
    return model
