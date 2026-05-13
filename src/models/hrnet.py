"""
HRNet-W32: High-Resolution Network for Pose Estimation.

Implements the full parallel-stream architecture as described in:
  "Deep High-Resolution Representation Learning for Visual Recognition"
  Wang et al., TPAMI 2020.

W32 configuration:
  - Stream widths: 32, 64, 128, 256 channels
  - 4 stages with repeated multi-resolution fusions
  - Output: (B, num_joints, H/4, W/4) heatmaps
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from . import register_model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ── Basic Building Block ──────────────────────────────────────────────────────


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


# ── Layer Builder ─────────────────────────────────────────────────────────────


def make_layer(block, inplanes, planes, num_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


# ── Multi-Resolution Fusion Module ───────────────────────────────────────────


class FusionLayer(nn.Module):
    """
    Fuses features from num_branches parallel streams.
    """

    def __init__(self, num_branches, channels):
        super().__init__()
        self.num_branches = num_branches
        # Use 'layers' to avoid conflict with the parent member name 'fuse_layers'
        self.layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j == i:
                    fuse_layer.append(nn.Identity())
                elif j < i:
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops += [
                                conv3x3(channels[j], channels[i], stride=2),
                                nn.BatchNorm2d(channels[i]),
                            ]
                        else:
                            ops += [
                                conv3x3(channels[j], channels[j], stride=2),
                                nn.BatchNorm2d(channels[j]),
                                nn.ReLU(inplace=True),
                            ]
                    fuse_layer.append(nn.Sequential(*ops))
                else:
                    fuse_layer.append(
                        nn.Sequential(
                            conv1x1(channels[j], channels[i]),
                            nn.BatchNorm2d(channels[i]),
                        )
                    )
            self.layers.append(fuse_layer)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = []
        for i in range(self.num_branches):
            acc = None
            for j in range(self.num_branches):
                feat = self.layers[i][j](x[j])
                if j > i:
                    feat = F.interpolate(
                        feat, size=x[i].shape[2:], mode="bilinear", align_corners=True
                    )
                acc = feat if acc is None else acc + feat
            y.append(self.relu(acc))
        return y


# ── HRNet Module (Parallel Branches + Fusion) ────────────────────────────────


class HRNetModule(nn.Module):
    def __init__(self, num_branches, channels, num_blocks=4):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                make_layer(BasicBlock, channels[b], channels[b], num_blocks)
                for b in range(num_branches)
            ]
        )
        self.fuse_layers = FusionLayer(num_branches, channels)

    def forward(self, x):
        x = [self.branches[b](x[b]) for b in range(len(x))]
        x = self.fuse_layers(x)
        return x


# ── Transition Layer ──────────────────────────────────────────────────────────


def make_transition(in_channels, out_channels_list):
    """
    Creates a transition layer that:
      - Adapts existing branches to new channel sizes.
      - Adds a new strided branch (half resolution).
    """
    num_out = len(out_channels_list)
    layers = nn.ModuleList()
    for i in range(num_out):
        if i < len(in_channels):
            # Existing branch — adapt channels if needed
            if in_channels[i] != out_channels_list[i]:
                layers.append(
                    nn.Sequential(
                        conv3x3(in_channels[i], out_channels_list[i]),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                layers.append(nn.Identity())
        else:
            # New branch — stride-2 conv from the last existing branch
            layers.append(
                nn.Sequential(
                    conv3x3(in_channels[-1], out_channels_list[i], stride=2),
                    nn.BatchNorm2d(out_channels_list[i]),
                    nn.ReLU(inplace=True),
                )
            )
    return layers


# ── HRNet-W32 ─────────────────────────────────────────────────────────────────


@register_model("hrnet")
class HRNet(BaseModel):
    """
    HRNet-W32 for human pose estimation.

    Architecture:
      Stem (2× stride-2 conv) → Stage1 (bottleneck) →
      Transition1 → Stage2 (2 branches, 1 module) →
      Transition2 → Stage3 (3 branches, 4 modules) →
      Transition3 → Stage4 (4 branches, 3 modules) →
      Head (upsample all → cat → 1×1 conv → num_joints heatmaps)
    """

    # W32 channel widths per stream
    W32 = [32, 64, 128, 256]

    def __init__(self, config):
        super().__init__(config)
        # Handle both full config and sub-config
        if "model" in config:
            config_model = config.get("model", {}).get("hrnet", {})
        else:
            config_model = config

        num_joints = config_model.get("num_joints", 14)
        in_channels = config_model.get("in_channels", 1)
        C = self.W32  # shorthand

        # --- Stem ---
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # --- Stage 1 ---
        self.layer1 = make_layer(Bottleneck, 64, 64, num_blocks=4)

        # --- Transition 1 ---
        self.transition1 = make_transition([256], [C[0], C[1]])

        # --- Stage 2: 2 branches ---
        self.stage2 = nn.Sequential(
            *[
                HRNetModule(num_branches=2, channels=[C[0], C[1]], num_blocks=4)
                for _ in range(1)
            ]
        )

        # --- Transition 2: [C[0], C[1]] → [C[0], C[1], C[2]] ---
        self.transition2 = make_transition([C[0], C[1]], [C[0], C[1], C[2]])

        # --- Stage 3: 3 branches ---
        self.stage3 = nn.Sequential(
            *[
                HRNetModule(num_branches=3, channels=[C[0], C[1], C[2]], num_blocks=4)
                for _ in range(4)
            ]
        )

        # --- Transition 3: [C[0..2]] → [C[0..3]] ---
        self.transition3 = make_transition([C[0], C[1], C[2]], [C[0], C[1], C[2], C[3]])

        # --- Stage 4: 4 branches ---
        self.stage4 = nn.Sequential(
            *[
                HRNetModule(
                    num_branches=4, channels=[C[0], C[1], C[2], C[3]], num_blocks=4
                )
                for _ in range(3)
            ]
        )

        # --- Head ---
        total_channels = sum(C)  # 32+64+128+256 = 480
        self.head = nn.Sequential(
            conv1x1(total_channels, total_channels),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels, num_joints, kernel_size=1),
        )

        # --- Pre-trained Initialization ---
        pretrained = config_model.get("pretrained", False)
        if pretrained:
            self._load_pretrained_weights(pretrained)

    def _load_pretrained_weights(self, pretrained_source):
        """
        Loads pre-trained weights from a URL or local path.
        Default URL is HRNet-W32 (OpenMMLab mirror).
        """
        DEFAULT_URL = "https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth"
        
        if isinstance(pretrained_source, bool) and pretrained_source:
            url = DEFAULT_URL
        elif isinstance(pretrained_source, str) and pretrained_source.startswith("http"):
            url = pretrained_source
        elif isinstance(pretrained_source, str) and os.path.exists(pretrained_source):
            print(f"[HRNet] Loading local pre-trained weights from {pretrained_source}")
            state_dict = torch.load(pretrained_source, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            return
        else:
            print(f"[HRNet] Pretrained source '{pretrained_source}' invalid or not found. Skipping.")
            return

        print(f"[HRNet] Downloading pre-trained weights from {url}")
        try:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(url, map_location="cpu", progress=True)
            
            # Handle possible key nesting in official weights
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Filter and remap keys
            in_channels = self.conv1.in_channels
            filtered_state = {}
            for k, v in state_dict.items():
                # Skip head as it won't match our num_joints/structure
                if k.startswith("head") or k.startswith("fc"):
                    continue
                # Skip stem if in_channels != 3
                if k.startswith("conv1") and in_channels != 3:
                    continue
                
                # Remap fuse_layers to match our FusionLayer.layers nesting
                # Official: stage2.0.fuse_layers.0.1.0.weight
                # Ours:     stage2.0.fuse_layers.layers.0.1.0.weight
                new_k = k.replace("fuse_layers.", "fuse_layers.layers.")
                
                filtered_state[new_k] = v
                
            msg = self.load_state_dict(filtered_state, strict=False)
            print(f"[HRNet] Pre-trained weights loaded. Matched: {len(filtered_state) - len(msg.missing_keys)}, Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            if in_channels != 3:
                print(f"[HRNet] Note: 'conv1' was skipped due to in_channels={in_channels} (expected 3 for ImageNet weights).")
        except Exception as e:
            print(f"[HRNet] Failed to load pre-trained weights: {e}")


    @property
    def output_type(self) -> str:
        return "heatmap"

    def _apply_transition(self, transition, x_list):
        """Apply transition layers, extending x_list if new branches are added."""
        result = []
        for i, layer in enumerate(transition):
            if i < len(x_list):
                result.append(layer(x_list[i]))
            else:
                # New branch: starts from the last existing stream
                result.append(layer(x_list[-1]))
        return result

    def forward(self, x, return_features=False):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Stage 1 (single bottleneck stream)
        x = self.layer1(x)

        # Transition 1 → Stage 2
        x = self._apply_transition(self.transition1, [x])
        x = self.stage2(x)

        # Transition 2 → Stage 3
        x = self._apply_transition(self.transition2, x)
        x = self.stage3(x)

        # Transition 3 → Stage 4
        x = self._apply_transition(self.transition3, x)
        x = self.stage4(x)

        # Head: upsample all streams to highest resolution, concatenate
        target_size = x[0].shape[2:]
        upsampled = [
            F.interpolate(xi, size=target_size, mode="bilinear", align_corners=True)
            for xi in x
        ]
        features = torch.cat(upsampled, dim=1)
        heatmaps = self.head(features)

        if return_features:
            return heatmaps, features
        return heatmaps
