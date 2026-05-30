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
from typing import List, Tuple, Union, Any, Dict, Optional, Type, cast
from .base import BaseModel
from .registry import register_model


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding.

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Stride of the convolution.

    Returns:
        A Conv2d layer.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution.

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Stride of the convolution.

    Returns:
        A Conv2d layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ── Basic Building Block ──────────────────────────────────────────────────────


class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.
    Used in HRNet Stage 2, 3, and 4.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        Initializes the BasicBlock.

        Args:
            inplanes: Number of input channels.
            planes: Number of output channels.
            stride: Stride of the first convolution.
            downsample: Optional downsampling layer for residual connection.
        """
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return cast(torch.Tensor, self.relu(out + residual))


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck block.
    Used in HRNet Stage 1.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        Initializes the Bottleneck block.

        Args:
            inplanes: Number of input channels.
            planes: Number of output channels (before expansion).
            stride: Stride of the 3x3 convolution.
            downsample: Optional downsampling layer for residual connection.
        """
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return cast(torch.Tensor, self.relu(out + residual))


# ── Layer Builder ─────────────────────────────────────────────────────────────


def make_layer(
    block: Type[Union[BasicBlock, Bottleneck]],
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    """
    Creates a layer with multiple blocks of the same type.

    Args:
        block: Type of block (BasicBlock or Bottleneck).
        inplanes: Number of input channels.
        planes: Number of output channels for each block.
        num_blocks: Number of blocks in the layer.
        stride: Stride of the first block.

    Returns:
        A Sequential container of blocks.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers: List[nn.Module] = [block(inplanes, planes, stride, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


# ── Multi-Resolution Fusion Module ───────────────────────────────────────────


class FusionLayer(nn.Module):
    """
    Fuses features from num_branches parallel streams.
    Implements multi-resolution fusion logic.
    """

    def __init__(self, num_branches: int, channels: List[int]) -> None:
        """
        Initializes the FusionLayer.

        Args:
            num_branches: Number of parallel streams.
            channels: List of channel widths for each stream.
        """
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
                    ops: List[nn.Module] = []
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

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fuses features from multiple input streams.

        Args:
            x: List of feature tensors from each stream.

        Returns:
            List of fused feature tensors.
        """
        y = []
        for i in range(self.num_branches):
            acc = None
            for j in range(self.num_branches):
                fuse_row = cast(nn.ModuleList, self.layers[i])
                feat = cast(torch.Tensor, fuse_row[j](x[j]))
                if j > i:
                    feat = F.interpolate(
                        feat, size=x[i].shape[2:], mode="bilinear", align_corners=True
                    )
                acc = feat if acc is None else acc + feat
            y.append(cast(torch.Tensor, self.relu(acc)))
        return y


# ── HRNet Module (Parallel Branches + Fusion) ────────────────────────────────


class HRNetModule(nn.Module):
    """
    A single HRNet module consisting of parallel branches followed by a fusion layer.
    """

    def __init__(
        self, num_branches: int, channels: List[int], num_blocks: int = 4
    ) -> None:
        """
        Initializes the HRNetModule.

        Args:
            num_branches: Number of parallel streams.
            channels: List of channel widths for each stream.
            num_blocks: Number of blocks per branch.
        """
        super().__init__()
        self.branches = nn.ModuleList(
            [
                make_layer(BasicBlock, channels[b], channels[b], num_blocks)
                for b in range(num_branches)
            ]
        )
        self.fuse_layers = FusionLayer(num_branches, channels)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass."""
        x = [cast(torch.Tensor, self.branches[b](x[b])) for b in range(len(x))]
        x = self.fuse_layers(x)
        return x


# ── Transition Layer ──────────────────────────────────────────────────────────


def make_transition(
    in_channels: List[int], out_channels_list: List[int]
) -> nn.ModuleList:
    """
    Creates a transition layer that:
      - Adapts existing branches to new channel sizes.
      - Adds a new strided branch (half resolution).

    Args:
        in_channels: List of input channel widths.
        out_channels_list: List of desired output channel widths.

    Returns:
        A ModuleList of transition operations.
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
    W32: List[int] = [32, 64, 128, 256]
    in_channels: int

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes HRNet-W32.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        # Handle both full config and sub-config
        if "model" in config:
            config_model = config.get("model", {}).get("hrnet", {})
        else:
            config_model = config

        num_joints = config_model.get("num_joints", 14)
        self.in_channels = config_model.get("in_channels", 1)
        C = self.W32  # shorthand

        # --- Stem ---
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1, bias=False)
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

        # --- Selective Freezing for Stabilized Transfer Learning ---
        if config_model.get("freeze_stem", False):
            print(
                "Selective Freezing: Freezing HRNet Stem layers (conv1, bn1, conv2, bn2)..."
            )
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
            for p in self.conv2.parameters():
                p.requires_grad = False
            for p in self.bn2.parameters():
                p.requires_grad = False

        if config_model.get("freeze_stage1", False):
            print("Selective Freezing: Freezing HRNet Stage 1 layers (layer1)...")
            for p in self.layer1.parameters():
                p.requires_grad = False

    def _load_pretrained_weights(self, pretrained_source: Union[bool, str]) -> None:
        """
        Loads pre-trained weights from a URL or local path.
        Default URL is HRNet-W32 (OpenMMLab mirror).

        Args:
            pretrained_source: URL, local path, or True to use default URL.
        """
        DEFAULT_URL = "https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth"

        if isinstance(pretrained_source, bool) and pretrained_source:
            url = DEFAULT_URL
        elif isinstance(pretrained_source, str) and pretrained_source.startswith(
            "http"
        ):
            url = pretrained_source
        elif isinstance(pretrained_source, str) and os.path.exists(pretrained_source):
            print(f"[HRNet] Loading local pre-trained weights from {pretrained_source}")
            state_dict = torch.load(pretrained_source, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
            return
        else:
            print(
                f"[HRNet] Pretrained source '{pretrained_source}' invalid or not found. Skipping."
            )
            return

        print(f"[HRNet] Downloading pre-trained weights from {url}")
        try:
            from torch.hub import load_state_dict_from_url

            state_dict = load_state_dict_from_url(
                url, map_location="cpu", progress=True
            )

            # Handle possible key nesting in official weights
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Filter and remap keys
            in_channels = self.conv1.in_channels
            filtered_state = {}
            model_keys = set(self.state_dict().keys())

            import re

            for k, v in state_dict.items():
                # Skip head as it won't match our num_joints/structure
                if k.startswith("head") or k.startswith("fc"):
                    continue

                new_k = k

                # 1. Handle transitions nesting
                m = re.match(r"^transition(\d+)\.(\d+)\.0\.(\d+)\.(.+)$", new_k)
                if m:
                    trans_num, branch_num, op_idx, param_name = m.groups()
                    new_k = f"transition{trans_num}.{branch_num}.{op_idx}.{param_name}"

                # 2. Handle stage multi-resolution fusions nesting (downsampling vs upsampling)
                if "fuse_layers." in new_k:
                    new_k = new_k.replace("fuse_layers.", "fuse_layers.layers.")

                    parts = new_k.split(".")
                    if (
                        len(parts) >= 9
                        and parts[2] == "fuse_layers"
                        and parts[3] == "layers"
                    ):
                        try:
                            i = int(parts[4])
                            j = int(parts[5])
                            if j < i:
                                step_idx = int(parts[6])
                                op_idx = int(parts[7])
                                param_name = ".".join(parts[8:])

                                flat_idx = 0
                                for step in range(step_idx):
                                    flat_idx += 3  # conv, BN, ReLU
                                flat_idx += op_idx

                                new_k = f"{parts[0]}.{parts[1]}.fuse_layers.layers.{i}.{j}.{flat_idx}.{param_name}"
                        except ValueError:
                            pass

                # Special handling for stem if in_channels != 3
                if k.startswith("conv1") and in_channels != 3:
                    if in_channels == 1 and v.dim() == 4 and v.shape[1] == 3:
                        filtered_state[k] = v.mean(dim=1, keepdim=True)
                    continue

                if new_k in model_keys:
                    filtered_state[new_k] = v
                elif k in model_keys:
                    filtered_state[k] = v

            msg = self.load_state_dict(filtered_state, strict=False)
            print(
                f"[HRNet] Pre-trained weights loaded. Matched: {len(filtered_state) - len(msg.missing_keys)}, Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}"
            )
            if in_channels != 3:
                if "conv1.weight" in filtered_state:
                    print(
                        f"[HRNet] Note: 'conv1' adapted from 3 to {in_channels} channels."
                    )
                else:
                    print(
                        f"[HRNet] Note: 'conv1' was skipped due to in_channels={in_channels} (expected 3 for ImageNet weights)."
                    )
        except Exception as e:
            print(f"[HRNet] Failed to load pre-trained weights: {e}")

    def unfreeze_all(self) -> None:
        """Re-enable gradient tracking on all parameters (for progressive unfreezing)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[HRNet] All parameters unfrozen (progressive unfreezing Phase 2).")

    @property
    def output_type(self) -> str:
        """Returns the type of output the model produces ('heatmap')."""
        return "heatmap"

    def _apply_transition(
        self, transition: nn.ModuleList, x_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply transition layers, extending x_list if new branches are added.

        Args:
            transition: ModuleList of transition layers.
            x_list: List of input feature tensors.

        Returns:
            List of output feature tensors.
        """
        result = []
        for i, layer in enumerate(transition):
            if i < len(x_list):
                result.append(cast(torch.Tensor, layer(x_list[i])))
            else:
                # New branch: starts from the last existing stream
                result.append(cast(torch.Tensor, layer(x_list[-1])))
        return result

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of HRNet-W32.

        Args:
            x: Input image tensor of shape (B, C, H, W).
            **kwargs: Additional arguments (e.g., return_features).

        Returns:
            Heatmaps or a tuple of (heatmaps, features).
        """
        return_features = kwargs.get("return_features", False)
        # Stem
        x = cast(torch.Tensor, self.relu(self.bn1(self.conv1(x))))
        x = cast(torch.Tensor, self.relu(self.bn2(self.conv2(x))))

        # Stage 1 (single bottleneck stream)
        x = cast(torch.Tensor, self.layer1(x))

        # Transition 1 → Stage 2
        x_list = self._apply_transition(self.transition1, [x])
        x_list = self.stage2(x_list)

        # Transition 2 → Stage 3
        x_list = self._apply_transition(self.transition2, x_list)
        x_list = self.stage3(x_list)

        # Transition 3 → Stage 4
        x_list = self._apply_transition(self.transition3, x_list)
        x_list = self.stage4(x_list)

        # Head: upsample all streams to highest resolution, concatenate
        target_size = x_list[0].shape[2:]
        upsampled = [
            F.interpolate(xi, size=target_size, mode="bilinear", align_corners=True)
            for xi in x_list
        ]
        features = torch.cat(upsampled, dim=1)
        heatmaps = cast(torch.Tensor, self.head(features))

        if return_features:
            return heatmaps, features
        return heatmaps
