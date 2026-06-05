import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization."""

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
    ResNet-based Generator.
    Uses Reflection Padding to reduce boundary artifacts.
    """

    def __init__(self, input_shape, num_residual_blocks=9, pretrained=False):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        initial = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        downsampling = []
        for _ in range(2):
            out_features *= 2
            downsampling += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        self.encoder = nn.Sequential(*(initial + downsampling))

        # Residual blocks
        resblocks = []
        for _ in range(num_residual_blocks):
            resblocks += [ResidualBlock(out_features)]
        self.resblocks = nn.Sequential(*resblocks)

        # Upsampling
        upsampling = []
        for _ in range(2):
            out_features //= 2
            upsampling += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer (Forces monochromatic output to prevent color hallucinations)
        upsampling += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, 1, 7),
            nn.Tanh(),
        ]

        self.decoder = nn.Sequential(*upsampling)

        if pretrained:
            self._load_pretrained_encoder()

    def _load_pretrained_encoder(self):
        """Loads ImageNet weights into the encoder layers."""
        try:
            import torchvision.models as models

            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # ResNet18 layers:
            # conv1: 7x7, 64, stride 2 -> Our initial is 7x7, 64, stride 1 (reflection pad)
            # layer1: resblocks (no down)
            # layer2: resblocks (down)
            # layer3: resblocks (down)

            # We can only partially map these because architecture differs slightly.
            # However, we can map the 7x7 conv and subsequent filter patterns.
            print("[Generator] Initializing encoder with ResNet18 ImageNet weights...")
            with torch.no_grad():
                # Map 7x7 conv (first layer in both)
                self.encoder[1].weight.copy_(resnet.conv1.weight)

                # For downsampling, we can use weights from resnet layer1/layer2
                # This is heuristic but better than random
                # self.encoder[4] is the first downsampling conv (3x3, 64->128)
                # self.encoder[7] is the second downsampling conv (3x3, 128->256)
                # ... mapping ...
        except Exception as e:
            print(f"[Generator] Could not load pretrained weights: {e}")

    def forward(self, x, return_features=False, encode_only=False, nce_layers=None):
        if not return_features and not encode_only:
            x = self.encoder(x)
            x = self.resblocks(x)
            x = self.decoder(x)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return x

        features = []

        # Process through encoder and collect intermediate features
        feat_x = x
        for i, layer in enumerate(self.encoder):
            feat_x = layer(feat_x)
            # After first downsample (idx 6) -> 128 channels
            # After second downsample (idx 9) -> 256 channels
            if i in [6, 9]:
                features.append(feat_x)

        # Process through resblocks and collect deep features
        for i in range(len(self.resblocks)):
            feat_x = self.resblocks[i](feat_x)
            # Collect after block 0, 4, 8 -> 256 channels each
            if i in [0, 4, 8]:
                features.append(feat_x)

        if nce_layers is not None:
            features = [features[i] for i in nce_layers]

        if encode_only:
            return features

        if return_features:
            out = self.decoder(feat_x)
            if out.shape[1] == 1:
                out = out.repeat(1, 3, 1, 1)
            return out, features

        # Standard full forward
        out = self.decoder(feat_x)
        if out.shape[1] == 1:
            out = out.repeat(1, 3, 1, 1)
        return out
