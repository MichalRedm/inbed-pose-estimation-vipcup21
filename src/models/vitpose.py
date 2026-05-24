import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from .base import BaseModel
from .registry import register_model


@register_model("vitpose")
class ViTPose(BaseModel):
    """
    ViTPose: Vision Transformer for Human Pose Estimation.
    Uses pre-trained ViT-B-16 from torchvision, with on-the-fly positional
    embedding interpolation for handling 256x256 image inputs.

    Features a classic upsampling decoder:
      ConvTranspose2d (768 -> 256, k=4, s=2, p=1) -> BN -> ReLU ->
      ConvTranspose2d (256 -> 128, k=4, s=2, p=1) -> BN -> ReLU ->
      Conv2d (128, num_joints, k=1) to yield (B, num_joints, 64, 64) keypoint heatmaps.
    """

    def __init__(self, config):
        super().__init__(config)

        # Handle both full config and sub-config dict structures
        if "model" in config:
            model_cfg = config.get("model", {}).get("vitpose", {})
        else:
            model_cfg = config

        self.num_joints = model_cfg.get("num_joints", 14)
        self.in_channels = model_cfg.get("in_channels", 3)
        pretrained_weights_path = model_cfg.get("pretrained_weights_path", None)
        pretrained = model_cfg.get("pretrained", True) if not pretrained_weights_path else False

        # Load backbone
        if pretrained:
            self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            print("[ViTPose] Loaded ImageNet pre-trained ViT-B-16 weights.")
        else:
            self.vit = models.vit_b_16(weights=None)
            print("[ViTPose] Initialized un-pretrained ViT-B-16 weights.")

        # Discard classification heads to save memory and avoid unused parameter gradient warnings
        self.vit.heads = nn.Identity()

        # Adapt first conv_proj if input channels is not 3 (e.g. 1-channel IR)
        if self.in_channels != 3:
            conv_proj = self.vit.conv_proj
            new_conv = nn.Conv2d(
                self.in_channels,
                conv_proj.out_channels,
                kernel_size=conv_proj.kernel_size,
                stride=conv_proj.stride,
                padding=conv_proj.padding,
                bias=conv_proj.bias is not None,
            )
            if pretrained:
                with torch.no_grad():
                    if self.in_channels == 1:
                        new_conv.weight.copy_(conv_proj.weight.mean(dim=1, keepdim=True))
                    else:
                        for c in range(self.in_channels):
                            new_conv.weight[:, c].copy_(conv_proj.weight[:, c % 3])
            self.vit.conv_proj = new_conv
            print(f"[ViTPose] Adapted first conv_proj to in_channels={self.in_channels}.")

        # Upsampling Decoder: maps (B, 768, 16, 16) -> (B, num_joints, 64, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_joints, kernel_size=1, stride=1, padding=0),
        )

        self._init_decoder()

        if pretrained_weights_path:
            self.load_pretrained_coco_weights(pretrained_weights_path)

    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def output_type(self) -> str:
        return "heatmap"

    def forward(self, x):
        # Apply ImageNet normalization internally (expected by torchvision pretrained ViT)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        if x.shape[1] == 3:
            x = (x - mean) / std
        elif x.shape[1] == 1:
            x = (x - mean[:, 0:1]) / std[:, 0:1]

        n, c, h_in, w_in = x.shape

        # 1. Conv projection (patches extraction): (B, 768, h_grid, w_grid)
        patch_feats = self.vit.conv_proj(x)
        h, w = patch_feats.shape[2], patch_feats.shape[3]

        # Flatten and permute: (B, 768, h_grid, w_grid) -> (B, h_grid * w_grid, 768)
        flat_patches = patch_feats.reshape(n, self.vit.hidden_dim, h * w).permute(0, 2, 1)

        # Expand class token to match batch size
        batch_class_token = self.vit.class_token.expand(n, -1, -1)

        # Concatenate class token and flat patches: (B, h_grid * w_grid + 1, 768)
        tokens = torch.cat([batch_class_token, flat_patches], dim=1)

        # 2. Dynamic positional embedding interpolation
        pos_embedding = self.vit.encoder.pos_embedding  # (1, num_patches_original + 1, 768)
        pos_embed_class = pos_embedding[:, :1, :]
        pos_embed_patches = pos_embedding[:, 1:, :]

        # Reshape to 2D grid: (1, 14, 14, 768) -> (1, 768, 14, 14)
        orig_h = orig_w = int(pos_embed_patches.shape[1] ** 0.5)
        pos_embed_patches = pos_embed_patches.reshape(
            1, orig_h, orig_w, self.vit.hidden_dim
        ).permute(0, 3, 1, 2)

        # Interpolate patches position embedding to current grid size (h, w)
        if h != orig_h or w != orig_w:
            pos_embed_patches_resized = F.interpolate(
                pos_embed_patches, size=(h, w), mode="bilinear", align_corners=False
            )
        else:
            pos_embed_patches_resized = pos_embed_patches

        # Reshape back to (1, h * w, 768)
        pos_embed_patches_resized = pos_embed_patches_resized.permute(0, 2, 3, 1).reshape(
            1, h * w, self.vit.hidden_dim
        )

        # Concatenate resized position embedding: (1, h_grid * w_grid + 1, 768)
        pos_embedding_resized = torch.cat([pos_embed_class, pos_embed_patches_resized], dim=1)

        # Add position embeddings and apply dropout
        tokens = tokens + pos_embedding_resized
        tokens = self.vit.encoder.dropout(tokens)

        # 3. Transformer Encoder pass
        encoder_out = self.vit.encoder.layers(tokens)
        encoder_out = self.vit.encoder.ln(encoder_out)

        # Extract spatial patches and discard class token: (B, h_grid * w_grid, 768)
        patch_out = encoder_out[:, 1:]

        # Reshape back to 2D grid: (B, 768, h, w)
        spatial_feats = patch_out.permute(0, 2, 1).reshape(n, self.vit.hidden_dim, h, w)

        # 4. Upsampling Decoder: (B, num_joints, H_out, W_out)
        heatmaps = self.decoder(spatial_feats)

        return heatmaps

    def load_pretrained_coco_weights(self, coco_path: str):
        print(f"[ViTPose] Loading COCO pretrained weights from {coco_path}...")
        coco_state = torch.load(coco_path, map_location='cpu')
        if 'state_dict' in coco_state:
            coco_state = coco_state['state_dict']
        elif 'model' in coco_state:
            coco_state = coco_state['model']
            
        model_state = self.state_dict()
        new_state = {}
        
        # 1. Map backbone weights
        new_state['vit.conv_proj.weight'] = coco_state['backbone.patch_embed.proj.weight']
        new_state['vit.conv_proj.bias'] = coco_state['backbone.patch_embed.proj.bias']
        
        if 'vit.class_token' in model_state:
            if 'backbone.cls_token' in coco_state:
                new_state['vit.class_token'] = coco_state['backbone.cls_token']
            else:
                new_state['vit.class_token'] = model_state['vit.class_token']
                
        # Pos embedding (needs interpolation from 16x12 to 14x14)
        pos_embed_pretrained = coco_state['backbone.pos_embed']  # (1, 193, 768)
        pos_embed_class = pos_embed_pretrained[:, :1, :]
        pos_embed_patches = pos_embed_pretrained[:, 1:, :]      # (1, 192, 768)
        # Reshape to (1, 768, 16, 12)
        pos_embed_patches = pos_embed_patches.reshape(1, 16, 12, self.vit.hidden_dim).permute(0, 3, 1, 2)
        # Interpolate to 14x14 (torchvision default)
        pos_embed_patches_resized = F.interpolate(
            pos_embed_patches, size=(14, 14), mode='bilinear', align_corners=False
        )
        pos_embed_patches_resized = pos_embed_patches_resized.permute(0, 2, 3, 1).reshape(1, 196, self.vit.hidden_dim)
        new_state['vit.encoder.pos_embedding'] = torch.cat([pos_embed_class, pos_embed_patches_resized], dim=1)
        
        # 2. Map blocks
        for i in range(12):
            prefix_coco = f'backbone.blocks.{i}'
            prefix_tv = f'vit.encoder.layers.encoder_layer_{i}'
            
            new_state[f'{prefix_tv}.ln_1.weight'] = coco_state[f'{prefix_coco}.norm1.weight']
            new_state[f'{prefix_tv}.ln_1.bias'] = coco_state[f'{prefix_coco}.norm1.bias']
            new_state[f'{prefix_tv}.ln_2.weight'] = coco_state[f'{prefix_coco}.norm2.weight']
            new_state[f'{prefix_tv}.ln_2.bias'] = coco_state[f'{prefix_coco}.norm2.bias']
            
            new_state[f'{prefix_tv}.self_attention.in_proj_weight'] = coco_state[f'{prefix_coco}.attn.qkv.weight']
            new_state[f'{prefix_tv}.self_attention.in_proj_bias'] = coco_state[f'{prefix_coco}.attn.qkv.bias']
            
            new_state[f'{prefix_tv}.self_attention.out_proj.weight'] = coco_state[f'{prefix_coco}.attn.proj.weight']
            new_state[f'{prefix_tv}.self_attention.out_proj.bias'] = coco_state[f'{prefix_coco}.attn.proj.bias']
            
            new_state[f'{prefix_tv}.mlp.0.weight'] = coco_state[f'{prefix_coco}.mlp.fc1.weight']
            new_state[f'{prefix_tv}.mlp.0.bias'] = coco_state[f'{prefix_coco}.mlp.fc1.bias']
            new_state[f'{prefix_tv}.mlp.3.weight'] = coco_state[f'{prefix_coco}.mlp.fc2.weight']
            new_state[f'{prefix_tv}.mlp.3.bias'] = coco_state[f'{prefix_coco}.mlp.fc2.bias']
            
        # Last norm
        new_state['vit.encoder.ln.weight'] = coco_state['backbone.last_norm.weight']
        new_state['vit.encoder.ln.bias'] = coco_state['backbone.last_norm.bias']
        
        # 3. Map decoder
        new_state['decoder.0.weight'] = coco_state['keypoint_head.deconv_layers.0.weight']
        new_state['decoder.1.weight'] = coco_state['keypoint_head.deconv_layers.1.weight']
        new_state['decoder.1.bias'] = coco_state['keypoint_head.deconv_layers.1.bias']
        new_state['decoder.1.running_mean'] = coco_state['keypoint_head.deconv_layers.1.running_mean']
        new_state['decoder.1.running_var'] = coco_state['keypoint_head.deconv_layers.1.running_var']
        new_state['decoder.1.num_batches_tracked'] = coco_state['keypoint_head.deconv_layers.1.num_batches_tracked']
        
        new_state['decoder.3.weight'] = coco_state['keypoint_head.deconv_layers.3.weight']
        new_state['decoder.4.weight'] = coco_state['keypoint_head.deconv_layers.4.weight']
        new_state['decoder.4.bias'] = coco_state['keypoint_head.deconv_layers.4.bias']
        new_state['decoder.4.running_mean'] = coco_state['keypoint_head.deconv_layers.4.running_mean']
        new_state['decoder.4.running_var'] = coco_state['keypoint_head.deconv_layers.4.running_var']
        new_state['decoder.4.num_batches_tracked'] = coco_state['keypoint_head.deconv_layers.4.num_batches_tracked']
        
        load_res = self.load_state_dict(new_state, strict=False)
        print(f"[ViTPose] Loaded COCO weights successfully with missing: {load_res.missing_keys}")
