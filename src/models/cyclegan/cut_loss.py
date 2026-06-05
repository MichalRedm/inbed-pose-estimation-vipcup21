"""
Provides loss functions and modules for Contrastive Unpaired Translation (CUT).
Includes the patch sampling MLP and the InfoNCE contrastive loss.
"""
import torch
import torch.nn as nn


class PatchSampleF(nn.Module):
    """
    MLP used for projecting intermediate feature maps into a shared
    embedding space for Patchwise Contrastive Learning (CUT).
    """

    def __init__(self, in_channels_list=[128, 256, 256, 256, 256], embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        self.mlps = nn.ModuleList()
        for in_channels in in_channels_list:
            mlp = nn.Sequential(
                nn.Linear(in_channels, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )
            self.mlps.append(mlp)

    def forward(self, features, patch_ids=None, num_patches=256):
        """
        Samples patches from features and projects them to the embedding space.

        Args:
            features: List of feature maps from the encoder.
            patch_ids: List of patch indices to sample. If None, samples randomly.
            num_patches: Number of patches to sample per feature map.

        Returns:
            A tuple containing a list of projected patch features and a list of sampled patch ids.
        """
        return_ids = []
        return_feats = []

        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)

            if patch_ids is not None:
                patch_id = patch_ids[i]
            else:
                # Sample random patches without replacement
                patch_id = torch.randperm(feat_reshape.shape[1], device=feat.device)
                patch_id = patch_id[:num_patches]

            # Gather patches
            x_sample = feat_reshape[:, patch_id, :]  # (B, num_patches, C)

            # Project to embed_dim
            mlp = self.mlps[i]
            x_sample = mlp(x_sample)

            # L2 Normalize
            norm = x_sample.pow(2).sum(2, keepdim=True).sqrt()
            x_sample = x_sample.div(norm + 1e-7)

            return_ids.append(patch_id)
            return_feats.append(x_sample)

        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    """
    Patchwise InfoNCE Loss.
    Maximizes mutual information between corresponding patches.
    """

    def __init__(self, tau=0.07):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        self.tau = tau

    def forward(self, feat_q, feat_k):
        """
        Calculates the InfoNCE loss between generated features and source features.

        Args:
            feat_q: List of projected features from the generated image.
            feat_k: List of projected features from the source image.

        Returns:
            The computed InfoNCE loss averaged across all feature layers.
        """
        loss = 0.0
        for q, k in zip(feat_q, feat_k):
            # q, k shape: (B, num_patches, embed_dim)
            B, num_patches, C = q.shape

            # bmm(q, k.transpose(1, 2)) -> (B, num_patches, num_patches)
            # Diagonal represents positive pairs (same spatial location), off-diagonal are negatives
            logits = torch.bmm(q, k.transpose(1, 2))
            logits = logits / self.tau

            # Labels: for each patch i in q, the positive is patch i in k.
            labels = (
                torch.arange(num_patches, dtype=torch.long, device=q.device)
                .unsqueeze(0)
                .expand(B, -1)
            )

            # Flatten for CE loss
            logits = logits.view(-1, num_patches)
            labels = labels.reshape(-1)

            loss += self.cross_entropy_loss(logits, labels)

        return loss / len(feat_q)
