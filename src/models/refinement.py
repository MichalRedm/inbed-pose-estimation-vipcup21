import torch
import torch.nn as nn
from typing import List, Tuple


class GCNLayer(nn.Module):
    """
    Simple GCN layer: A * X * W
    """

    adj_norm: torch.Tensor

    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Normalize adjacency matrix: D^-0.5 * A * D^-0.5
        adj = adj + torch.eye(adj.size(0))
        degree = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        self.register_buffer("adj_norm", d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt)

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, num_joints, in_features)
        # adj_norm shape: (num_joints, num_joints)
        support = torch.matmul(x, self.weight)  # (B, J, out_features)
        output = torch.matmul(self.adj_norm, support)
        return output


class PoseRefinementGCN(nn.Module):
    """
    Refines joint coordinates using a Graph Convolutional Network.
    Takes (B, 14, 2) coordinates and outputs refined (B, 14, 2).
    """

    def __init__(self, num_joints: int = 14, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_joints = num_joints

        # Define LSP skeleton adjacency
        # 0: R Ankle, 1: R Knee, 2: R Hip, 3: L Hip, 4: L Knee, 5: L Ankle,
        # 6: R Wrist, 7: R Elbow, 8: R Shoulder, 9: L Shoulder, 10: L Elbow, 11: L Wrist,
        # 12: Neck, 13: Head
        adj = torch.zeros(num_joints, num_joints)
        edges: List[Tuple[int, int]] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),  # Legs
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),  # Arms
            (2, 8),
            (3, 9),  # Torso
            (8, 12),
            (9, 12),  # Neck base
            (12, 13),  # Head
        ]
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1

        self.gcn1 = GCNLayer(2, hidden_dim, adj)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, adj)
        self.gcn3 = GCNLayer(hidden_dim, 2, adj)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 14, 2) initial coordinates from soft-argmax
        residual = x
        x = self.relu(self.gcn1(x))
        x = self.relu(self.gcn2(x))
        x = self.gcn3(x)
        return residual + x
