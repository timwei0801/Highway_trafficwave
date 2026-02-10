# -- coding: utf-8 --
import torch
import torch.nn as nn
from .attention import SpatialAttention, TemporalAttention
from .layers import GraphConvolution


def fusion_gate(x, y):
    """Element-wise gating: z = sigmoid(x*y); h = z*x + (1-z)*y."""
    z = torch.sigmoid(x * y)
    return z * x + (1 - z) * y


class STAttBlock(nn.Module):
    """Spatio-Temporal Attention Block.

    Flow: TemporalAttention -> GCN branch + SpatialAttention -> FusionGate -> Residual

    Input:  [B, T, N, D]
    Output: [B, T, N, D]
    """

    def __init__(self, K, d, D, N, sp_max_len, adj_np, bn=False, dropout=0.0,
                 is_physical=True):
        super().__init__()
        self.temporal_attn = TemporalAttention(K, d, D, bn=bn, dropout=dropout,
                                                is_physical=is_physical)
        self.spatial_attn = SpatialAttention(K, d, D, N, sp_max_len, bn=bn,
                                              dropout=dropout,
                                              is_physical=is_physical)
        self.gcn = GraphConvolution(D, D, adj_np)

    def forward(self, X, STE, mask=False, dis=None, sp=None,
                in_deg=None, out_deg=None):
        """
        X:       [B, T, N, D]
        STE:     [B, T, N, D]
        mask:    bool — causal mask for temporal attention
        dis:     [N, N]
        sp:      [N*N, max_len, D]
        in_deg:  broadcastable to [B, T, N, D]
        out_deg: broadcastable to [B, T, N, D]
        """
        B, T, N, D = X.shape

        # Step 1: Temporal Attention
        HT = self.temporal_attn(X, STE, mask=mask,
                                 in_deg=in_deg, out_deg=out_deg)

        # Step 2: GCN branch — reshape to [B*T, N, D] for graph conv
        GS = HT.reshape(B * T, N, D)
        GS = self.gcn(GS)
        GS = GS.reshape(B, T, N, D)

        # Step 3: Spatial Attention
        HS = self.spatial_attn(HT, STE, dis=dis, sp=sp,
                                in_deg=in_deg, out_deg=out_deg)

        # Step 4: Fusion Gate
        H = fusion_gate(HS, GS)

        # Step 5: Residual
        return X + H
