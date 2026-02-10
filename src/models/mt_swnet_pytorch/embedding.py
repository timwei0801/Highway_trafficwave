# -- coding: utf-8 --
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FC


class TokenEmbedding(nn.Module):
    """Lookup table embedding with optional zero-padding for index 0."""

    def __init__(self, vocab_size, embed_dim, zero_pad=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.trunc_normal_(self.embed.weight, mean=0, std=1)
        self.zero_pad = zero_pad
        if zero_pad:
            with torch.no_grad():
                self.embed.weight[0].fill_(0)

    def forward(self, x):
        out = self.embed(x)
        if self.zero_pad and self.training:
            with torch.no_grad():
                self.embed.weight.data[0].fill_(0)
        return out


class STEmbedding(nn.Module):
    """Spatio-Temporal Embedding.

    SE (spatial):  position_embed -> FC([D,D]) -> [1, 1, N, D]
    TE (temporal): concat(DoW_emb, M_emb) [B, P+Q, N, 2D] -> FC([D,D]) -> [B, P+Q, N, D]
    Output: SE + TE (broadcast) -> [B, P+Q, N, D]
    """

    def __init__(self, D, bn=False):
        super().__init__()
        self.fc_se = FC(D, [D, D], [F.relu, None], bn=bn)
        self.fc_te = FC(D * 2, [D, D], [F.relu, None], bn=bn)

    def forward(self, SE, dow_emb, mod_emb):
        """
        SE:      [1, 1, N, D] position embedding
        dow_emb: [B, P+Q, N, D] day-of-week embedding
        mod_emb: [B, P+Q, N, D] minute-of-day embedding
        """
        SE = self.fc_se(SE)                              # [1, 1, N, D]
        TE = torch.cat([dow_emb, mod_emb], dim=-1)       # [B, P+Q, N, 2D]
        TE = self.fc_te(TE)                              # [B, P+Q, N, D]
        return SE + TE                                    # [B, P+Q, N, D]
