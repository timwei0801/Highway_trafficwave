# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FC


class SpatialAttention(nn.Module):
    """Multi-head spatial attention with physical information injection.

    Attention is computed over the N (node) dimension per time step.
    Physical info: distance matrix bias, shortest-path learnable bias,
    degree embeddings added to input.

    Input:  X   [B, T, N, D]
            STE [B, T, N, D]
    Output:     [B, T, N, D]
    """

    def __init__(self, K, d, D, N, sp_max_len, bn=False, dropout=0.0,
                 is_physical=True):
        super().__init__()
        self.K = K
        self.d = d
        self.D = D
        self.is_physical = is_physical

        self.fc_q = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_k = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_v = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_out = FC(D, [D, D], [F.relu, None], bn=bn)

        if is_physical:
            # Learnable projection for shortest-path embeddings
            # sp shape after reshape: [1, N, N, sp_max_len, D]
            # w_n projects D -> 1 for each (N, sp_max_len) entry
            self.w_n = nn.Parameter(torch.empty(1, 1, N, D, 1))
            nn.init.xavier_uniform_(self.w_n.view(1, N, -1))

    def forward(self, X, STE, dis=None, sp=None, in_deg=None, out_deg=None):
        """
        X:       [B, T, N, D]
        STE:     [B, T, N, D]
        dis:     [N, N]  distance matrix (float tensor on device)
        sp:      [N*N, max_len, D]  shortest-path embeddings
        in_deg:  [B, T, N, D] or broadcastable
        out_deg: [B, T, N, D] or broadcastable
        """
        B, T, N, D = X.shape
        K, d = self.K, self.d

        # Phase 1: input fusion
        X = X + STE
        if self.is_physical and in_deg is not None:
            X = X + in_deg + out_deg

        # Phase 2: Q/K/V projection
        query = self.fc_q(X)     # [B, T, N, D]
        key = self.fc_k(X)       # [B, T, N, D]
        value = self.fc_v(X)     # [B, T, N, D]

        # Phase 3: multi-head split — reshape then merge heads into batch
        # [B, T, N, D] -> [B, T, N, K, d] -> [B, K, T, N, d] -> [B*K, T, N, d]
        query = query.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)
        key = key.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)
        value = value.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)

        # Phase 4: attention scores  [B*K, T, N, N]
        scores = torch.matmul(query, key.transpose(-1, -2)) / (d ** 0.5)

        # Phase 5: physical info injection
        if self.is_physical and dis is not None:
            # Distance bias: [N, N] -> [1, 1, N, N]
            dis_bias = dis.unsqueeze(0).unsqueeze(0)
            scores = scores + dis_bias

        if self.is_physical and sp is not None:
            # sp: [N*N, max_len, D] -> [N, N, max_len, D] -> [1, N, N, max_len, D]
            sp_4d = sp.reshape(N, N, -1, D).unsqueeze(0)
            # w_n: [1, 1, N, D, 1] — matmul on last two dims
            # [1, N, N, max_len, D] @ [1, 1, N, D, 1] -> [1, N, N, max_len, 1]
            sp_proj = torch.matmul(sp_4d, self.w_n)
            # [1, N, N, max_len, 1] -> [1, N, N, max_len] -> [1, N, N] -> [1, 1, N, N]
            sp_bias = sp_proj.squeeze(-1).mean(dim=-1).unsqueeze(1)
            scores = scores + sp_bias

        # Phase 6: softmax + weighted sum
        attn = F.softmax(scores, dim=-1)              # [B*K, T, N, N]
        out = torch.matmul(attn, value)               # [B*K, T, N, d]

        # Reverse multi-head: [B*K, T, N, d] -> [B, K, T, N, d] -> [B, T, N, K, d] -> [B, T, N, D]
        out = out.reshape(B, K, T, N, d).permute(0, 2, 3, 1, 4).reshape(B, T, N, D)

        # Feed-forward
        out = self.fc_out(out)                        # [B, T, N, D]
        return out


class TemporalAttention(nn.Module):
    """Multi-head temporal attention with optional causal masking.

    Attention is computed over the T (time) dimension per node.

    Input:  X   [B, T, N, D]
            STE [B, T, N, D]
    Output:     [B, T, N, D]
    """

    def __init__(self, K, d, D, bn=False, dropout=0.0, is_physical=True):
        super().__init__()
        self.K = K
        self.d = d
        self.D = D
        self.is_physical = is_physical

        self.fc_q = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_k = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_v = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_out = FC(D, [D, D], [F.relu, None], bn=bn)

    def forward(self, X, STE, mask=False, in_deg=None, out_deg=None):
        """
        X:       [B, T, N, D]
        STE:     [B, T, N, D]
        mask:    bool — if True, apply causal (lower-triangular) mask
        in_deg:  [B, T, N, D] or broadcastable
        out_deg: [B, T, N, D] or broadcastable
        """
        B, T, N, D = X.shape
        K, d = self.K, self.d

        # Phase 1: input fusion
        X = X + STE
        if self.is_physical and in_deg is not None:
            X = X + in_deg + out_deg

        # Phase 2: Q/K/V projection
        query = self.fc_q(X)     # [B, T, N, D]
        key = self.fc_k(X)       # [B, T, N, D]
        value = self.fc_v(X)     # [B, T, N, D]

        # Phase 3: multi-head split -> [B*K, T, N, d]
        query = query.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)
        key = key.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)
        value = value.reshape(B, T, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, T, N, d)

        # Phase 4: transpose for temporal attention
        # [B*K, T, N, d] -> [B*K, N, T, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # Phase 5: attention scores [B*K, N, T, T]
        scores = torch.matmul(query, key.transpose(-1, -2)) / (d ** 0.5)

        # Phase 6: causal mask
        if mask:
            # Lower-triangular: position i can attend to positions <= i
            causal = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
            # [T, T] -> [1, 1, T, T] broadcast to [B*K, N, T, T]
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Phase 7: softmax + weighted sum
        attn = F.softmax(scores, dim=-1)              # [B*K, N, T, T]
        out = torch.matmul(attn, value)               # [B*K, N, T, d]

        # Transpose back: [B*K, N, T, d] -> [B*K, T, N, d]
        out = out.permute(0, 2, 1, 3)

        # Reverse multi-head: [B*K, T, N, d] -> [B, K, T, N, d] -> [B, T, N, D]
        out = out.reshape(B, K, T, N, d).permute(0, 2, 3, 1, 4).reshape(B, T, N, D)

        # Feed-forward
        out = self.fc_out(out)                        # [B, T, N, D]
        return out


class BridgeAttention(nn.Module):
    """Cross-attention bridge for autoregressive decoding.

    Query from decoder STE, Key/Value from encoder output.
    Weight sharing across autoregressive steps is automatic in PyTorch —
    the same nn.Module instance is called repeatedly.

    Input:  X     [B, P, N, D]  encoder output (grows each step)
            STE_P [B, P, N, D]  encoder STE (grows each step)
            STE_Q [B, Q, N, D]  decoder STE query (typically Q=1)
    Output:       [B, Q, N, D]
    """

    def __init__(self, K, d, D, bn=False, dropout=0.0):
        super().__init__()
        self.K = K
        self.d = d
        self.D = D

        self.fc_q = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_k = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_v = FC(D, D, F.relu, bn=bn, dropout=dropout)
        self.fc_out = FC(D, [D, D], [F.relu, None], bn=bn)

    def forward(self, X, STE_P, STE_Q):
        """
        X:     [B, P, N, D]  encoder values
        STE_P: [B, P, N, D]  encoder spatio-temporal embedding
        STE_Q: [B, Q, N, D]  decoder query STE
        """
        B = X.shape[0]
        N = X.shape[2]
        P = X.shape[1]
        Q = STE_Q.shape[1]
        K, d, D = self.K, self.d, self.D

        # Phase 1: Q/K/V projection
        query = self.fc_q(STE_Q)    # [B, Q, N, D]
        key = self.fc_k(STE_P)      # [B, P, N, D]
        value = self.fc_v(X)        # [B, P, N, D]

        # Phase 2: multi-head split -> merge heads into batch
        query = query.reshape(B, Q, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, Q, N, d)
        key = key.reshape(B, P, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, P, N, d)
        value = value.reshape(B, P, N, K, d).permute(0, 3, 1, 2, 4).reshape(B * K, P, N, d)

        # Phase 3: transpose for temporal cross-attention
        # query: [B*K, Q, N, d] -> [B*K, N, Q, d]
        # key:   [B*K, P, N, d] -> [B*K, N, d, P]  (transposed)
        # value: [B*K, P, N, d] -> [B*K, N, P, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        # Phase 4: cross-attention scores [B*K, N, Q, P]
        scores = torch.matmul(query, key) / (d ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Phase 5: weighted sum [B*K, N, Q, d]
        out = torch.matmul(attn, value)

        # Transpose back: [B*K, N, Q, d] -> [B*K, Q, N, d]
        out = out.permute(0, 2, 1, 3)

        # Reverse multi-head: [B*K, Q, N, d] -> [B, K, Q, N, d] -> [B, Q, N, D]
        out = out.reshape(B, K, Q, N, d).permute(0, 2, 3, 1, 4).reshape(B, Q, N, D)

        # Feed-forward
        out = self.fc_out(out)      # [B, Q, N, D]
        return out
