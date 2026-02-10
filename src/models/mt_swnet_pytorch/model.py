# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FC
from .embedding import TokenEmbedding, STEmbedding
from .attention import BridgeAttention
from .st_block import STAttBlock


class MTSWNet(nn.Module):
    """Multi-Task Shockwave Network (PyTorch).

    Architecture:
        Embeddings -> Input Projection -> Encoder (L blocks) ->
        Decoder (1 block) -> Autoregressive Bridge (Q steps) ->
        Multi-task Output Heads (3 groups)

    Output: [B, N, Q]  (predicted traffic flow)
    """

    def __init__(self, hp, adj_np):
        """
        hp:     argparse namespace with all hyperparameters
        adj_np: [N, N] numpy adjacency matrix (binary)
        """
        super().__init__()
        self.hp = hp
        N = hp.site_num         # 62
        D = hp.emb_size         # 64
        K = hp.num_heads        # 8
        d = D // K              # 8
        P = hp.input_length     # 12
        Q = hp.output_length    # 12
        S = hp.pre_len          # 6
        L = hp.num_blocks       # 1

        self.N = N
        self.D = D
        self.K = K
        self.d = d
        self.P = P
        self.Q = Q
        self.S = S

        # -- Embedding layers --
        self.pos_embed = TokenEmbedding(N, D, zero_pad=False)
        self.dow_embed = TokenEmbedding(7, D, zero_pad=False)
        self.mod_embed = TokenEmbedding(288, D, zero_pad=False)  # 24*60/5
        self.sp_embed = TokenEmbedding(hp.edge_num + 1, D, zero_pad=True)  # 145
        self.in_deg_embed = TokenEmbedding(5, D, zero_pad=False)
        self.out_deg_embed = TokenEmbedding(5, D, zero_pad=False)
        self.st_embed = STEmbedding(D)

        # -- Input projection: [B, P, N, 2] -> [B, P, N, D] --
        self.fc_input = FC(2, [D, D], [F.relu, None])
        # -- Last-week future projection: [B, Q, N, 1] -> [B, Q, N, D] --
        self.fc_xq = FC(1, [D, D], [F.relu, None])

        # -- Encoder blocks --
        sp_max_len = 15  # fixed from data
        self.encoder_blocks = nn.ModuleList([
            STAttBlock(K, d, D, N, sp_max_len, adj_np,
                       is_physical=hp.is_physical)
            for _ in range(L)
        ])

        # -- Decoder block (single) --
        self.decoder_block = STAttBlock(K, d, D, N, sp_max_len, adj_np,
                                         is_physical=hp.is_physical)

        # -- Bridge attention (shared across Q autoregressive steps) --
        self.bridge = BridgeAttention(K, d, D)

        # -- Multi-task output heads --
        # Split: [0:13], [13:26], [26:62]
        self.head1 = FC(D, [D, 1], [F.relu, None], dropout=0.1)
        self.head2 = FC(D, [D, 1], [F.relu, None], dropout=0.1)
        self.head3 = FC(D, [D, 1], [F.relu, None], dropout=0.1)

    def forward(self, X, X_all, day_of_week, minute_of_day,
                position, sp, dis, in_deg, out_deg):
        """
        X:             [B, P, N, 1]   current traffic
        X_all:         [B, P+Q, N, 1] last-week same window
        day_of_week:   [B, P+Q, N]    int indices (0-6)
        minute_of_day: [B, P+Q, N]    int indices (0-287)
        position:      [1, N]         int indices (0..N-1)
        sp:            [N*N, 15]      int indices (shortest path node IDs)
        dis:           [N, N]         float distance matrix
        in_deg:        [1, N]         int indices
        out_deg:       [1, N]         int indices
        """
        B = X.shape[0]
        P, Q, S, N, D = self.P, self.Q, self.S, self.N, self.D
        K, d = self.K, self.d

        # === EMBEDDINGS ===
        # Position embedding: [1, N] -> [1, N, D] -> [1, 1, N, D]
        pos_emb = self.pos_embed(position).unsqueeze(1)   # [1, 1, N, D]

        # Temporal embeddings: [B, P+Q, N] -> [B, P+Q, N, D]
        dow_emb = self.dow_embed(day_of_week)
        mod_emb = self.mod_embed(minute_of_day)

        # Spatio-Temporal Embedding
        STE = self.st_embed(pos_emb, dow_emb, mod_emb)    # [B, P+Q, N, D]
        STE_P = STE[:, :P]                                # [B, P, N, D]
        STE_Q = STE[:, P:]                                # [B, Q, N, D]

        # Shortest path embedding: [N*N, 15] -> [N*N, 15, D]
        sp_emb = self.sp_embed(sp)                         # [N*N, 15, D]

        # Degree embeddings: [1, N] -> [1, N, D] -> [1, 1, N, D]
        in_deg_emb = self.in_deg_embed(in_deg).unsqueeze(1)   # [1, 1, N, D]
        out_deg_emb = self.out_deg_embed(out_deg).unsqueeze(1) # [1, 1, N, D]

        # Distance as float tensor (already on device via caller)
        # dis: [N, N]

        # === INPUT PROJECTION ===
        # Encoder input: concat current + last-week historical
        X_enc = torch.cat([X, X_all[:, :P]], dim=-1)      # [B, P, N, 2]
        X_enc = self.fc_input(X_enc)                       # [B, P, N, D]

        # Decoder future input from last-week
        X_Q = self.fc_xq(X_all[:, P:])                    # [B, Q, N, D]

        # === ENCODER ===
        enc_out = X_enc
        for block in self.encoder_blocks:
            enc_out = block(enc_out, STE_P, mask=False,
                            dis=dis, sp=sp_emb,
                            in_deg=in_deg_emb, out_deg=out_deg_emb)
        X_E = enc_out                                      # [B, P, N, D]

        # === DECODER ===
        dec_input = torch.cat([X_E[:, -S:], X_Q], dim=1)  # [B, S+Q, N, D]
        dec_STE = STE[:, -(Q + S):]                        # [B, S+Q, N, D]
        dec_out = self.decoder_block(dec_input, dec_STE, mask=True,
                                      dis=dis, sp=sp_emb,
                                      in_deg=in_deg_emb, out_deg=out_deg_emb)
        # X_D = dec_out[:, -Q:]  # decoder output (not used by bridge)

        # === AUTOREGRESSIVE BRIDGE ===
        for i in range(Q):
            query_ste = X_E[:, -1:] + STE_Q[:, i:i + 1]   # [B, 1, N, D]
            bridge_out = self.bridge(X_E, X_E, query_ste)  # [B, 1, N, D]
            X_E = torch.cat([X_E, bridge_out + STE_Q[:, i:i + 1]], dim=1)

        X_out = X_E[:, -Q:]                               # [B, Q, N, D]

        # === MULTI-TASK OUTPUT HEADS ===
        pre1 = self.head1(X_out[:, :, :13])                # [B, Q, 13, 1]
        pre2 = self.head2(X_out[:, :, 13:26])              # [B, Q, 13, 1]
        pre3 = self.head3(X_out[:, :, 26:])                # [B, Q, 36, 1]

        pre = torch.cat([pre1, pre2, pre3], dim=2)         # [B, Q, N, 1]
        pre = pre.squeeze(-1)                              # [B, Q, N]
        pre = pre.permute(0, 2, 1)                         # [B, N, Q]

        return pre
