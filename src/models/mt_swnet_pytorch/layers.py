# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()


def preprocess_adj(adj_np):
    """Add self-loops and normalize. Returns dense numpy array [N, N]."""
    adj_with_self = adj_np + np.eye(adj_np.shape[0])
    return normalize_adj(adj_with_self).astype(np.float32)


class FC(nn.Module):
    """Stack of linear layers with optional BN, activation, and dropout.

    Replaces TF's 1x1 Conv2d stack. nn.Linear operates on the last dim,
    which is equivalent to 1x1 convolution on channels-last tensors.

    Input:  [..., C_in]  (any leading dimensions)
    Output: [..., C_out]
    """

    def __init__(self, input_dim, units, activations, bn=False, dropout=0.0):
        super().__init__()
        if isinstance(units, int):
            units = [units]
            activations = [activations]

        self.blocks = nn.ModuleList()
        self.acts = []
        in_dim = input_dim
        for out_dim, act in zip(units, activations):
            block = nn.ModuleDict()
            block['linear'] = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(block['linear'].weight)
            nn.init.zeros_(block['linear'].bias)
            if bn and act is not None:
                block['bn'] = nn.BatchNorm1d(out_dim)
            if dropout > 0:
                block['dropout'] = nn.Dropout(dropout)
            self.blocks.append(block)
            self.acts.append(act)
            in_dim = out_dim

    def forward(self, x):
        for block, act in zip(self.blocks, self.acts):
            if 'dropout' in block:
                x = block['dropout'](x)
            x = block['linear'](x)
            if act is not None:
                if 'bn' in block:
                    shape = x.shape
                    x = block['bn'](x.reshape(-1, shape[-1])).reshape(shape)
                x = act(x)
        return x


class GraphConvolution(nn.Module):
    """Single GCN layer with residual connection.

    Uses dense matmul (N=62 is small enough).
    Input:  [batch*T, N, D]
    Output: [batch*T, N, D]  = relu(adj @ X @ W + bias) + X
    """

    def __init__(self, in_features, out_features, adj_np):
        super().__init__()
        adj_normalized = preprocess_adj(adj_np)
        self.register_buffer('adj', torch.FloatTensor(adj_normalized))
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: [batch*T, N, D]
        support = torch.matmul(x, self.weight)       # [batch*T, N, D_out]
        output = torch.matmul(self.adj, support)      # [batch*T, N, D_out]
        output = output + self.bias
        return F.relu(output) + x                     # residual
