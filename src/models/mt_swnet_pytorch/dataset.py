# -- coding: utf-8 --
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent paths so we can import the existing data_loader
_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_stnet_dir = os.path.join(os.path.dirname(_this_dir), 'mt_stnet')
if _mt_stnet_dir not in sys.path:
    sys.path.insert(0, _mt_stnet_dir)

from data.data_loader import loadData


class TrafficDataset(Dataset):
    """PyTorch Dataset wrapping the existing numpy data_loader output.

    Each sample returns a dict with:
        X:             [P, N, 1]     current traffic (z-score normalized)
        X_all:         [P+Q, N, 1]   last-week same window (z-score normalized)
        day_of_week:   [P+Q, N]      int (0-6)
        minute_of_day: [P+Q, N]      int (0-287)
        label:         [N, Q]        raw traffic flow
    """

    def __init__(self, X, DoW, M, L, XAll):
        """
        X:    [N_samples, P, N_sites]
        DoW:  [N_samples, P+Q, N_sites]
        M:    [N_samples, P+Q, N_sites]
        L:    [N_samples, N_sites, Q]
        XAll: [N_samples, P+Q, N_sites]
        """
        self.X = X.astype(np.float32)
        self.DoW = DoW.astype(np.int64)
        self.M = M.astype(np.int64)
        self.L = L.astype(np.float32)
        self.XAll = XAll.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.X[idx])[..., None],         # [P, N, 1]
            'X_all': torch.from_numpy(self.XAll[idx])[..., None],  # [P+Q, N, 1]
            'day_of_week': torch.from_numpy(self.DoW[idx]),         # [P+Q, N]
            'minute_of_day': torch.from_numpy(self.M[idx]),         # [P+Q, N]
            'label': torch.from_numpy(self.L[idx]),                 # [N, Q]
        }


def load_graph_data(hp):
    """Load static graph data (adjacency, shortest path, distance, degree).

    These are loaded once and shared across all samples.

    Returns:
        adj_np:   [N, N]        numpy binary adjacency
        sp:       [N*N, 15]     numpy int shortest path IDs
        dis:      [N, N]        numpy float distance
        in_deg:   [1, N]        numpy int in-degree
        out_deg:  [1, N]        numpy int out-degree
        position: [1, N]        numpy int [0, 1, ..., N-1]
    """
    import pandas as pd

    N = hp.site_num

    # Adjacency matrix
    adj_df = pd.read_csv(hp.file_adj)
    adj_np = np.zeros((N, N), dtype=np.float32)
    for _, row in adj_df.iterrows():
        src = int(row['src_FID']) - 1  # 1-based to 0-based
        nbr = int(row['nbr_FID']) - 1
        adj_np[src, nbr] = 1.0

    # Shortest path: [N*N, 15]
    sp = pd.read_csv(hp.file_sp, header=None).values.astype(np.int64)

    # Distance: [N, N]
    dis = pd.read_csv(hp.file_dis, header=None).values.astype(np.float32)

    # Degree: [1, N] — CSV has header row ('index', 'in_deg'/'out_deg'), take column 1
    in_deg = pd.read_csv(hp.file_in_deg).values[:, 1].astype(np.int64).reshape(1, N)
    out_deg = pd.read_csv(hp.file_out_deg).values[:, 1].astype(np.int64).reshape(1, N)

    # Position: [1, N]
    position = np.arange(N, dtype=np.int64).reshape(1, N)

    return adj_np, sp, dis, in_deg, out_deg, position


def build_datasets(hp):
    """Load data and build train/val/test datasets.

    Returns:
        train_ds, val_ds, test_ds: TrafficDataset instances
        mean, std: float scalars for inverse transform
        graph_data: tuple (adj_np, sp, dis, in_deg, out_deg, position)
    """
    data = loadData(hp)
    (trainX, trainDoW, _, _, trainM, trainL, trainXAll,
     valX, valDoW, _, _, valM, valL, valXAll,
     testX, testDoW, _, _, testM, testL, testXAll,
     mean, std) = data

    train_ds = TrafficDataset(trainX, trainDoW, trainM, trainL, trainXAll)
    val_ds = TrafficDataset(valX, valDoW, valM, valL, valXAll)
    test_ds = TrafficDataset(testX, testDoW, testM, testL, testXAll)

    graph_data = load_graph_data(hp)

    return train_ds, val_ds, test_ds, float(mean), float(std), graph_data
