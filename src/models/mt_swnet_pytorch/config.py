# -- coding: utf-8 --
import argparse
import torch


def get_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(requested)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='MT-SWNet PyTorch')

    # Data paths
    parser.add_argument('--file_train_f', type=str,
                        default='../../../data/Taiwan/train.csv')
    parser.add_argument('--file_sp', type=str,
                        default='../../../data/Taiwan/sp.csv')
    parser.add_argument('--file_dis', type=str,
                        default='../../../data/Taiwan/dis.csv')
    parser.add_argument('--file_in_deg', type=str,
                        default='../../../data/Taiwan/in_deg.csv')
    parser.add_argument('--file_out_deg', type=str,
                        default='../../../data/Taiwan/out_deg.csv')
    parser.add_argument('--file_adj', type=str,
                        default='../../../data/Taiwan/adjacent_fully.csv')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--validate_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # Model architecture
    parser.add_argument('--site_num', type=int, default=62)
    parser.add_argument('--edge_num', type=int, default=144)
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--features', type=int, default=1)
    parser.add_argument('--input_length', type=int, default=12)
    parser.add_argument('--output_length', type=int, default=12)
    parser.add_argument('--pre_len', type=int, default=6)
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--is_physical', type=bool, default=True)
    parser.add_argument('--granularity', type=int, default=5)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--decay_epoch', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # PyTorch specific
    parser.add_argument('--device', type=str, default='auto',
                        help='cpu/mps/cuda/auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')

    return parser.parse_args(argv if argv is not None else [])
