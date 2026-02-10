# -- coding: utf-8 --
"""Training script for MT-SWNet PyTorch."""
import os
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import get_args, get_device
from .dataset import build_datasets
from .model import MTSWNet
from .loss import masked_mae_loss
from .metrics import metric


def train(argv=None):
    hp = get_args(argv)
    device = get_device(hp.device)
    print(f'Device: {device}')

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    # --- Data ---
    print('Loading data...')
    train_ds, val_ds, test_ds, mean, std, graph_data = build_datasets(hp)
    adj_np, sp_np, dis_np, in_deg_np, out_deg_np, position_np = graph_data

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size,
                              shuffle=True, num_workers=hp.num_workers,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size,
                            shuffle=False, num_workers=hp.num_workers)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size,
                             shuffle=False, num_workers=hp.num_workers)

    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

    # --- Static tensors (loaded once, moved to device) ---
    sp_t = torch.from_numpy(sp_np).to(device)
    dis_t = torch.from_numpy(dis_np).to(device)
    in_deg_t = torch.from_numpy(in_deg_np).to(device)
    out_deg_t = torch.from_numpy(out_deg_np).to(device)
    position_t = torch.from_numpy(position_np).to(device)

    # --- Model ---
    model = MTSWNet(hp, adj_np).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate,
                                  weight_decay=hp.weight_decay)

    # StepLR: decay every decay_epoch epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=hp.decay_epoch, gamma=hp.lr_decay)

    # --- Training loop ---
    os.makedirs(hp.checkpoint_dir, exist_ok=True)
    best_val_mae = float('inf')
    best_path = os.path.join(hp.checkpoint_dir, 'best_model.pt')

    for epoch in range(hp.epochs):
        # -- Train --
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in train_loader:
            X = batch['X'].to(device)
            X_all = batch['X_all'].to(device)
            dow = batch['day_of_week'].to(device)
            mod = batch['minute_of_day'].to(device)
            label = batch['label'].to(device)

            pred = model(X, X_all, dow, mod,
                         position_t, sp_t, dis_t, in_deg_t, out_deg_t)

            # De-normalize prediction for loss
            pred_raw = pred * std + mean
            loss = masked_mae_loss(pred_raw, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # Enforce min LR
        if current_lr < hp.min_lr:
            for pg in optimizer.param_groups:
                pg['lr'] = hp.min_lr
            current_lr = hp.min_lr

        train_time = time.time() - t0
        avg_loss = epoch_loss / max(num_batches, 1)

        # -- Validate --
        val_mae = evaluate(model, val_loader, device, mean, std,
                           position_t, sp_t, dis_t, in_deg_t, out_deg_t)

        print(f'Epoch {epoch + 1:3d}/{hp.epochs} | '
              f'Loss: {avg_loss:.4f} | Val MAE: {val_mae:.3f} | '
              f'LR: {current_lr:.2e} | Time: {train_time:.1f}s')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_path)
            print(f'  -> Best model saved (MAE: {best_val_mae:.3f})')

    # --- Final test ---
    print('\n--- Final Test ---')
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_results(model, test_loader, device, mean, std,
                 position_t, sp_t, dis_t, in_deg_t, out_deg_t,
                 hp.output_length)

    return model


@torch.no_grad()
def evaluate(model, loader, device, mean, std,
             position, sp, dis, in_deg, out_deg):
    """Run evaluation, return overall MAE."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        X = batch['X'].to(device)
        X_all = batch['X_all'].to(device)
        dow = batch['day_of_week'].to(device)
        mod = batch['minute_of_day'].to(device)
        label = batch['label']

        pred = model(X, X_all, dow, mod, position, sp, dis, in_deg, out_deg)
        pred_raw = pred * std + mean
        preds.append(pred_raw.cpu().numpy())
        labels.append(label.numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    mae, rmse, mape = metric(preds, labels)
    return mae


@torch.no_grad()
def test_results(model, loader, device, mean, std,
                 position, sp, dis, in_deg, out_deg, output_length):
    """Run full test with per-step metrics."""
    model.eval()
    preds, labels = [], []
    t0 = time.time()

    for batch in loader:
        X = batch['X'].to(device)
        X_all = batch['X_all'].to(device)
        dow = batch['day_of_week'].to(device)
        mod = batch['minute_of_day'].to(device)
        label = batch['label']

        pred = model(X, X_all, dow, mod, position, sp, dis, in_deg, out_deg)
        pred_raw = pred * std + mean
        preds.append(pred_raw.cpu().numpy())
        labels.append(label.numpy())

    infer_time = time.time() - t0
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(f'Inference time: {infer_time:.2f}s ({len(preds)} samples)')
    print(f'{"Step":>6}  {"MAE":>8}  {"RMSE":>8}  {"MAPE%":>8}')
    print('-' * 38)
    for i in range(output_length):
        mae, rmse, mape = metric(preds[:, :, i], labels[:, :, i])
        print(f'{i + 1:6d}  {mae:8.3f}  {rmse:8.3f}  {mape * 100:8.3f}')

    mae, rmse, mape = metric(preds, labels)
    print('-' * 38)
    print(f'{"Avg":>6}  {mae:8.3f}  {rmse:8.3f}  {mape * 100:8.3f}')

    return preds, labels


if __name__ == '__main__':
    train()
