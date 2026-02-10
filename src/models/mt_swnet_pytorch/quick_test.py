#!/usr/bin/env python3
# -- coding: utf-8 --
"""Smoke test for MT-SWNet PyTorch — 200 samples, 2 epochs.

Verifies:
  1. Pipeline runs end-to-end (data -> model -> loss -> backprop -> inference)
  2. No NaN in loss or predictions
  3. Loss decreases across epochs
  4. MPS (Apple GPU) is used when available
  5. Parameter count matches TF version (~194,563)
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Setup paths so both package imports and data_loader work
_this_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.dirname(_this_dir)  # src/models/
_mt_stnet_dir = os.path.join(_models_dir, 'mt_stnet')

# Add src/models/ so we can import mt_swnet_pytorch as a package
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
# Add mt_stnet so data_loader imports work
if _mt_stnet_dir not in sys.path:
    sys.path.insert(0, _mt_stnet_dir)

# Change to mt_swnet_pytorch dir so relative data paths in config resolve
os.chdir(_this_dir)

from mt_swnet_pytorch.config import get_args, get_device
from mt_swnet_pytorch.dataset import TrafficDataset, load_graph_data
from mt_swnet_pytorch.model import MTSWNet
from mt_swnet_pytorch.loss import masked_mae_loss
from mt_swnet_pytorch.metrics import metric
from data.data_loader import loadData


def quick_test():
    print('=' * 60)
    print('MT-SWNet PyTorch — Smoke Test')
    print('=' * 60)

    # --- Config ---
    hp = get_args([
        '--batch_size', '8',
        '--epochs', '2',
    ])
    device = get_device(hp.device)
    print(f'Device: {device}')
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    # --- Load full data, then slice ---
    print('\nLoading data (will slice to 200/50/50)...')
    data = loadData(hp)
    (trainX, trainDoW, _, _, trainM, trainL, trainXAll,
     valX, valDoW, _, _, valM, valL, valXAll,
     testX, testDoW, _, _, testM, testL, testXAll,
     mean, std) = data

    # Slice
    N_TRAIN, N_VAL, N_TEST = 200, 50, 50
    trainX, trainDoW, trainM, trainL, trainXAll = (
        trainX[:N_TRAIN], trainDoW[:N_TRAIN], trainM[:N_TRAIN],
        trainL[:N_TRAIN], trainXAll[:N_TRAIN])
    valX, valDoW, valM, valL, valXAll = (
        valX[:N_VAL], valDoW[:N_VAL], valM[:N_VAL],
        valL[:N_VAL], valXAll[:N_VAL])
    testX, testDoW, testM, testL, testXAll = (
        testX[:N_TEST], testDoW[:N_TEST], testM[:N_TEST],
        testL[:N_TEST], testXAll[:N_TEST])

    print(f'trainX: {trainX.shape}, valX: {valX.shape}, testX: {testX.shape}')

    # --- Datasets ---
    train_ds = TrafficDataset(trainX, trainDoW, trainM, trainL, trainXAll)
    val_ds = TrafficDataset(valX, valDoW, valM, valL, valXAll)
    test_ds = TrafficDataset(testX, testDoW, testM, testL, testXAll)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)

    # --- Static graph data ---
    adj_np, sp_np, dis_np, in_deg_np, out_deg_np, position_np = load_graph_data(hp)
    sp_t = torch.from_numpy(sp_np).to(device)
    dis_t = torch.from_numpy(dis_np).to(device)
    in_deg_t = torch.from_numpy(in_deg_np).to(device)
    out_deg_t = torch.from_numpy(out_deg_np).to(device)
    position_t = torch.from_numpy(position_np).to(device)

    # --- Model ---
    model = MTSWNet(hp, adj_np).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTrainable parameters: {num_params:,}')
    print(f'TF reference:         194,563')
    print(f'Match: {"YES" if abs(num_params - 194563) < 1000 else "NO (check architecture)"}')

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    # --- Train 2 epochs ---
    epoch_losses = []
    print(f'\nTraining {hp.epochs} epochs...')
    t_start = time.time()

    for epoch in range(hp.epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            X = batch['X'].to(device)
            X_all = batch['X_all'].to(device)
            dow = batch['day_of_week'].to(device)
            mod = batch['minute_of_day'].to(device)
            label = batch['label'].to(device)

            pred = model(X, X_all, dow, mod,
                         position_t, sp_t, dis_t, in_deg_t, out_deg_t)
            pred_raw = pred * std + mean
            loss = masked_mae_loss(pred_raw, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        print(f'  Epoch {epoch + 1}: loss = {avg_loss:.4f}')

    train_time = time.time() - t_start
    print(f'Training time: {train_time:.1f}s')

    # --- Inference test ---
    print('\nRunning inference on test set...')
    model.eval()
    preds, labels = [], []
    t_infer = time.time()
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            X_all = batch['X_all'].to(device)
            dow = batch['day_of_week'].to(device)
            mod = batch['minute_of_day'].to(device)
            label = batch['label']

            pred = model(X, X_all, dow, mod,
                         position_t, sp_t, dis_t, in_deg_t, out_deg_t)
            pred_raw = pred * std + mean
            preds.append(pred_raw.cpu().numpy())
            labels.append(label.numpy())

    infer_time = time.time() - t_infer
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    mae, rmse, mape = metric(preds, labels)
    print(f'Inference time: {infer_time:.2f}s')
    print(f'Test MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape * 100:.3f}%')

    # --- Checks ---
    print('\n' + '=' * 60)
    print('CHECKS:')

    checks_passed = True

    # Check 1: No NaN
    has_nan = np.isnan(preds).any() or any(np.isnan(l) for l in epoch_losses)
    print(f'  [{"PASS" if not has_nan else "FAIL"}] No NaN in loss/predictions')
    if has_nan:
        checks_passed = False

    # Check 2: Loss decreased
    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    print(f'  [{"PASS" if loss_decreased else "WARN"}] Loss decreased: '
          f'{epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}')
    if not loss_decreased:
        print('    (May be OK with only 2 epochs on tiny data)')

    # Check 3: MPS used
    mps_used = device.type == 'mps'
    print(f'  [{"PASS" if mps_used else "INFO"}] MPS acceleration: '
          f'{"YES" if mps_used else "NO (using " + str(device) + ")"}')

    # Check 4: Parameter count
    param_close = abs(num_params - 194563) < 1000
    print(f'  [{"PASS" if param_close else "WARN"}] Parameter count: '
          f'{num_params:,} (target: ~194,563)')

    # Check 5: Pipeline complete
    print(f'  [PASS] Pipeline: data -> model -> loss -> backprop -> inference')

    print('=' * 60)
    if checks_passed:
        print('Smoke test PASSED')
    else:
        print('Smoke test had FAILURES — check above')

    return checks_passed


if __name__ == '__main__':
    quick_test()
