# -- coding: utf-8 --
"""
Quick smoke test: verify MT-SWNet training pipeline works end-to-end.
Uses only a tiny slice of data (200 samples) and runs 2 epochs.
"""
import sys
import os
import math
import datetime
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import run_train first — it calls tf.reset_default_graph() and tf.set_random_seed()
# at module level, so we must let it initialize the graph before we do anything else.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from run_train import Model
from utils.inits import *
from config.config import parameter
from data.data_loader import seq2instance

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


def load_small_data(args, max_train=200, max_val=50, max_test=50):
    """Load only a tiny subset of data for quick testing."""
    df = pd.read_csv(args.file_train_f)
    Traffic = df.values
    total_samples = df.shape[0] // args.site_num

    train_low = 60 // args.granularity * 24 * 7  # 2016
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # Only take small slices
    train_high = min(val_low, train_low + max_train + args.input_length + args.output_length)
    val_high = min(test_low, val_low + max_val + args.input_length + args.output_length)
    test_high = min(total_samples, test_low + max_test + args.input_length + args.output_length)

    print(f"Loading tiny subset: train [{train_low}:{train_high}], val [{val_low}:{val_high}], test [{test_low}:{test_high}]")

    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=train_low, high_index=train_high,
        granularity=args.granularity, sites=args.site_num, type='train')

    valX, valDoW, valD, valH, valM, valL, valXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=val_low, high_index=val_high,
        granularity=args.granularity, sites=args.site_num, type='validation')

    testX, testDoW, testD, testH, testM, testL, testXAll = seq2instance(
        Traffic, args.input_length, args.output_length,
        low_index=test_low, high_index=test_high,
        granularity=args.granularity, sites=args.site_num, type='test')

    mean, std = np.mean(trainX), np.std(trainX)
    trainX, trainXAll = (trainX - mean) / std, (trainXAll - mean) / std
    valX, valXAll = (valX - mean) / std, (valXAll - mean) / std
    testX, testXAll = (testX - mean) / std, (testXAll - mean) / std

    return (trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
            valX, valDoW, valD, valH, valM, valL, valXAll,
            testX, testDoW, testD, testH, testM, testL, testXAll,
            mean, std)


def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    para = parameter(argparse.ArgumentParser())
    para = para.get_para()
    para.is_training = True
    para.epochs = 2
    para.batch_size = 8

    print("=" * 60)
    print("MT-SWNet Quick Smoke Test")
    print("=" * 60)
    print(f"Epochs: {para.epochs}, Batch size: {para.batch_size}")
    print(f"Sites: {para.site_num}, Input: {para.input_length}, Output: {para.output_length}")

    # Load tiny data
    print("\n[1/4] Loading small dataset...")
    data = load_small_data(para, max_train=200, max_val=50, max_test=50)
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = data[:7]
    valX, valDoW, valD, valH, valM, valL, valXAll = data[7:14]
    testX, testDoW, testD, testH, testM, testL, testXAll = data[14:21]
    mean, std = data[21], data[22]

    print(f"  trainX: {trainX.shape}, valX: {valX.shape}, testX: {testX.shape}")

    # Build model
    print("\n[2/4] Building model...")
    pre_model = Model(para, mean=mean, std=std)
    pre_model.initialize_session(session)
    print("  Model built successfully")

    # Train
    print("\n[3/4] Training (2 epochs)...")
    start = datetime.datetime.now()
    pre_model.run_epoch(
        trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll,
        valX, valDoW, valD, valH, valM, valL, valXAll,
        testX, testDoW, testD, testH, testM, testL, testXAll)
    elapsed = (datetime.datetime.now() - start).total_seconds()
    print(f"  Training completed in {elapsed:.1f}s")

    # Test inference
    print("\n[4/4] Running inference on test set...")
    mae = pre_model.evaluate(testX, testDoW, testD, testH, testM, testL, testXAll)
    print(f"  Test MAE: {mae:.3f}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED - Pipeline is functional")
    print(f"  Data loading: OK")
    print(f"  Model build:  OK")
    print(f"  Training:     OK")
    print(f"  Inference:    OK (MAE={mae:.3f})")
    print("=" * 60)

    session.close()


if __name__ == '__main__':
    main()
