import argparse
import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.ATIO import DecAlignTrainer
from utils import assign_gpu, setup_seed
from models.model import DecAlign

import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('DecAlign')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('DecAlign')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DMD_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="", data_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode='', is_distill=False
):

    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else:  # use default config file
        config_file = Path(__file__).parent / "config" / "dec_config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    # Get config as SimpleNamespace (supports dot notation access)
    args = get_config_regression(model_name, dataset_name, config_file, data_dir=data_dir if data_dir else None)

    # Set additional attributes
    args.is_distill = False
    args.mode = mode  # train or test
    args.model_save_path = str(Path(model_save_dir) / f"{args.model_name}-{args.dataset_name}.pth")
    args.device = assign_gpu(gpu_ids)
    args.feature_T = feature_T
    args.feature_A = feature_A
    args.feature_V = feature_V

    # Update with additional config if provided
    if config:
        for key, value in config.items():
            setattr(args, key, value)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []

    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args.cur_seed = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    # Save results to CSV file
    criterions = list(model_results[0].keys())
    csv_file = res_save_dir / f"{dataset_name}.csv"
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    res = [model_name]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)
    # Build DecAlign model
    model = DecAlign(args)
    model = model.cuda()

    trainer = DecAlignTrainer(args)

    if args.mode == 'test':
        model.load_state_dict(torch.load(args.model_save_path))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model.load_state_dict(torch.load(args.model_save_path))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='DecAlign: Multimodal Sentiment Analysis')
    parser.add_argument('--model', type=str, default='decalign', help='Model name')
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei', 'iemocap'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--model_save_dir', type=str, default='./pt', help='Directory to save models')
    parser.add_argument('--res_save_dir', type=str, default='./result', help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./log', help='Directory to save logs')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111], help='Random seeds')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPU IDs to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    return parser.parse_args()


def main():
    args = parse_args()

    DMD_run(
        model_name=args.model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        res_save_dir=args.res_save_dir,
        log_dir=args.log_dir,
        mode=args.mode,
        seeds=args.seeds,
        gpu_ids=args.gpu_ids,
        num_workers=args.num_workers,
        is_tune=False,
        is_distill=False
    )


if __name__ == '__main__':
    main()
