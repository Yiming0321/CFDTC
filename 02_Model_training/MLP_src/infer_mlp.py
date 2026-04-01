#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference entry – supports file path, DataFrame, or numpy array as input
"""
import argparse
from train_mlp import inference


def get_args():
    parser = argparse.ArgumentParser(description="MLP Inference")
    parser.add_argument("--model", required=True, help="path to MLP model file (.pth)")
    parser.add_argument("--data", required=True, help="path to input file (.xlsx or .csv) or directly pd.DataFrame/np.ndarray")
    parser.add_argument("--output_dir", default="./output", help="output directory for results")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prediction_results = inference(
        model_path=args.model,
        data=args.data,
        output_dir=args.output_dir
    )