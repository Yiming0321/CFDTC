#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM regression training script – every hyper-parameter exposed via argparse.
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime


def train(data: str,
          model_dir: str,
          feat_cols: list[str],
          target_col: str,
          max_depth: int,
          num_leaves: int,
          learning_rate: float,
          bagging_fraction: float,
          feature_fraction: float,
          n_estimators: int,
          nfold: int,
          seed: int) -> None:
    """
    Train a LightGBM regressor with the supplied hyper-parameters and K-fold CV.
    """

    os.makedirs(model_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")

    # 1. load data
    df = pd.read_excel(data)
    X = df[feat_cols].copy()
    y = df[target_col].copy()

    # 2. create LightGBM dataset
    train_data = lgb.Dataset(X, label=y)

    # 3. parameter dict
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "verbose": -1,
    }

    # 4. cross-validation
    print("Running CV …")
    cv_res = lgb.cv(
        params,
        train_data,
        num_boost_round=n_estimators,
        nfold=nfold,
        stratified=False,
        shuffle=True,
        metrics="rmse",
        seed=seed,
        return_cvbooster=True,
    )

    best_iter = len(cv_res["valid rmse-mean"])
    best_rmse = cv_res["valid rmse-mean"][-1]
    print(f"Best iteration = {best_iter},  Mean RMSE = {best_rmse:.4f}")

    # 5. save CV curve
    cv_file = os.path.join(model_dir, f"cv_curve_{time_str}.xlsx")
    pd.DataFrame(cv_res).to_excel(cv_file, index=False)
    print(f"CV curve saved -> {cv_file}")

    # 6. retrain on full data
    final_model = lgb.train(params, train_data, num_boost_round=best_iter)

    # 7. persist model
    model_file = os.path.join(model_dir, f"lightgbm_regression_{time_str}.pkl")
    joblib.dump(final_model, model_file)
    print(f"Model saved -> {model_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM regression – all hyper-parameters injectable"
    )

    # data / I/O
    parser.add_argument("--data", "-t", required=True,
                        help="Path to training Excel file")
    parser.add_argument("--model_dir", "-o", default="./model",
                        help="Output folder for models and logs (default: ./model)")
    parser.add_argument("--feat", "-f", nargs="+",
                        default=["Right_final", "Left_final", "Difference", "room_temperature"],
                        help="Feature column names (space-separated)")
    parser.add_argument("--target", "-g", default="P1(uW)",
                        help="Target column name")

    # LightGBM core hyper-parameters
    parser.add_argument("--max_depth", type=int, default=50, help="max_depth (default: 50)")
    parser.add_argument("--num_leaves", type=int, default=300, help="num_leaves (default: 300)")
    parser.add_argument("--learning_rate", type=float, default=0.1,  help="learning_rate (default: 0.1)")
    parser.add_argument("--bagging_fraction", type=float, default=0.7, help="bagging_fraction (default: 0.7)")
    parser.add_argument("--feature_fraction", type=float, default=0.95,  help="feature_fraction (default: 0.95)")
    parser.add_argument("--n_estimators", type=int, default=2000,  help="num_boost_round (default: 2000)")
    parser.add_argument("--nfold", type=int, default=5,  help="CV nfold (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data,
        model_dir=args.model_dir,
        feat_cols=args.feat,
        target_col=args.target,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        bagging_fraction=args.bagging_fraction,
        feature_fraction=args.feature_fraction,
        n_estimators=args.n_estimators,
        nfold=args.nfold,
        seed=args.seed,
    )