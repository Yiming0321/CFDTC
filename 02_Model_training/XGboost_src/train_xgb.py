#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python train.py --data data.xlsx --out_dir exp1 --max_depth 8 --eta 0.05
"""
import os
import argparse
from typing import Union
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime


def load_data(data_input: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    load data from data path (str) and pd.Dataframe
    """
    if isinstance(data_input, pd.DataFrame):
        return data_input.copy()
    
    if isinstance(data_input, str):
        if not os.path.isfile(data_input):
            raise FileNotFoundError(f"File not found: {data_input}")
        return pd.read_excel(data_input) if data_input.lower().endswith(".xlsx") else pd.read_csv(data_input)
    
    raise TypeError(f"data have to be data path(str) or DataFrame, but it actually recieves {type(data_input)}")


def train(
    data: Union[str, pd.DataFrame],  
    model_dir: str = None,
    max_depth: int = 50,
    eta: float = 0.1,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    num_round: int = 100,
    early_stop: int = 20,
    seed: int = 42,
    feat_cols=["Right_final", "Left_final", "Difference", "room_temperature"],
    label_col="P1(uW)",
    log_target: bool = True,
):
    """Train an XGBoost model and save it; no meta file generated.

    log_target=True 时用 log10(P1) 作为训练目标：平方误差变成相对误差，
    低功率样本不再被高功率样本淹没（推理时需 10**pred 还原）。
    """
    os.makedirs(model_dir, exist_ok=True)

    # load data
    print("[Train] Loading data…")
    df = load_data(data)
    X = df[feat_cols].values
    y = np.log10(df[label_col].values) if log_target else df[label_col].values
    dtrain = xgb.DMatrix(X, label=y)
    print(f"[Train] 目标空间: {'log10(P1)' if log_target else 'P1 原始值'}")

    # params
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=max_depth,
        eta=eta,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        seed=seed,
    )

    # CV
    print("[Train] Starting cross-validation…")
    cv_res = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_round,
        nfold=5,
        metrics="rmse",
        early_stopping_rounds=early_stop,
        as_pandas=True,
        seed=seed,
    )
    best_rounds = int(cv_res['test-rmse-mean'].idxmin()) + 1
    print(f"[Train] Best CV RMSE: {cv_res['test-rmse-mean'].min():.6f}  "
          f"rounds={best_rounds}（CV 结果 {cv_res.shape[0]} 行）")

    # final model
    model = xgb.train(params, dtrain, num_boost_round=best_rounds)

    # save（把目标空间写进模型属性，推理时自动识别，无需额外的 meta 文件）
    model.set_attr(log_target="1" if log_target else "0")
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(model_dir, f"xgb_model_{ts}.json")
    model.save_model(model_path)
    print(f"[Train] Model saved -> {model_path}")
    return model_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model_dir", default="./model")
    parser.add_argument("--max_depth", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample", type=float, default=0.9)
    parser.add_argument("--num_round", type=int, default=300)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--no-log-target", dest="log_target", action="store_false",
                        help="用原始 P1 训练（默认用 log10(P1)）")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    train(
        data=args.data,  
        model_dir=args.model_dir,
        max_depth=args.max_depth,
        eta=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        num_round=args.num_round,
        early_stop=args.early_stop,
        seed=args.seed,
        log_target=args.log_target,
    )

