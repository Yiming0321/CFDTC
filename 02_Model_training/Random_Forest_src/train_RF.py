#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Regression Training Script (CLI Version)
基于Train/Test Split的详细评估

Example:
    python train_rf.py -d data.xlsx -k 50 --test_size 0.15
    python train_rf.py --n_estimators 200 --max_depth 15
"""

import argparse
import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------- argument parsing ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Forest regression with Train/Test split and detailed metrics")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to raw-data Excel file/ data in dataframe format")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where artefacts will be saved")
    parser.add_argument("-k", "--random_key", type=int, default=50,help="Random seed for train_test_split")
    parser.add_argument("--test_size", type=float, default=0.15,help="Test set ratio (e.g., 0.15 for 15%%)")
    parser.add_argument("--n_estimators", type=int, default=100,help="Number of trees in the forest" )
    parser.add_argument( "--max_depth", type=int, default=10, help="Maximum depth of the tree (None for unlimited)")
    parser.add_argument( "--n_jobs", type=int, default=-1, help="Parallel jobs (-1 = use all cores)" )
    parser.add_argument( "--feature_cols", nargs="+", default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Feature column names (space-separated)" )
    parser.add_argument("--target_col", type=str, default="P1(uW)", help="Target column name")
    
    return parser.parse_args()


# ---------- metrics calculation ----------
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE with zero-protection
    mask = actual != 0
    if np.any(mask):
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
        
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }

def load_data(data, feat_cols):
    if isinstance(data, str):
        # 从文件读取
        if data.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(data)
        elif data.lower().endswith(".csv"):
            return pd.read_csv(data)
        else:
            raise ValueError(f"Unsupported file format: {data}. Use .xlsx, .xls, or .csv")

        
    elif isinstance(data, pd.DataFrame):
        # 直接使用 DataFrame
        return data.copy()
        
    elif isinstance(data, np.ndarray):
        # numpy 数组转换为 DataFrame
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(feat_cols):
            raise ValueError(f"Feature dimension mismatch: array has {data.shape[1]} columns, "
                           f"but feat_cols has {len(feat_cols)}")
        

        return pd.DataFrame(data, columns=feat_cols)
        
    else:
        raise TypeError(f"data must be str (file path), pd.DataFrame, or np.ndarray, got {type(data)}")


# ---------- metrics printing ----------
def print_metrics(metrics: dict, dataset_name: str = "Dataset") -> None:
    print(f"\n[Eval] {dataset_name} dataset result")
    print(f"   MSE:  {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   MAPE: {metrics['mape']:.4f}%")


# ---------- main pipeline ----------
def train(data: str,
          model_dir: str = "models",
          random_key: int = 50,
          test_size: float = 0.15,
          n_estimators: int = 100,
          max_depth: int | None = 10,
          n_jobs: int = -1,
          feat_cols: list[str]  = ["Right_final", "Left_final", "Difference", "room_temperature"],
          target_col: str = "P1(uW)") -> None:
    """
    Train a Random-Forest regressor with Train/Test split,
    output detailed metrics for both sets, and persist model + artefacts.
    """
    if feat_cols is None:
        feat_cols = ["Right_final", "Left_final", "Difference", "room_temperature"]
    
    # 1. prepare directory
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. load data
    print("[Train] Loading data…")
    df = load_data(data=data, feat_cols=feat_cols)
    X = df[feat_cols].values
    y = df[target_col].values
    
    # 3. train/test split (NO scaler for RF)
    print(f"[Train] Splitting data (test_size={test_size}, random_state={random_key})…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_key
    )
    
    # 4. model configuration
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': 42,
        'n_jobs': n_jobs
    }
    
    # 5. model training
    print("[Train] Training Random Forest…")
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    
    # 6. evaluation and printing (refactored)
    y_pred_train = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    print_metrics(train_metrics, "Train")
    
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    print_metrics(test_metrics, "Test")


    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    
    # 9. save model 
    model_file = os.path.join(model_dir, f'rf_model_{timestamp}__{random_key}.pkl')
    try:
        joblib.dump(model, model_file)
        print(f"[Save] Model saved to {model_file}")
    except Exception as e:
        print(f"[Error] Failed to save model: {str(e)}")
    
    print(f"\n[Done] Training complete. All artefacts saved to {model_dir}/")



if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data_path,
        model_dir=args.model_dir,
        random_key=args.random_key,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        feat_cols=args.feature_cols,
        target_col=args.target_col
    )

