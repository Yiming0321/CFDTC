#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Regression Training Script 
"""

import argparse
import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


# ---------- argument parsing ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Forest regression with Train/Test split and detailed metrics")
    parser.add_argument("--data", type=str, required=True, help="Path to raw-data Excel file/ data in dataframe format")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where artefacts will be saved")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=100,help="Number of trees in the forest" )
    parser.add_argument( "--max_depth", type=int, default=10, help="Maximum depth of the tree (None for unlimited)")
    parser.add_argument( "--n_jobs", type=int, default=-1, help="Parallel jobs (-1 = use all cores)" )
    parser.add_argument( "--feature_cols", nargs="+", default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Feature column names (space-separated)" )
    parser.add_argument("--target_col", type=str, default="P1(uW)", help="Target column name")
    
    return parser.parse_args()




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


# ---------- main pipeline ----------
def train(data: str,
          model_dir: str = "models",
          random_state: int = 42,
          n_estimators: int = 100,
          max_depth: int | None = 10,
          n_jobs: int = -1,
          feat_cols: list[str]  = ["Right_final", "Left_final", "Difference", "room_temperature"],
          target_col: str = "P1(uW)") -> None:
    """
    Train a Random-Forest regressor using all data,
    and persist model + artefacts.
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
    
    # 3. model configuration
    rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'random_state': random_state,
        'n_jobs': n_jobs
    }
    
    # 4. model training
    print("[Train] Training Random Forest…")
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 5. save model 
    model_file = os.path.join(model_dir, f'rf_model_{timestamp}.pkl')
    try:
        joblib.dump(model, model_file)
        print(f"[Save] Model saved to {model_file}")
    except Exception as e:
        print(f"[Error] Failed to save model: {str(e)}")
    
    print(f"\n[Done] Training complete. All artefacts saved to {model_dir}/")



if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data,
        model_dir=args.model_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        feat_cols=args.feature_cols,
        target_col=args.target_col
    )
