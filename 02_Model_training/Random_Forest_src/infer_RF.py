#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest Inference Script
Compatible with train_rf.py output (no scaler, timestamped filenames)

Example:
    # Specify exact model file
    python infer_rf.py --model models/rf_model_20250320143000__50.pkl --data new_data.xlsx
    
    # Auto-load latest model from directory
    python infer_rf.py --model_dir models --data new_data.xlsx --out_dir ./infer_output
    
    # Single sample inference
    python infer_rf.py --model_dir models --data "1.2,3.4,5.6,7.8" --feature_cols Right_final Left_final Difference room_temperature
"""

import argparse
import os
import pathlib
import re
import datetime
from typing import Union, List, Dict, Optional
import numpy as np
import pandas as pd
import joblib




def load_model(model_path: str) -> object:
    """Load RandomForest model from joblib file."""
    print(f"[Load] Model loading from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    return model


# ---------- inference ----------
def inference(
    data: Union[str, Dict[str, float], List[float], np.ndarray],
    model: Optional[object] = None,
    model_path: Optional[str] = None,
    feature_cols: Optional[List[str]] = ['Right_final', 'Left_final', 'Difference', 'room_temperature'],
    output_dir: str = "./output"
) -> Union[pd.DataFrame, float]:
    """
    Unified inference interface for Random Forest.
    
    Args:
        data: Input data - file path (str), single sample (dict/list/ndarray), or DataFrame
        model: Pre-loaded model object (optional if model_path/model_dir provided)
        model_path: Direct path to model file (.pkl)
        feature_cols: Feature column names (default: ['Right_final', 'Left_final', 'Difference', 'room_temperature'])
        out_dir: Output directory for batch inference results
    
    Returns:
        - Batch (file path or DataFrame): DataFrame with predictions
        - Single sample: float prediction value
    """
    # Default features
    feature_cols = feature_cols or ["Right_final", "Left_final", "Difference", "room_temperature"]
    
    # Load model if not provided

    model = load_model(model_path)
    
    print("[Load] Data loading ...")
    # ---------- Single sample: dict ----------
    if isinstance(data, dict):
        x = np.array([[data[col] for col in feature_cols]])
        pred = model.predict(x)[0]
        return float(pred)
    
    # ---------- Single sample: list or ndarray ----------
    if isinstance(data, (list, np.ndarray)):
        x = np.array([data]) if isinstance(data, list) else data.reshape(1, -1)
        if x.shape[1] != len(feature_cols):
            raise ValueError(f"Input dimension {x.shape[1]} != expected {len(feature_cols)}")
        pred = model.predict(x)[0]
        return float(pred)
    
    # ---------- Batch: DataFrame ----------
    if isinstance(data, pd.DataFrame):
        assert all(c in data.columns for c in feature_cols), \
            f"Input DataFrame missing columns. Required: {feature_cols}"
        
        X = data[feature_cols].values
        preds = model.predict(X)
        
        result = data.copy()
        result["Predicted_P1(uW)"] = preds
        return result
    
    # ---------- Batch: file path ----------
    if isinstance(data, str):
        data_path = pathlib.Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data}")
        
        # Load data
        if data.lower().endswith(".xlsx"):
            df = pd.read_excel(data)
        elif data.lower().endswith(".csv"):
            df = pd.read_csv(data)
        else:
            raise ValueError("Data file must be .xlsx or .csv")
        
        # Validate columns
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input file: {missing}")
        
        # Predict
        print("[Predict] Prediction proicessing...")
        X = df[feature_cols].values
        preds = model.predict(X)
        
        # Prepare outputprint("[Predict] prediction processing ...")
        result = df.copy()
        result["Predicted_P1(uW)"] = preds
        
        # Save
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        out_file = f"rf_infer_result_{timestamp}.xlsx"
        out_path = pathlib.Path(output_dir) / out_file
        result.to_excel(out_path, index=False)
        print(f"[Save] Inference results saved to {out_path}")
        
        return result
    
    raise TypeError(f"Unsupported data type: {type(data)}")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random Forest Inference (compatible with train_rf.py output)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Direct path to model file (.pkl). If not provided, uses --model_dir"
    )
    parser.add_argument(
        "-d", "--data", type=str, required=True,
        help="Input data: file path (.xlsx/.csv), or comma-separated values for single inference"
    )
    parser.add_argument(
        "--feature_cols", nargs="+",
        default=["Right_final", "Left_final", "Difference", "room_temperature"],
        help="Feature column names (space-separated)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./infer_output",
        help="Output directory for batch inference results"
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Treat --data as comma-separated single sample (e.g., '1.2,2.3,3.4,4.5')"
    )
    
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    result = inference(
        model_path=args.model,
        data=args.data,  # 这里传入文件路径字符串
        feat_cols=args.feat_cols,
        output_dir=args.output_dir
    )

