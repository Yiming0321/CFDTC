#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM inference script – fully parameterised via argparse.
"""

import argparse
import os
import datetime
import joblib
import pandas as pd


def inference(model_path: str,
              input_path: str,
              output_path: str,
              feat_cols: list[str],
              pred_col: str) -> None:
    """
    Load a LightGBM model saved with joblib and generate predictions.
    All arguments are injected explicitly.
    """

    # 1. load model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # 2. load new data
    df = pd.read_excel(input_path)
    X = df[feat_cols].copy()

    # 3. predict
    y_pred = model.predict(X)
    df[pred_col] = y_pred

    # 4. save results
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(output_path, f"LightGBM_result_{timestamp}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Prediction finished -> {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference with a LightGBM regression model"
    )
    parser.add_argument("--model", "-m", required=True, help="Path to trained *.pkl model")
    parser.add_argument("--input", "-i", required=True, help="Path to Excel file for inference")
    parser.add_argument("--out", "-o", default="./output", help="Folder where prediction file will be saved")
    parser.add_argument("--feat", "-f", nargs="+", default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Feature column names (must match training order)")
    parser.add_argument("--pred_col", "-c", default="P1(uW)", help="Column name for predictions")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(
        model_path=args.model,
        input_path=args.input,
        output_path=args.out,
        feat_cols=args.feat,
        pred_col=args.pred_col
    )