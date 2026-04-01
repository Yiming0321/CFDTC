#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial-regression inference script – single-parameter version  
(input can be either a file path or a numeric vector)
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def dict_to_model(model_dict: dict):
    """
    Restore (PolynomialFeatures, LinearRegression) tuple from a JSON-serialisable dict.
    """
    # 1. Re-build PolynomialFeatures
    degree = model_dict["degree"]
    n_features_in = model_dict["n_features"]
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # 2. Dummy fit to initialise internal attributes
    dummy = np.zeros((1, n_features_in))
    _ = poly.fit_transform(dummy)

    # 3. Overwrite powers_ (bypass read-only property via __dict__)
    poly.__dict__["powers"] = np.array(model_dict["powers"], dtype=int)

    # 4. Re-build LinearRegression
    lin = LinearRegression()
    lin.coef_ = np.array(model_dict["coefficients"])
    lin.intercept_ = model_dict["intercept"]
    return poly, lin


def predict(X_new: np.ndarray, poly: PolynomialFeatures, lin: LinearRegression):
    """Generate predictions using the restored model."""
    return lin.predict(poly.transform(X_new))


def _is_numeric_list(lst: list):
    """Quick heuristic: treat list as numeric if its first element can be cast to float."""
    try:
        float(lst[0])
        return True
    except ValueError:
        return False


def inference(model_path: str,
              data: str,
              feat_cols: list = ["Right_final", "Left_final", "Difference", "room_temperature"],
              output_path: str=None):
    """
    Main inference entry point.

    Parameters
    ----------
    model_path : str
        Path to the JSON file containing serialised model.
    input_arg : str
        Either a path to a .csv/.xlsx file or a whitespace/comma-separated
        numeric vector, e.g. "23.1 45 0.8 22".
    x_cols : list[str]
        Column names to be used as independent variables when reading a file.
        Ignored when input_arg is a numeric vector.
    output_path : str
        Directory where output.csv will be saved.
    """
    # 1. Load model
    print("[load]Model loaing from json...")
    with open(model_path, "r", encoding="utf-8") as f:
        poly, lin = dict_to_model(json.load(f))

    # 2. Parse input_arg
    print("[load]Data loading...")
    if isinstance(data, str):
        # 从文件读取
        
        if data.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(data)
        elif data.lower().endswith(".csv"):
            df = pd.read_csv(data)
        else:
            raise ValueError(f"Unsupported file format: {data}. Use .xlsx, .xls, or .csv")
        input_source = "file"
        
    elif isinstance(data, pd.DataFrame):
        # 直接使用 DataFrame
        df = data.copy()
        input_source = "dataframe"
        
    elif isinstance(data, np.ndarray):
        # numpy 数组转换为 DataFrame
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(feat_cols):
            raise ValueError(f"Feature dimension mismatch: array has {data.shape[1]} columns, "
                           f"but feat_cols has {len(feat_cols)}")
        df = pd.DataFrame(data, columns=feat_cols)
        input_source = "ndarray"
        
    else:
        raise TypeError(f"data must be str (file path), pd.DataFrame, or np.ndarray, got {type(data)}")


    # 3. Predict
    print("[Infer]Prediction processing...")
    pred = predict(df.values.tolist(), poly, lin)
    print("[Predict] finished")

    # 4. Save results
    if output_path is not None:
        if os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, "output.csv")
        pd.DataFrame({"prediction": pred}).to_csv(out_file, index=False, float_format="%.6f")
        print(f"Results saved → {out_file}")

    return pred


def parse_args():
    parser = argparse.ArgumentParser( description="Polynomial-regression inference (single-parameter version: input can be file or numeric vector)")
    parser.add_argument("-m", "--model", required=True, help="Path to JSON model file")
    parser.add_argument("-i", "--data", required=True, help='Data: file path (.csv/.xlsx) or numeric list, e.g. "23.1 45 0.8 22"')
    parser.add_argument("--x_cols", nargs="+",  default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Column names for independent variables (ignored for numeric input)")
    parser.add_argument("-o", "--output", default="output", help="Directory where output.csv will be saved")
    return parser.parse_args()


if __name__ == "__main__":

    # # example 1 # call from CMD format
    args = parse_args()
    inference(model_path=args.model,
              data=args.data,
              x_cols=args.x_cols,
              output_path=args.output)

    # # example 2 # call from API format
    # predict = inference(
    #     model_path=r"your absolte model path ",
    #     data=r"your absolte output path "
    # )

