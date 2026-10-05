#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference entry – supports file path, DataFrame, or numpy array as input
"""
import os
import argparse
from typing import Union, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime


def load_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def inference(
    model_path: str,
    data: Union[str, pd.DataFrame, np.ndarray],
    feat_cols = ["Right_final", "Left_final", "Difference", "room_temperature"],
    output_dir: str = "./output",
    return_df: bool = True,
    log_target: Optional[bool] = None,
):
    """
    Run inference.
    
    Args:
        model_path: Path to XGBoost model file (.json)
        data: File path (str), pandas DataFrame, or numpy array
        feat_cols: Feature column names (used for both DataFrame selection and numpy array labeling)
        out_dir: Output directory for results (only used when input is file path)
        return_df: If True, return DataFrame; if False, return raw predictions
        log_target: 模型是否在 log10(P1) 空间训练。None=自动读模型属性判断；
                    旧模型（无该属性）按 P1 原始值处理。
    
    Returns:
        pd.DataFrame or np.ndarray: Prediction results
    """
    
    # 处理不同类型的输入数据
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

    # 提取特征 (对 DataFrame 模式确保列存在)
    missing_cols = set(feat_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing feature columns: {missing_cols}")
    
    X = df[feat_cols].values
    dm = xgb.DMatrix(X)

    # 预测
    model = load_model(model_path)

    # 判断模型是否在 log10(P1) 空间训练：优先用显式传参，否则读模型自带属性
    if log_target is None:
        attr = model.attributes().get("log_target")
        log_target = (attr == "1")
        if attr is None:
            print("[Infer] 模型未记录目标空间，按 P1 原始值处理；"
                  "若该模型是用 log10(P1) 训练的，请显式传 log_target=True")
    print(f"[Infer] 目标空间: {'log10(P1)' if log_target else 'P1 原始值'}")

    y_pred = model.predict(dm)
    if log_target:
        y_pred = np.power(10.0, y_pred)      # log10 -> uW

    # 构建结果
    res_df = df.copy()
    res_df["Predicted"] = y_pred

    # 仅在输入为文件路径时保存到磁盘
    if input_source == "file" and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = f"xgb_result_{datetime.now():%Y%m%d%H%M%S}.xlsx"
        out_path = os.path.join(output_dir, out_file)
        res_df.to_excel(out_path, index=False)
        print(f"[Infer] Results saved -> {out_path}")

    if return_df:
        return res_df
    else:
        return y_pred


def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Inference")
    parser.add_argument("--model", required=True, help="path to xgb_model_xx.json")
    parser.add_argument("--data", required=True, help="path to input file (.xlsx or .csv) or directly np.ndarray/pd.DataFrame")
    parser.add_argument("--feat_cols", nargs="+", 
                       default=["Right_final", "Left_final", "Difference", "room_temperature"], 
                       help="feature column names")
    parser.add_argument("--output", default="./output", help="output directory")
    parser.add_argument("--log_target", choices=["auto", "true", "false"], default="auto",
                        help="目标空间：auto=读模型属性自动判断（默认），true/false=强制")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    result = inference(
        model_path=args.model,
        data=args.data, 
        feat_cols=args.feat_cols,
        output_dir=args.output,
        log_target={"auto": None, "true": True, "false": False}[args.log_target],
    )
