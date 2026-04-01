#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polynomial Regression Training Script (CLI Version)
Supports 1-6 degree polynomial fitting with model parameters saved as JSON.

Example:
    python train_poly.py -d data.xlsx --degree 3 --out_dir ./output
    python train_poly.py -d data.xlsx --degree_range 1 5 --feature_cols A B C D
"""

import argparse
import json
import os
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# ---------- argument parsing ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polynomial regression fitting with coefficient preservation")
    parser.add_argument("--data", type=str, required=True, help="Path to input Excel file")
    parser.add_argument("--model_dir", type=str, default="model", help="Output directory for model saving")
    parser.add_argument("--degree", type=int, default=None, help="Single polynomial degree (e.g., 3). If not set, uses degree_range")
    parser.add_argument("--degree_range", nargs=2, type=int, default=[1, 5], metavar=("START", "END"), help="Range of degrees to fit (inclusive), default: 1 5")
    parser.add_argument("--feature_cols", nargs="+", default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Feature column names (space-separated)")
    parser.add_argument("--target_col", type=str, default="P1(uW)", help="Target column name")
    
    return parser.parse_args()



# ---------- data loading helper ----------
def load_data(data: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load data from file path or return DataFrame directly.
    
    Args:
        data: Either a file path (str) or pandas DataFrame
        
    Returns:
        pandas DataFrame
        
    Raises:
        FileNotFoundError: If file path does not exist
        TypeError: If data is neither str nor pd.DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"Data file not found: {data}")
        return pd.read_excel(data)
    else:
        raise TypeError(f"data must be str (file path) or pd.DataFrame, got {type(data).__name__}")


# ---------- polynomial fitting ----------
def fit_polynomial(
    X: pd.DataFrame,
    y: pd.Series,
    degree: int,
    feature_cols: list[str]
) -> tuple[pd.DataFrame, object]:
    """
    Fit polynomial regression and return coefficients and model.
    
    Returns:
        coef_df: Coefficients with feature names
        model: Fitted pipeline
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    
    model.fit(X, y)
    
    coef = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(feature_cols)
    
    coefficients = np.insert(coef, 0, intercept)
    coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + list(feature_names),
        'Coefficient': coefficients
    })
    
    return coef_df, model


# ---------- save model parameters ----------
def save_model_params(
    model: object,
    feature_cols: list[str],
    degree: int,
    out_path: str
) -> None:
    """
    Save polynomial regression model parameters to JSON file.
    
    Args:
        model: Fitted sklearn pipeline
        feature_cols: List of feature column names
        degree: Polynomial degree
        out_path: Output JSON file path
    """
    # Extract model components
    poly_features = model.named_steps['polynomialfeatures']
    linear_reg = model.named_steps['linearregression']
    
    # Build parameters dictionary
    params = {
        'model_type': 'polynomial_regression',
        'degree': degree,
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'n_output_features': int(poly_features.n_output_features_),
        'powers': poly_features.powers_.tolist(),
        'intercept': float(linear_reg.intercept_),
        'coefficients': linear_reg.coef_.tolist(),
        'feature_names': poly_features.get_feature_names_out(feature_cols).tolist()
    }
    
    # Save to JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"[Save] Model parameters saved to {out_path}")


# ---------- main pipeline ----------
def train(
    data: str | pd.DataFrame,
    model_dir: str = r".\output",
    degree: int | None = None,
    degree_range: tuple[int, int] = (1, 5),
    feature_cols: list[str] | None = None,
    target_col: str = "P1(uW)"
) -> None:
    """
    Train polynomial regression for specified degree(s).
    
    Args:
        data: Input data - either file path (str) or pandas DataFrame
        model_dir: Output directory for results
        degree: Single polynomial degree (overrides degree_range if set)
        degree_range: Range of degrees to fit (inclusive)
        feature_cols: Feature column names
        target_col: Target column name
    """
    if feature_cols is None:
        feature_cols = ["Right_final", "Left_final", "Difference", "room_temperature"]
    
    # Create model directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    print("[Load] Loading data...")
    df = load_data(data)
    X = df[feature_cols]
    y = df[target_col]
    
    # Determine degrees to fit
    if degree is not None:
        degrees = [degree]
    else:
        degrees = range(degree_range[0], degree_range[1] + 1)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Fit for each degree
    for d in degrees:
        print(f"\n[Train] Fitting polynomial degree {d}...")
        
        # Fit model
        coef_df, model = fit_polynomial(X, y, d, feature_cols)
        

        result_df, coef_df, model = fit_polynomial(X, y, d, feature_cols)
        # Save model parameters to JSON
        json_file = f'{model_dir}/poly_degree_{d}_{timestamp}.json'
        save_model_params(model, feature_cols, d, str(json_file))
        
        json_file = model_dir / f'poly_degree_{d}_{timestamp}__{random_key}.json'
        print("-" * 50)
    
    print(f"\n[Done] All polynomial fittings completed. Models saved in {model_dir}/")


# ---------- entry point ----------
if __name__ == "__main__":
    
    
    args = parse_args()
    train(
        data=args.data,
        model_dir=args.model_dir,
        degree=args.degree,
        degree_range=tuple(args.degree_range),
        feature_cols=args.feature_cols,
        target_col=args.target_col
    )


