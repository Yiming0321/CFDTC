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
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------- argument parsing ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polynomial regression fitting with coefficient preservation")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to input Excel file")
    parser.add_argument("--out_dir", type=str, default="poly_output", help="Output directory for results")
    parser.add_argument("-k", "--random_key", type=int, default=50, help="Random seed for reproducibility")
    parser.add_argument("--degree", type=int, default=None, help="Single polynomial degree (e.g., 3). If not set, uses degree_range")
    parser.add_argument("--degree_range", nargs=2, type=int, default=[1, 5], metavar=("START", "END"), help="Range of degrees to fit (inclusive), default: 1 5")
    parser.add_argument("--feature_cols", nargs="+", default=["Right_final", "Left_final", "Difference", "room_temperature"], help="Feature column names (space-separated)")
    parser.add_argument("--target_col", type=str, default="P1(uW)", help="Target column name")
    
    return parser.parse_args()


# ---------- metrics calculation ----------
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate regression metrics with zero-protection for MAPE.
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if np.any(mask) else np.nan
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(1 - mse / np.var(actual))
    }


# ---------- metrics printing ----------
def print_metrics(metrics: dict, dataset_name: str = "Dataset") -> None:
    """Format and print evaluation metrics."""
    print(f"\n[Eval] {dataset_name} fitting result")
    print(f"   MSE:   {metrics['mse']:.6f}")
    print(f"   RMSE:  {metrics['rmse']:.6f}")
    print(f"   MAE:   {metrics['mae']:.6f}")
    print(f"   MAPE:  {metrics['mape']:.4f}%")
    print(f"   R²:    {metrics['r2']:.6f}")


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
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Fit polynomial regression and return results.
    
    Returns:
        result_df: Predictions and residuals
        coef_df: Coefficients with feature names
        model: Fitted pipeline
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    coef = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    feature_names = model.named_steps['polynomialfeatures'].get_feature_names_out(feature_cols)
    
    residual = y - y_pred
    result_df = pd.DataFrame({
        'Actual': y.values,
        'Predicted': y_pred,
        'Residual': residual,
        'Residual_Ratio': np.abs(residual / np.where(y != 0, y, np.nan))
    })
    
    coefficients = np.insert(coef, 0, intercept)
    coef_df = pd.DataFrame({
        'Feature': ['Intercept'] + list(feature_names),
        'Coefficient': coefficients
    })
    
    return result_df, coef_df, model


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
    out_dir: str = r".\output",
    random_key: int = 50,
    degree: int | None = None,
    degree_range: tuple[int, int] = (1, 5),
    feature_cols: list[str] | None = None,
    target_col: str = "P1(uW)"
) -> None:
    """
    Train polynomial regression for specified degree(s).
    
    Args:
        data: Input data - either file path (str) or pandas DataFrame
        out_dir: Output directory for results
        random_key: Random seed for reproducibility
        degree: Single polynomial degree (overrides degree_range if set)
        degree_range: Range of degrees to fit (inclusive)
        feature_cols: Feature column names
        target_col: Target column name
    """
    if feature_cols is None:
        feature_cols = ["Right_final", "Left_final", "Difference", "room_temperature"]
    
    # Create output directories
    out_path = pathlib.Path(out_dir)
    model_dir = out_path / "model"
    out_path.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
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
        result_df, coef_df, model = fit_polynomial(X, y, d, feature_cols)
        
        # Calculate and print metrics
        metrics = calculate_metrics(y.values, result_df['Predicted'].values)
        print_metrics(metrics, f"Degree-{d}")
        
        # Save model parameters to JSON
        json_file = model_dir / f'poly_degree_{d}_{timestamp}__{random_key}.json'
        save_model_params(model, feature_cols, d, str(json_file))
        
        print(f"[Info] Coefficients count: {len(coef_df)}")
        print("-" * 50)
    
    print(f"\n[Done] All polynomial fittings completed. Models saved in {model_dir}/")


# ---------- entry point ----------
if __name__ == "__main__":
    
    # # example 1 # call from CMD format
    args = parse_args()
    train(
        data=args.data,
        out_dir=args.out_dir,
        random_key=args.random_key,
        degree=args.degree,
        degree_range=tuple(args.degree_range),
        feature_cols=args.feature_cols,
        target_col=args.target_col
    )

    # # example 2 # call from API format
    # train(
    #     data=r"your absolte data path ",
    #     out_dir=r"your absolte output path "
    # )