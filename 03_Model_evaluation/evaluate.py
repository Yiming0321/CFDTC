#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Algorithm Regression Metrics Batch Calculator
Command-line driven, results written to Excel
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calc_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate common regression metrics and return as dictionary.
    
    Parameters:
    -----------
    y_true : pd.Series
        Ground truth actual values
    y_pred : pd.Series
        Predicted values from algorithm
        
    Returns:
    --------
    dict : Dictionary containing MSE, RMSE, MAE, and MAPE(%)
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE with zero-division protection
    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.inf
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}


def read_data(file_path: str | Path,
              sheet: str | int = 0,
              actual_col: str = "Actual") -> pd.DataFrame:
    """
    Read Excel file and perform basic validation.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to input Excel file
    sheet : str or int, default 0
        Sheet name or index to read
    actual_col : str, default "Actual"
        Column name containing ground truth values
        
    Returns:
    --------
    pd.DataFrame : Loaded and validated dataframe
    
    Raises:
    -------
    FileNotFoundError : If file does not exist
    ValueError : If Excel reading fails
    KeyError : If actual_col is not found in columns
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except Exception as e:
        raise ValueError(f"Failed to read Excel: {e}") from e

    if actual_col not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise KeyError(f"Column '{actual_col}' not found. Available columns: {available_cols}")

    # Check for NaN values in actual column
    if df[actual_col].isna().any():
        print(f"Warning: Found {df[actual_col].isna().sum()} NaN values in '{actual_col}' column")

    return df


def build_cli() -> argparse.ArgumentParser:
    """
    Build and configure argument parser for command line interface.
    
    Returns:
    --------
    argparse.ArgumentParser : Configured parser object
    """
    parser = argparse.ArgumentParser(
        description="Batch calculate regression metrics for multiple algorithms, write results to Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python metrics_calculator.py -i results.xlsx -o output.xlsx
  python metrics_calculator.py -i predictions.xlsx --actual-col True_Value -s Sheet2
        """
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Input Excel file path containing actual and predicted values"
    )
    parser.add_argument(
        "-s", "--sheet", 
        default=0, 
        help="Sheet name or index (default: 0, first sheet)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="metrics.xlsx", 
        help="Output Excel file path (default: metrics.xlsx)"
    )
    parser.add_argument(
        "--actual-col", 
        default="Actual",
        help="Column name for ground truth values (default: Actual)"
    )
    return parser


def main(argv=None):
    """
    Main execution function.
    
    Parameters:
    -----------
    argv : list, optional
        Command line arguments (default: None, uses sys.argv)
    """
    args = build_cli().parse_args(argv)

    # ---- Read input data ----
    print(f"Reading data from: {args.input}")
    df = read_data(args.input, args.sheet, args.actual_col)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ---- Automatically identify algorithm columns ----
    # All columns except the actual value column are treated as algorithm predictions
    algo_cols = [col for col in df.columns if col != args.actual_col]
    
    if not algo_cols:
        raise ValueError(f"No prediction columns found. Only '{args.actual_col}' column present.")
    
    print(f"Found {len(algo_cols)} algorithm columns: {', '.join(algo_cols)}")

    # ---- Batch calculation of metrics ----
    records = []
    for col in algo_cols:
        # Skip columns with all NaN values
        if df[col].isna().all():
            print(f"Warning: Skipping '{col}' - all values are NaN")
            continue
            
        # Calculate metrics for valid data points
        valid_mask = df[args.actual_col].notna() & df[col].notna()
        if valid_mask.sum() == 0:
            print(f"Warning: Skipping '{col}' - no valid data pairs found")
            continue
            
        y_true_valid = df.loc[valid_mask, args.actual_col]
        y_pred_valid = df.loc[valid_mask, col]
        
        metrics = calc_regression_metrics(y_true_valid, y_pred_valid)
        metrics["Algorithm"] = col
        metrics["Valid_Samples"] = valid_mask.sum()
        records.append(metrics)

    if not records:
        raise ValueError("No valid metrics calculated. Check input data quality.")

    # Create output dataframe with Algorithm as index
    out_df = pd.DataFrame(records).set_index("Algorithm")
    
    # Reorder columns for better readability
    col_order = ["Valid_Samples", "MSE", "RMSE", "MAE", "MAPE(%)"]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]

    # ---- Write results to Excel ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_df.to_excel(out_path)
    print(f"\nEvaluation completed successfully")
    print(f"Results saved to: {out_path.resolve()}")
    print(f"\nMetrics Summary:")
    print(out_df.to_string())


if __name__ == "__main__":
    main()