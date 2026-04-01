#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smooth Transfer Inference

This script performs inference using both XGBoost and MLP models,
and then combines their predictions using the ensemble_predict function.

Usage:
    python inference.py --xgb_model path/to/xgb_model.json --mlp_model path/to/mlp_model.pth --data path/to/data.xlsx --output path/to/output.xlsx
"""
import os
import argparse
import pandas as pd
from datetime import datetime

# Import inference functions
from transfer import ensemble_predict


def load_xgb_inference():
    """Load XGBoost inference module"""
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_Model_training", "XGboost_src"))
        from infer_xgb import inference as xgb_inference
        return xgb_inference
    except ImportError as e:
        print(f"Error loading XGBoost inference: {e}")
        raise


def load_mlp_inference():
    """Load MLP inference module"""
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_Model_training", "MLP_src"))
        from train_mlp import inference as mlp_inference
        return mlp_inference
    except ImportError as e:
        print(f"Error loading MLP inference: {e}")
        raise


def run_ensemble_inference(xgb_model_path, mlp_model_path, data_path, output_path=None):
    """
    Run ensemble inference using XGBoost and MLP models.
    
    Args:
        xgb_model_path: Path to XGBoost model file
        mlp_model_path: Path to MLP model file
        data_path: Path to input data file
        output_path: Path to save output file
        
    Returns:
        pd.DataFrame: Data with predictions from both models and ensemble result
    """
    print("="*60)
    print("Running ensemble inference")
    print("="*60)
    
    # Load inference functions
    xgb_inference = load_xgb_inference()
    mlp_inference = load_mlp_inference()
    
    # Run XGBoost inference
    print("\n1. Running XGBoost inference...")
    xgb_results = xgb_inference(
        model_path=xgb_model_path,
        data=data_path,
        return_df=True
    )
    
    # Run MLP inference
    print("\n2. Running MLP inference...")
    mlp_results = mlp_inference(
        model_path=mlp_model_path,
        data=data_path,
        save_dir=None  # We'll save results ourselves
    )
    
    # Ensure both dataframes have the same index
    if len(xgb_results) != len(mlp_results):
        raise ValueError("XGBoost and MLP results have different lengths")
    
    # Combine results
    print("\n3. Combining results...")
    combined_results = xgb_results.copy()
    
    # Get predictions from both models
    xgb_preds = combined_results["Predicted"]
    mlp_preds = mlp_results["Predicted_Power(uW)"]
    
    # Apply ensemble prediction
    ensemble_preds = []
    for xgb_pred, mlp_pred in zip(xgb_preds, mlp_preds):
        ensemble_pred = ensemble_predict(mlp_pred, xgb_pred)
        ensemble_preds.append(ensemble_pred)
    
    # Add ensemble predictions to results
    combined_results["MLP_Predicted"] = mlp_preds
    combined_results["Ensemble_Predicted"] = ensemble_preds
    
    # Save results if output path is provided
    if output_path:
        print(f"\n4. Saving results to {output_path}...")
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        combined_results.to_excel(output_path, index=False)
        print(f"Results saved successfully!")
    
    print("\n" + "="*60)
    print("Ensemble inference completed")
    print("="*60)
    
    return combined_results


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smooth Transfer Ensemble Inference")
    parser.add_argument("--xgb_model", required=True, help="Path to XGBoost model file")
    parser.add_argument("--mlp_model", required=True, help="Path to MLP model file")
    parser.add_argument("--data", required=True, help="Path to input data file")
    parser.add_argument("--output", help="Path to save output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_ensemble_inference(
        xgb_model_path=args.xgb_model,
        mlp_model_path=args.mlp_model,
        data_path=args.data,
        output_path=args.output
    )