#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TE Adjuster Runner

This script provides a unified interface to run either the single or multi (category-based) TE adjuster.

Usage:
    python TEAdjuster_runner.py --mode single --load_path parm_single.json --data data-out_of_limit.xlsx --output output_single.xlsx
    python TEAdjuster_runner.py --mode multi --parm_file parm.json --data data-out_of_limit.xlsx --output output_multi.xlsx
    python TEAdjuster_runner.py --mode single --data_path original_data.xlsx --save_path parm_single.json
    python TEAdjuster_runner.py --mode multi --data_file original_data.xlsx --parm_output_file parm.json
"""
import argparse
import pandas as pd
import os
from TEAdjuster_multi import TEAdjuster as multi_adjuster
from TEAdjuster_single import TEAdjuster as single_adjuster

def run_single_adjuster(args):
    """Run the single (simple linear) TE adjuster"""

    
    # Initialize adjuster
    if args.data_path:
        # Training mode
        adjuster = single_adjuster(
            data_path=args.data_path,
            save_path=args.save_path
        )
        print("Training completed." if args.save_path else "Training completed (no save path provided).")
        return
    else:
        # Loading mode
        adjuster = single_adjuster(
            load_path=args.parm_path
        )
    
    # Process data
    if args.data:
        process_data(adjuster, args.data, args.output, args.temp_threshold, args.temp_target)


def run_multi_adjuster(args):
    """Run the multi (category-based) TE adjuster"""

    
    # Initialize adjuster
    if args.data_path:
        # Training mode
        adjuster = multi_adjuster(
            data_file=args.data_path,
            save_path=args.save_path
        )
        print("Training completed." if args.save_path else "Training completed (no output file provided).")
        return
    else:
        # Loading mode
        adjuster = multi_adjuster(
            parm_file=args.parm_path
        )
    
    # Process data
    if args.data:
        process_data(adjuster, args.data, args.output, args.temp_threshold, args.temp_target)


def process_data(adjuster, data_path, output_path, temp_threshold, temp_target):
    """Process data using the provided adjuster"""
    # Load data
    data = pd.read_excel(data_path)
    print(f"Loaded data from {data_path}")

    # Filter data where temperature exceeds threshold
    mask = data['room_temperature'] > temp_threshold
    print(f"Found {mask.sum()} records with temperature > {temp_threshold}°C")
    
    if mask.sum() == 0:
        print("No data to adjust. Exiting.")
        return
    
    new_temperatures = data.loc[mask, 'room_temperature']
    new_right_te = data.loc[mask, 'Right_final']
    new_left_te = data.loc[mask, 'Left_final']

    # Adjust to standard temperature
    print(f"Adjusting TE values to {temp_target}°C")
    adjusted_right, adjusted_left, adjusted_temp = adjuster.adjust_values(
        new_temperatures, new_right_te, new_left_te, temp_target
    )

    # Update dataframe with adjusted values
    data.loc[mask, 'Right_final'] = adjusted_right
    data.loc[mask, 'Left_final'] = adjusted_left
    data.loc[mask, 'room_temperature'] = adjusted_temp
    data.loc[mask, 'Difference'] = adjusted_right - adjusted_left

    # Save results
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No output path provided. Results not saved.")


def get_args():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description="TE Adjuster Runner")
    
    # Mode selection
    parser.add_argument("--mode", required=True, choices=["single", "multi"], help="Adjuster mode: single (simple linear) or multi (category-based)")
    
    # Common parameters
    parser.add_argument("--data", help="Path to data file to process")
    parser.add_argument("--output", help="Path to save output file")
    parser.add_argument("--temp_threshold", type=float, default=25.0, help="Temperature threshold for adjustment (default: 25.0°C)")
    parser.add_argument("--temp_target", type=float, default=23.0, help="Target temperature for adjustment (default: 23.0°C)")
    
    # Single mode parameters
    parser.add_argument("--parm_path", help="Path to load parameters")
    parser.add_argument("--data_path", help="Path to training data")
    parser.add_argument("--save_path", help="Path to save parameters")

    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    # Validate parameters based on mode
    if args.mode == "single":
        if not args.parm_path and not args.data_path:
            parser = argparse.ArgumentParser()
            parser.error("For single mode, either --parm_path or --data_path must be provided")
        run_single_adjuster(args)
    else:  # multi mode
        if not args.parm_path and not args.data_path:
            parser = argparse.ArgumentParser()
            parser.error("For multi mode, either --parm_path or --data_path must be provided")
        run_multi_adjuster(args)