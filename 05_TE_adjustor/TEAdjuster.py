import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import json
import os
from typing import Union, Optional, Dict, List, Tuple
from datetime import datetime


class TEAdjuster:
    """
    Thermoelectric (TE) Temperature Adjustment Tool with persistent parameter support.
    Fixes JSON serialization key type mismatch issues by enforcing string keys.
    """
    
    def __init__(
        self, 
        data_file: Optional[str] = None,
        parm_file: Optional[str] = None,
        parm_output_file: Optional[str] = None
    ):
        """
        Initialize TEAdjuster with either data fitting or parameter loading.
        
        Args:
            data_file: Path to Excel data for initial fitting
            parm_file: Path to JSON parameter file for loading existing model
            parm_output_file: Path to save fitted parameters (only used with data_file)
            
        Raises:
            ValueError: If both data_file and parm_file provided, or neither provided
        """
        # Validate parameter combinations
        if data_file is not None and parm_file is not None:
            raise ValueError("Cannot provide both data_file and parm_file. Choose initialization mode.")
        if data_file is None and parm_file is None:
            raise ValueError("Must provide either data_file or parm_file.")
            
        self.categories: List[str] = []  # Ordered list of category identifiers
        self.line_parm_right: Dict[str, List[float]] = {}  # Right TE line parameters {category: [k, b]}
        self.line_parm_left: Dict[str, List[float]] = {}   # Left TE line parameters {category: [k, b]}
        
        if parm_file is not None:
            self._load_params(parm_file)
            print(f"Parameters loaded from: {parm_file}")
        else:
            self._fit_from_data(data_file, parm_output_file)
            print(f"Fitting completed from data source: {data_file}")
    
    def _fit_from_data(self, data_file: str, parm_output_file: Optional[str] = None):
        """
        Perform linear fitting from Excel data.
        
        Fits linear relationships between temperature and TE values for each category,
        storing slope (k) and intercept (b) for y = kx + b.
        """
        data = pd.read_excel(data_file)
        
        # Force string conversion and sorting to ensure fixed order and JSON compatibility
        self.categories = sorted([str(c) for c in set(data['category'])])
        
        for cat_key in self.categories:
            # Filter data for current category (handle both string and int original types)
            if data['category'].dtype == object:
                cat_data = data[data['category'] == cat_key]
            else:
                cat_data = data[data['category'] == int(cat_key)]
            
            temperature = cat_data["room_temperature"]
            
            # Fit Left TE line: TE_left = k * temp + b
            te_left = cat_data["Left_final"]
            k_left, b_left = self.linear_fit(X=temperature, Y=te_left)
            self.line_parm_left[cat_key] = [float(k_left), float(b_left)]
            
            # Fit Right TE line: TE_right = k * temp + b
            te_right = cat_data["Right_final"]
            k_right, b_right = self.linear_fit(X=temperature, Y=te_right)
            self.line_parm_right[cat_key] = [float(k_right), float(b_right)]

        if parm_output_file:
            self.save_params(output_file=parm_output_file)
    
    def save_params(self, output_file: str) -> str:
        """
        Save fitted parameters to JSON file.
        
        All keys are strings to ensure JSON compatibility.
        
        Args:
            output_file: Path to save JSON file
            
        Returns:
            Absolute path of saved file
        """
        params = {
            "line_parm_right": self.line_parm_right,  # Keys already strings
            "line_parm_left": self.line_parm_left,
            "categories": self.categories,  # String list
            "created_at": datetime.now().isoformat()
        }
        
        # Create output directory if not exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        
        abs_path = os.path.abspath(output_file)
        print(f"Parameters saved to: {abs_path}")
        return abs_path
    
    def _load_params(self, param_file: str):
        """
        Load parameters from JSON file.
        
        JSON loads all keys as strings, matching self.categories (string list).
        """
        with open(param_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        self.line_parm_right = params["line_parm_right"]
        self.line_parm_left = params["line_parm_left"]
        self.categories = params["categories"]  # Already string list

    def linear_fit(self, X: pd.Series, Y: pd.Series) -> Tuple[float, float]:
        """
        Perform 1st-degree polynomial fitting (linear regression).
        
        Returns:
            Tuple of (slope, intercept) for y = slope * x + intercept
        """
        k, b = np.polyfit(x=X, y=Y, deg=1)
        return k, b

    def find_2pnts(self, checked_value: float, value_list: List[float]) -> Union[Tuple[int, int], Tuple[int]]:
        """
        Find nearest mapping points for interpolation/extrapolation.
        
        Locates where checked_value falls within value_list to determine
        interpolation boundaries or nearest point for extrapolation.
        
        Args:
            checked_value: Target value to locate
            value_list: List of reference values (category theoretical values at specific temp)
            
        Returns:
            - (left_idx, right_idx): Two-point tuple for interpolation (normal range)
            - (nearest_idx,): Single-point tuple for extrapolation (out of bounds)
        """
        value_array = np.array(value_list)
        
        # === Case 1: Normal range, has both left and right bounding points ===
        smaller_indices = np.where(value_array < checked_value)[0]
        larger_indices = np.where(value_array > checked_value)[0]
        
        if len(smaller_indices) > 0 and len(larger_indices) > 0:
            # Max of smaller values (closest from left)
            left_idx = smaller_indices[np.argmax(value_array[smaller_indices])]
            # Min of larger values (closest from right)
            right_idx = larger_indices[np.argmin(value_array[larger_indices])]
            return (left_idx, right_idx)
        
        # === Case 2: Below minimum, take nearest (minimum) ===
        elif len(smaller_indices) == 0 and len(larger_indices) > 0:
            nearest_idx = np.argmin(value_array)  # Index of minimum value
            return (nearest_idx,)
        
        # === Case 3: Above maximum, take nearest (maximum) ===
        elif len(larger_indices) == 0 and len(smaller_indices) > 0:
            nearest_idx = np.argmax(value_array)  # Index of maximum value
            return (nearest_idx,)
        
        # === Case 4: All values equal (degenerate case) ===
        else:
            return (0,)

    def single_converter(self, temperature: float, te_value: float, 
                        specified_temperature: float, line_parm: Dict) -> float:
        """
        Core TE temperature conversion algorithm.
        
        Adjusts TE value from measured temperature to specified standard temperature
        using category-based linear interpolation/extrapolation.
        
        Logic branches:
        - Two-point mode: Locate TE between two categories at current temp, 
                         map proportionally to specified temp (interpolation)
        - Single-point mode: TE outside category range, use nearest category's 
                            theoretical value at specified temp (extrapolation)
        
        Args:
            temperature: Measured room temperature
            te_value: Measured TE value (Left or Right)
            specified_temperature: Target standard temperature for adjustment
            line_parm: Line parameters dict (either left or right)
            
        Returns:
            Temperature-adjusted TE value
        """
        # Step 1: Calculate theoretical TE values for each category at current temperature
        te_values_current = [np.polyval(line_parm[cat], temperature) for cat in self.categories]
        
        # Step 2: Find mapping points (two-point or single-point)
        arg_result = self.find_2pnts(checked_value=te_value, value_list=te_values_current)
        
        # === Branch A: Single-point mapping (boundary extrapolation) ===
        if len(arg_result) == 1:
            cat_idx = arg_result[0]
            cat_key = self.categories[cat_idx]
            # Directly use this category's theoretical value at specified temperature
            te_adjusted = np.polyval(line_parm[cat_key], specified_temperature)
            return te_adjusted
        
        # === Branch B: Two-point interpolation (normal range) ===
        # Get theoretical values at current temperature for bounding categories
        left_te_current = te_values_current[arg_result[0]]
        right_te_current = te_values_current[arg_result[1]]
        
        # Calculate position ratio r: (TE - left) / (right - TE)
        # r represents relative distance from left (r=1 means midpoint, r>1 leans right, r<1 leans left)
        r = (te_value - left_te_current) / (right_te_current - te_value)
        
        # Calculate theoretical values at specified temperature for bounding categories
        left_cat = self.categories[arg_result[0]]
        right_cat = self.categories[arg_result[1]]
        left_te_std = np.polyval(line_parm[left_cat], specified_temperature)
        right_te_std = np.polyval(line_parm[right_cat], specified_temperature)
        
        # Weighted harmonic interpolation: apply same ratio r at standard temperature
        # Derivation: r = (x-left)/(right-x) → x = (r*right + left)/(1+r)
        te_adjusted = (r * right_te_std + left_te_std) / (1 + r)
        
        return te_adjusted

    def adjust_values(self, new_temperatures: pd.Series, new_right_te: pd.Series, 
                     new_left_te: pd.Series, specified_temperature: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Batch adjustment of TE values to specified temperature.
        
        Args:
            new_temperatures: Series of measured temperatures
            new_right_te: Series of measured right TE values
            new_left_te: Series of measured left TE values
            specified_temperature: Target temperature for adjustment
            
        Returns:
            Tuple of (adjusted_right_te, adjusted_left_te, target_temperature_series)
        """
        adjusted_right = []
        adjusted_left = []
        
        for temp, te_right, te_left in zip(new_temperatures, new_right_te, new_left_te):
            # Adjust right TE
            adj_right = self.single_converter(
                temperature=temp, 
                te_value=te_right, 
                specified_temperature=specified_temperature,
                line_parm=self.line_parm_right
            )
            adjusted_right.append(adj_right)
            
            # Adjust left TE
            adj_left = self.single_converter(
                temperature=temp, 
                te_value=te_left, 
                specified_temperature=specified_temperature,
                line_parm=self.line_parm_left
            )
            adjusted_left.append(adj_left)
            
        return (pd.Series(adjusted_right), 
                pd.Series(adjusted_left), 
                pd.Series([specified_temperature] * len(adjusted_right)))


if __name__ == "__main__":
    # # First run: Fit and save parameters
    # adjuster = TEAdjuster(
    #     data_file=r"absolute path to data, e.g., original_data.xlsx",
    #     parm_output_file=r"absolute output path for parm.json"
    # )
    
    # Subsequent runs: Load existing parameters
    adjuster = TEAdjuster(
        parm_file=r"parm.json"
    )
    
    # Process new data
    data = pd.read_excel(r"data-out_of_limit.xlsx")

    # Filter data where temperature exceeds threshold (e.g., > 25°C)
    mask = data['room_temperature'] > 25
    new_temperatures = data.loc[mask, 'room_temperature']
    new_right_te = data.loc[mask, 'Right_final']
    new_left_te = data.loc[mask, 'Left_final']

    # Adjust to standard temperature (e.g., 23°C)
    adjusted_right, adjusted_left, adjusted_temp = adjuster.adjust_values(
        new_temperatures, new_right_te, new_left_te, 23
    )

    # Update dataframe with adjusted values
    data.loc[mask, 'Right_final'] = adjusted_right
    data.loc[mask, 'Left_final'] = adjusted_left
    data.loc[mask, 'room_temperature'] = adjusted_temp
    data.loc[mask, 'Difference'] = adjusted_right - adjusted_left

    # Save results
    data.to_excel(r'your_output_path.xlsx', index=False)