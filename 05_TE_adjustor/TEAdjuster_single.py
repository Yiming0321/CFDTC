"""
Functionality:
    1. Automatically train linear relationship between temperature and left/right TE values.
    2. Adjust measured TE values to a specified reference temperature using fitted parameters.
    3. Support parameter persistence (JSON save/load) for future reuse.
Version: v2.0
Modified: 2026/03/29
"""
import numpy as np
import pandas as pd
from scipy import stats
import json
import os
from typing import Union, List, Optional, Dict, Any


class TEAdjuster:
    """
    Thermoelectric (TE) Value Temperature Adjuster.
    
    Calibrates TE measurements to a standard temperature by establishing
    linear relationships between room temperature and TE readings (Left/Right).
    """
    
    def __init__(
        self, 
        load_path: Optional[str] = None, 
        data_path: Optional[str] = None, 
        save_path: Optional[str] = None
    ):
        """
        Initialize TEAdjuster in one of two modes:
        1. Training mode: Fit from Excel data and optionally save parameters.
        2. Loading mode: Restore previously fitted parameters from JSON.
        
        Args:
            load_path: Path to JSON parameter file for loading existing model.
            data_path: Path to Excel file containing training data (room_temperature, Right_final, Left_final).
            save_path: Path to save JSON parameters after training (only used with data_path).
            
        Raises:
            ValueError: If insufficient arguments provided (must provide either load_path or data_path).
        """
        if load_path is not None:
            # Loading mode: Restore parameters from file
            print(f"Loading parameters from {load_path}")
            self._load_params(load_path)
        elif data_path is not None:
            # Training mode: Fit linear model from data
            data = pd.read_excel(data_path)
            temperatures = data["room_temperature"]
            right_te = data["Right_final"]
            left_te = data["Left_final"]
            print("Data loaded, starting fitting...")
            
            # Perform linear regression fitting
            self._fit(temperatures, right_te, left_te)
            
            # Auto-save if path provided
            if save_path is not None:
                self.save_params(save_path)
                print(f"Parameters saved to {save_path}")
        else:
            raise ValueError("Must provide either data_path (for training) or load_path (for loading existing parameters)")
    
    def _fit(
        self, 
        temperatures: Union[pd.Series, np.ndarray, List], 
        right_te: Union[pd.Series, np.ndarray, List], 
        left_te: Union[pd.Series, np.ndarray, List]
    ):
        """
        Internal method: Perform linear regression fitting.
        
        Fits two linear models:
        - Temperature ~ Right_TE (to get slope for right side adjustment)
        - Temperature ~ Left_TE (to get slope for left side adjustment)
        
        Note: We regress Temperature on TE (not TE on Temperature) to get dT/d(TE),
        which is the reciprocal of d(TE)/dT needed for temperature correction.
        """
        # Calculate linear parameters for Right TE vs Temperature
        # stats.linregress(x, y) returns: slope, intercept, r_value, p_value, std_err
        self.right_slope, self.right_intercept, self.right_r_value, self.right_p_value, self.right_std_err = \
            stats.linregress(right_te, temperatures)
        
        # Calculate linear parameters for Left TE vs Temperature
        self.left_slope, self.left_intercept, self.left_r_value, self.left_p_value, self.left_std_err = \
            stats.linregress(left_te, temperatures)
        
        self.fitted = True
        print("Fitting completed successfully")
    
    def _load_params(self, load_path: str):
        """
        Internal method: Load parameters from JSON file.
        
        Args:
            load_path: Path to JSON file containing saved linear parameters.
            
        Raises:
            FileNotFoundError: If parameter file does not exist.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Parameter file not found: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            params: Dict[str, Any] = json.load(f)
        
        # Load Right TE parameters
        self.right_slope = params['right_slope']
        self.right_intercept = params['right_intercept']
        self.right_r_value = params.get('right_r_value', None)
        self.right_p_value = params.get('right_p_value', None)
        self.right_std_err = params.get('right_std_err', None)
        
        # Load Left TE parameters
        self.left_slope = params['left_slope']
        self.left_intercept = params['left_intercept']
        self.left_r_value = params.get('left_r_value', None)
        self.left_p_value = params.get('left_p_value', None)
        self.left_std_err = params.get('left_std_err', None)
        
        self.fitted = True
        print(f"Parameters loaded from {load_path}")
    
    def save_params(self, save_path: str):
        """
        Save linear regression parameters to JSON file.
        
        Args:
            save_path: Target JSON file path.
            
        Raises:
            RuntimeError: If model has not been fitted yet.
        """
        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Model not fitted yet, cannot save parameters")
        
        # Ensure output directory exists
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        params = {
            'right_slope': float(self.right_slope),
            'right_intercept': float(self.right_intercept),
            'right_r_value': float(self.right_r_value) if self.right_r_value is not None else None,
            'right_p_value': float(self.right_p_value) if self.right_p_value is not None else None,
            'right_std_err': float(self.right_std_err) if self.right_std_err is not None else None,
            'left_slope': float(self.left_slope),
            'left_intercept': float(self.left_intercept),
            'left_r_value': float(self.left_r_value) if self.left_r_value is not None else None,
            'left_p_value': float(self.left_p_value) if self.left_p_value is not None else None,
            'left_std_err': float(self.left_std_err) if self.left_std_err is not None else None
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4, ensure_ascii=False)
        
        print(f"Parameters saved to {save_path}")
    
    def adjust_values(
        self, 
        new_temperatures: Union[pd.Series, np.ndarray, List], 
        new_right_te: Union[pd.Series, np.ndarray, List], 
        new_left_te: Union[pd.Series, np.ndarray, List], 
        specified_temperature: float
    ):
        """
        Adjust measured TE values to a specified reference temperature.
        
        Correction formula based on linear approximation:
        If Temperature = slope * TE + intercept,
        then d(TE)/d(Temp) = 1/slope,
        therefore: TE_corrected = TE_original + (T_target - T_measured) / slope
        
        Args:
            new_temperatures: Array of measured room temperatures.
            new_right_te: Array of measured Right TE values.
            new_left_te: Array of measured Left TE values.
            specified_temperature: Target reference temperature (e.g., 23°C).
            
        Returns:
            Tuple of (adjusted_right_te, adjusted_left_te, target_temp_array)
        """
        # Calculate temperature deviation from target
        temp_diff = specified_temperature - new_temperatures
        
        # Adjust Right TE: correction = temp_diff / slope
        right_correction = temp_diff / self.right_slope
        adjusted_right_te = new_right_te + right_correction

        # Adjust Left TE
        left_correction = temp_diff / self.left_slope
        adjusted_left_te = new_left_te + left_correction

        return adjusted_right_te, adjusted_left_te, [specified_temperature] * len(adjusted_left_te)


# Example usage
if __name__ == "__main__":
    # # ========== Example 1: Train from Excel and save parameters ==========

    # te_adjuster = TEAdjuster(
    #     data_path=r"Absolute data path, e.g. original_data.xlsx",
    #     save_path=r"parm_single.json"
    # )

    
    # ========== Example 2: Load existing parameters and process data ==========
    adjuster = TEAdjuster(
        load_path=r"parm_single.json"
    )
    
    # Load data to be processed
    data = pd.read_excel(
        r"Your absolute data path, e.g. original_data.xlsx"
    )

    # Filter data where temperature exceeds threshold (e.g., > 25°C)
    mask = data['room_temperature'] > 25
    new_temperatures = data.loc[mask, 'room_temperature']
    new_right_te = data.loc[mask, 'Right_final']
    new_left_te = data.loc[mask, 'Left_final']

    # Adjust to specified temperature (e.g., standard 23°C)
    adjusted_right, adjusted_left, adjusted_temp = adjuster.adjust_values(
        new_temperatures, new_right_te, new_left_te, 23
    )

    # Update dataframe with adjusted values
    data.loc[mask, 'Right_final'] = adjusted_right
    data.loc[mask, 'Left_final'] = adjusted_left
    data.loc[mask, 'room_temperature'] = adjusted_temp
    data.loc[mask, 'Difference'] = adjusted_right - adjusted_left

    # Save results
    output_path = r"data_reflect_single.xlsx"
    data.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")