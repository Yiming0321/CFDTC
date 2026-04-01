

# SI-Code_and_Dataset 

## Project Overview

This project is a regression prediction system based on multiple machine learning algorithms, primarily used for predicting power values (P1(uW)). The project includes modules for data splitting, multiple model training, model evaluation, smooth transfer, and TE adjuster, providing a complete workflow from data processing to model deployment.

### Project Capabilities

| Module | Functionality | Description |
|--------|---------------|-------------|
| Data Splitting | Data partitioning | Splits original data into training and testing sets with timestamps |
| Model Training | Multiple algorithm support | Supports Linear Regression, MLP, Random Forest, XGBoost, and LightGBM models |
| Model Evaluation | Performance assessment | Evaluates models using MSE, RMSE, MAE, and MAPE metrics |
| Smooth Transfer | Ensemble prediction | Combines predictions from multiple models for more robust results |
| TE Adjuster | Temperature adjustment | Adjusts TE values to a specified reference temperature |

### Workflow

1. **Data Preparation**: Prepare original data with required columns
2. **Data Splitting**: Split data into training and testing sets
3. **Model Training**: Train one or multiple models using the training set
4. **Model Evaluation**: Evaluate model performance on the testing set
5. **Model Inference**: Use trained models to make predictions
6. **Smooth Transfer** (Optional): Combine predictions from multiple models
7. **TE Adjustment** (Optional): Adjust TE values to reference temperature

This workflow provides a comprehensive solution for power value prediction, from data preparation to model deployment, with flexibility to choose specific components based on requirements.

## Directory Structure

```
SI-Code_Dataset/
├── 01_Data_split/            # Data splitting module
│   ├── data_split.py         # Data splitting script
│   └── original_data.xlsx    # Original data
├── 02_Model_training/        # Model training module
│   ├── Linear_Model_src/     # Linear regression model
│   ├── MLP_src/              # Multi-layer perceptron model
│   ├── Random_Forest_src/    # Random Forest model
│   ├── XGboost_src/          # XGBoost model
│   └── lightGBM_src/         # LightGBM model
├── 03_Model_evaluation/      # Model evaluation module
│   └── evaluate.py           # Model evaluation script
├── 04_Smooth_transfer/       # Smooth transfer module
│   ├── transfer.py           # Smooth transfer script
│   └── inference.py          # Ensemble inference script
└── 05_TE_adjustor/           # TE adjuster module
    ├── TEAdjuster_multi.py    # Multi TE adjuster
    ├── TEAdjuster_single.py   # Single TE adjuster
    └── TEAdjuster_runner.py   # TE adjuster runner
```

## Installation Instructions

### Dependencies

This project requires the following Python libraries:

- numpy>=1.21.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- torch>=1.9.0 (for MLP model)
- lightgbm>=3.3.0
- xgboost>=1.5.0
- joblib>=1.1.0
- matplotlib>=3.5.0
- openpyxl>=3.0.0

### Installation Command

You can install all dependencies at once using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies individually:

```bash
pip install 'numpy>=1.21.0' 'pandas>=1.3.0' 'scipy>=1.7.0' 'scikit-learn>=1.0.0' 'torch>=1.9.0' 'lightgbm>=3.3.0' 'xgboost>=1.5.0' 'joblib>=1.1.0' 'matplotlib>=3.5.0' 'openpyxl>=3.0.0'
```

## Usage

### 1. Dataset and Splitting

#### 1.1 Dataset definition
The dataset is used for training and evaluating regression models to predict power values (P1(uW)). The original dataset should contain at least the following five columns:

| Column Name | Description | Data Type |
|------------|-------------|-----------|
| Right_final | Right channel signal | float |
| Left_final | Left channel signal | float |
| Difference | Signal difference | float |
| room_temperature | Room temperature during testing | float |
| P1(uW) | Heat release power | float |


#### 1.2 Dataset splitting

use the `data_split.py` script to split the original data into training and testing sets:

```bash
cd 01_Data_split
python data_split.py
```

This will generate training and testing set files with timestamps in the same directory.

### 2. Model Training and Inference

#### 2.1 Linear Regression Model

- Supports 1-6 degree polynomial fitting
- Saves model parameters in JSON format
- Supports input as file path or pandas DataFrame

##### Training
Training model with cmd command
```bash
cd 02_Model_training/Linear_Model_src
# Training for specific degree polynomial fitting
python train_linear.py --data ../../01_Data_split/train_split_*.xlsx --degree 3 --model_dir  ../model
# Training for degree range polynomial fitting
python train_linear.py --data ../../01_Data_split/train_split_*.xlsx --degree_range 1 5 --model_dir  ../model
```
Training model with python script
```python
from train_linear import train
import pandas as pd

# Example 1: Train with specific degree
train(
    data=r"path/to/your/data.xlsx",
    model_dir="../model",
    degree=3,  # Specify polynomial degree
)
```

##### Inference
Inference model with cmd command
```bash
cd 02_Model_training/Linear_Model_src
# inference for specific degree polynomial fitting
python infer_linear.py --data ../../01_Data_split/test_split_*.xlsx --model ../model/poly_degree_3_*.json --output ../output
# inference for degree range polynomial fitting
python infer_linear.py --data ../../01_Data_split/test_split_*.xlsx --model ../model/poly_degree_range_*.json --output ../output
```
Inference model with python script
```python
from infer_linear import inference
import pandas as pd

# Example 1: Using file path
inference(
    data=r"path/to/your/data.xlsx",
    model_path ="./model/poly_degree_3_*.json",
    output_dir="./output"
)

# Example 2: Using pandas DataFrame
df = pd.read_excel(r"path/to/your/data.xlsx")
inference(
    data=df,
    model_path ="./model/poly_degree_3_*.json",
    output_dir="./output"
)
```


#### 2.2 MLP Model

- Configurable multi-layer perceptron neural network
- Supports batch normalization and different activation functions
- Uses Adam optimizer and early stopping mechanism
- Applies log10 transformation to target variable to improve performance
- Supports input as file path, pandas DataFrame, or numpy array

##### Training
Training model with cmd command
```bash
cd 02_Model_training/MLP_src
# Example 1: Default parameters
python train_mlp.py --data ../../01_Data_split/train_split_*.xlsx --model_dir ./model

# Example 2: Custom parameters
python train_mlp.py --data ../../01_Data_split/train_split_*.xlsx --model_dir ./model  --hidden_sizes 64 128 64  --lr 0.001  --epochs 500  --patience 50
```
training model with python script
```python
from train_mlp import train

training_results = train(
    data=r"path/to/your/data.xlsx",
    model_dir=r"path/to/output/directory",
    test_size=0.2,
    random_state=42
)
```

##### Inference
Inference model with cmd command
```bash
cd 02_Model_training/MLP_src
python infer_mlp.py --model model/mlp_regression_*.pth --data ../../01_Data_split/test_split_*.xlsx --output_dir ./output
```
Inference model with python script
```python
from train_mlp import inference

prediction_results = inference(
    model_path=r"path/to/model/mlp_regression_*.pth",
    data=r"path/to/your/data.xlsx",
    output_dir=r"path/to/output"
)
```


#### 2.3 Random Forest Model

- Ensemble learning method based on decision trees
- Supports parallel computing to improve training speed
- Does not require feature standardization
- Supports input as file path, pandas DataFrame, or numpy array

##### Training
Training model with cmd command
```bash
cd 02_Model_training/Random_Forest_src
python train_RF.py --data ../../01_Data_split/train_split_*.xlsx
```
training model with python script
```python
from train_RF import train

train(
    data=r"path/to/your/data.xlsx",
    model_dir=r"path/to/output/directory",
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    feat_cols=["Right_final", "Left_final", "Difference", "room_temperature"],
    target_col="P1(uW)"
)

```

##### Inference
Inference model with cmd command
```bash
cd 02_Model_training/Random_Forest_src
python infer_RF.py -d ../../01_Data_split/test_split_*.xlsx --model models/rf_model_*.pkl --output  ./output
```
Inference model with python script
```python
from infer_RF import inference

prediction_results = inference(
    model_path=r"path/to/models/rf_model_*.pkl",
    data=r"path/to/your/data.xlsx",
    output_dir=r"path/to/output/directory"
)

```


#### 2.4 XGBoost Model

- Efficient algorithm based on gradient boosting trees
- Supports cross-validation and early stopping mechanism
- Automatically handles missing values
- Supports input as file path, pandas DataFrame, or numpy array

##### Training
Training model with cmd command
```bash
cd 02_Model_training/XGboost_src
python train_xgb.py --data ../../01_Data_split/train_split_*.xlsx --model_dir ./model \
    --max_depth 8 --eta 0.05 --subsample 0.9 \
    --colsample_bytree 0.9 --num_round 300 \
    --early_stop 20 --seed 42 \
    --feat_cols Right_final Left_final Difference room_temperature \
    --label_col P1(uW)
```
training model with python script
```python
from train_xgb import train

train(
    data=r"path/to/your/data.xlsx",
    model_dir=r"path/to/output/directory",
    max_depth=50,
    eta=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    num_round=100,
)

```

##### Inference
Inference model with cmd command
```bash
cd 02_Model_training/XGboost_src
python infer_xgb.py --model model/xgb_model_*.json --data ../../01_Data_split/test_split_*.xlsx --output ./output
```
Inference model with python script
```python
from infer_xgb import inference

prediction_results = inference(
    model_path=r"path/to/model/xgb_model_*.json",
    data=r"path/to/your/data.xlsx",
    output_dir=r"path/to/output/directory"
)
```

#### 2.5 LightGBM Model

- Efficient algorithm based on gradient boosting trees
- Uses leaf-wise splitting strategy for faster training
- Supports parallel computing and GPU acceleration

##### Training
Training model with cmd command
```bash
cd 02_Model_training/lightGBM_src
python train_LightGBM.py --data ../../01_Data_split/train_split_*.xlsx
```
training model with python script
```python
from train_LightGBM import train

# Example: Train LightGBM model
train(
    data=r"path/to/your/data.xlsx",
    model_dir=r"path/to/output/directory",
    feat_cols=["Right_final", "Left_final", "Difference", "room_temperature"],
    target_col="P1(uW)",
    max_depth=50,
    num_leaves=300,
    n_estimators=2000,
    nfold=5,
    seed=42
)
```

##### Inference
Inference model with cmd command
```bash
cd 02_Model_training/lightGBM_src
python infer_LightGBM.py -d ../../01_Data_split/test_split_*.xlsx --model ./model/lightgbm_regression_*.pkl --output ./output
```
Inference model with python script
```python
from infer_LightGBM import inference

prediction_results = inference(
    model_path=r"path/to/model/lightgbm_regression_*.pkl",
    data=r"path/to/your/data.xlsx",
    output_dir=r"path/to/output/directory"
)
```


### 3. Model Evaluation

Use the `evaluate.py` script to evaluate model performance:

```bash
cd 03_Model_evaluation
python evaluate.py -i predictions.xlsx -o metrics.xlsx
```

## Evaluation Metrics

This project uses the following evaluation metrics:

- **MSE** (Mean Squared Error): Average of squared differences between predicted and actual values
- **RMSE** (Root Mean Squared Error): Square root of MSE
- **MAE** (Mean Absolute Error): Average of absolute differences between predicted and actual values
- **MAPE** (Mean Absolute Percentage Error): Average of absolute percentage differences between predicted and actual values


### 4. Smooth Transfer

The Smooth Transfer module combines predictions from multiple models to create a more robust and accurate prediction. It currently supports ensemble prediction using XGBoost and MLP models.

#### 4.1 Ensemble Inference

The `inference.py` script runs both XGBoost and MLP models on the same input data and then combines their predictions using the `ensemble_predict` function.

```bash
cd 04_Smooth_transfer
python inference.py --xgb_model path/to/xgb_model.json --mlp_model path/to/mlp_model.pth --data path/to/data.xlsx --output path/to/output.xlsx
```

Example:

```bash
cd 04_Smooth_transfer
python inference.py --xgb_model ../02_Model_training/XGboost_src/model/xgb_model_20260401.json --mlp_model ../02_Model_training/MLP_src/model/mlp_regression_20260401.pth --data ../01_Data_split/test_split_20260401.xlsx --output ensemble_results.xlsx
```


#### 4.2 Transfer Function

The `transfer.py` script contains the `ensemble_predict` function that implements the smooth transfer logic between models. This function is used by the `inference.py` script.

##### Function Details

The `ensemble_predict` function takes predictions from MLP and XGBoost models and combines them using a weighted average based on the prediction values. This helps to leverage the strengths of both models and create a more balanced prediction.


### 5. TE Adjuster

The TE (Thermoelectric) Adjuster is used to adjust TE values to a specified reference temperature. The project provides two implementations and a unified runner script for easy usage.

#### TEAdjuster_runner 

The `TEAdjuster_runner.py` script provides a unified interface to run either the single or multi (category-based) TE adjuster. It supports both training mode (to generate adjustment parameters) and adjustment mode (to apply adjustments to data).

### Usage

#### Training Mode

```bash
# Get into TE_adjustor directory
cd 05_TE_adjustor
# Train single adjuster
python TEAdjuster_runner.py --mode single --data_path original_data.xlsx --save_path parm_single.json

# Train multi adjuster
python TEAdjuster_runner.py --mode multi --data_path original_data.xlsx --save_path parm.json
```

#### Adjustment Mode

```bash
# Adjust data using single adjuster
python TEAdjuster_runner.py --mode single --parm_path parm_single.json --data data-out_of_limit.xlsx --output output_single.xlsx

# Adjust data using multi adjuster
python TEAdjuster_runner.py --mode multi --parm_path parm.json --data data-out_of_limit.xlsx --output output_multi.xlsx
```

### Parameters

- `--mode`: Adjuster mode, either `single` (simple linear) or `multi` (category-based)
- `--data_path`: Path to training data file (for training mode)
- `--save_path`: Path to save parameters (for training mode)
- `--parm_path`: Path to load parameters (for adjustment mode)
- `--data`: Path to data file to process (for adjustment mode)
- `--output`: Path to save output file (for adjustment mode)
- `--temp_threshold`: Temperature threshold for adjustment (default: 25.0°C)
- `--temp_target`: Target temperature for adjustment (default: 23.0°C)





## Notes

1. Ensure the original data file `original_data.xlsx` exists in the `01_Data_split` directory
2. Model training will generate model files and result files, please ensure the target directory has write permissions
3. For MLP model, sufficient computing resources are required, especially when the hidden layers are deep
4. Model parameters can be adjusted based on actual data to achieve better performance

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact the project maintainer.