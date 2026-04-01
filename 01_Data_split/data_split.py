import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration paths
# Automatically use original_data.xlsx in the same folder as the script
input_path = os.path.join(script_dir, "original_data.xlsx")
output_dir = script_dir  # Save output to the same directory

# Check if input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}\nPlease ensure 'original_data.xlsx' exists in the script directory.")

# Load raw data from Excel
df = pd.read_excel(input_path)
print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

# Split data into training and testing sets
# test_size=0.15 means 15% for test, 85% for train
# random_state=42 ensures reproducibility
train_df, test_df = train_test_split(
    df, 
    test_size=0.15, 
    random_state=42
)

print(f"Training set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
train_file = os.path.join(output_dir, f"train_split_{timestamp}.xlsx")
test_file = os.path.join(output_dir, f"test_split_{timestamp}.xlsx")

# Save split datasets to Excel files
train_df.to_excel(train_file, index=False)
test_df.to_excel(test_file, index=False)

print(f"\nFiles saved:")
print(f"  Training set -> {train_file}")
print(f"  Testing set -> {test_file}")