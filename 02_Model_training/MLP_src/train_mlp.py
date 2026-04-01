import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import copy

# -------------------------------
# 1. Define Configurable MLP Model 
# -------------------------------
class FlexibleMLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP) model with flexible hidden layer architecture.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    hidden_sizes : list
        List defining the number of neurons in each hidden layer
    use_bn : bool
        Whether to use Batch Normalization after each linear layer
    activation : nn.Module
        Activation function class (default: nn.ReLU)
    output_size : int
        Number of output neurons (default: 1 for regression)
    """
    def __init__(
        self,
        input_size,
        hidden_sizes=[128] + [256] * 16 + [128],
        use_bn=True,
        activation=nn.ReLU,
        output_size=1
    ):
        super(FlexibleMLP, self).__init__()
        
        # Save configuration parameters for model reconstruction
        self.config = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'use_bn': use_bn,
            'activation': activation.__name__,
            'output_size': output_size
        }
        
        layers = []
        in_dim = input_size
        
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation())
            in_dim = h
        
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

# -------------------------------
# 2. Data Preparation Function
# -------------------------------
def prepare_data(path, test_size=0.2, random_state=42):
    """
    Prepare training and testing data from Excel file.
    
    Loads data, applies log10 transformation to target variable, 
    splits into train/test sets, and standardizes features.
    
    Parameters:
    -----------
    path : str
        Path to the Excel data file
    test_size : float
        Proportion of dataset to include in test split (0-1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, 
             scaler, X_all_tensor, y_all_log, X_data)
        - X_train_tensor: Training features as torch tensor
        - X_test_tensor: Testing features as torch tensor  
        - y_train_tensor: Training targets (log10 transformed)
        - y_test_tensor: Testing targets (log10 transformed)
        - scaler: Fitted StandardScaler object
        - X_all_tensor: All features as torch tensor
        - y_all_log: All targets as numpy array (log10 transformed)
        - X_data: Original feature DataFrame
    """
    # Load data
    data = pd.read_excel(path)
    X = data[['Right_final', 'Left_final', 'Difference', 'room_temperature']]
    y = np.log10(data['P1(uW)'])  # Apply log10 transformation to target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)
    
    # Process all data for final predictions
    X_all_scaled = scaler.transform(X)
    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)
    y_all_log = y.values
    
    return (X_train_tensor, X_test_tensor, 
            y_train_tensor, y_test_tensor, 
            scaler, X_all_tensor, y_all_log, X)

# -------------------------------
# 3. Model Training Function
# -------------------------------
def train_mlp_model(
    X_train, y_train, X_test, y_test,
    hidden_sizes=[128] + [256] * 16 + [128],
    lr=0.0003, epochs=1000, patience=80
):
    """
    Train MLP model with early stopping.
    
    Parameters:
    -----------
    X_train, y_train : torch.Tensor
        Training data tensors
    X_test, y_test : torch.Tensor
        Testing data tensors for validation
    hidden_sizes : list
        Hidden layer architecture (list of neuron counts)
    lr : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum number of training epochs
    patience : int
        Early stopping patience (epochs without improvement)
        
    Returns:
    --------
    tuple : (model, train_losses, test_losses)
        - model: Trained model with best weights loaded
        - train_losses: List of training losses per epoch
        - test_losses: List of testing losses per epoch
    """
    # Initialize model
    input_size = X_train.shape[1]
    model = FlexibleMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        output_size=1
    )
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    best_test_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    print(f"Starting MLP training, hidden layer structure: {hidden_sizes}")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        # Update best model
        if test_loss < best_test_loss - 1e-9:
            best_test_loss = test_loss
            no_improve_count = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_count += 1
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Test Loss: {test_loss.item():.4f}')
        
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Best model weights loaded.")
    
    return model, train_losses, test_losses

# -------------------------------
# 4. Save Model Function
# -------------------------------
def save_model(model, scaler, X_data, save_dir):
    """
    Save trained model.
    
    Saves model state dict with configuration and scaler.
    
    Parameters:
    -----------
    model : FlexibleMLP
        Trained model
    scaler : StandardScaler
        Fitted scaler for feature standardization
    X_data : pd.DataFrame
        Original feature DataFrame
    save_dir : str
        Directory path for saving files
        
    Returns:
    --------
    str : Path to saved model checkpoint (.pth)
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model checkpoint (state dict + config + scaler)
    model_file = os.path.join(save_dir, f'mlp_regression_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'scaler': scaler,
        'input_size': X_data.shape[1]
    }, model_file)
    
    print(f"Model saved to: {model_file}")
    
    return model_file

# -------------------------------
# 5. Training Visualization Function
# -------------------------------
def visualize_training(model, X_test, y_test, train_losses, test_losses, save_dir):
    """
    Visualize training process and prediction results.
    
    Creates a figure with two subplots: loss curves and predictions vs actual scatter plot.
    
    Parameters:
    -----------
    model : FlexibleMLP
        Trained model
    X_test : torch.Tensor
        Test features
    y_test : torch.Tensor
        Test targets (log scale)
    train_losses : list
        Training loss history
    test_losses : list
        Testing loss history
    save_dir : str
        Directory to save the plot
        
    Returns:
    --------
    str : Path to saved plot file
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # Prediction scatter plot
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    plt.subplot(1, 2, 2)
    plt.scatter(y_test.numpy(), predictions, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values (log scale)')
    plt.ylabel('Predictions (log scale)')
    plt.title('Predictions vs Actual Values')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"training_plot_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved to: {plot_path}")
    
    return plot_path

# -------------------------------
# 6. Load Model Function
# -------------------------------
def load_mlp_model(model_path):
    """
    Load a saved MLP model from checkpoint file.
    
    Reconstructs model architecture from saved configuration and loads weights.
    
    Parameters:
    -----------
    model_path : str
        Path to the model checkpoint file (.pth)
        
    Returns:
    --------
    tuple : (model, scaler)
        - model: Loaded model in eval mode
        - scaler: Fitted StandardScaler
        
    Raises:
    -------
    FileNotFoundError: If model file does not exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Activation function mapping
    activation_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh
    }
    
    # Get model configuration from checkpoint
    config = checkpoint['model_config']
    
    # Reconstruct model using saved configuration
    model = FlexibleMLP(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        use_bn=config['use_bn'],
        activation=activation_map.get(config['activation'], nn.ReLU),
        output_size=config['output_size']
    )
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get scaler
    scaler = checkpoint['scaler']
    
    print(f"Model successfully loaded: {model_path}")
    return model, scaler

# -------------------------------
# 7. Prediction Function
# -------------------------------
def predict_with_model(model, scaler, input_data):
    """
    Make predictions using trained model.
    
    Accepts DataFrame or dict input, applies standardization, 
    and returns predictions in original power units (inverse log10).
    
    Parameters:
    -----------
    model : FlexibleMLP
        Trained model
    scaler : StandardScaler
        Scaler fitted on training data
    input_data : pd.DataFrame or dict
        Input features with columns: Right_final, Left_final, Difference, room_temperature
        
    Returns:
    --------
    np.array : Predictions in original power units (uW)
    
    Raises:
    -------
    TypeError: If input_data is not DataFrame or dict
    ValueError: If required columns are missing
    """
    # Convert dict to DataFrame if necessary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise TypeError("Input data must be DataFrame or dictionary")
    
    # Ensure correct column order
    required_columns = ['Right_final', 'Left_final', 'Difference', 'room_temperature']
    if not all(col in input_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in input_df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Preprocess
    X_input = scaler.transform(input_df[required_columns])
    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    
    # Predict
    model.eval()
    with torch.no_grad():
        log_predictions = model(X_tensor)
        predictions = 10**log_predictions.numpy().flatten()
    
    return predictions

# -------------------------------
# 8. Complete Training Pipeline
# -------------------------------
def train(data, model_dir, test_size=0.2, random_state=42, hidden_sizes=[128] + [256] * 16 + [128], lr=0.0003, epochs=1000, patience=80):
    """
    End-to-end training pipeline.
    
    Orchestrates data preparation, model training, visualization, 
    and model saving in one workflow.
    
    Parameters:
    -----------
    data : str, pd.DataFrame, or np.ndarray
        Path to Excel data file, pandas DataFrame, or numpy array
    save_dir : str
        Directory for saving model
    test_size : float
        Test set proportion (0-1)
    random_state : int
        Random seed for reproducibility
    hidden_sizes : list
        Hidden layer architecture (list of neuron counts)
    lr : float
        Learning rate for Adam optimizer
    epochs : int
        Maximum number of training epochs
    patience : int
        Early stopping patience (epochs without improvement)
        
    Returns:
    --------
    dict : Training results containing model, scaler, and file path
    """
    print("="*50)
    print("Start training ...")
    print("="*50)
    
    # 1. Prepare data
    print("\n[load] Preparing data...")
    (X_train, X_test, y_train, y_test, 
     scaler, X_all, y_all_log, X_data) = prepare_data(
        data, test_size=test_size, random_state=random_state
    )
    
    # 2. Train model
    print("\n[train] Starting training with train set...")
    model, train_losses, test_losses = train_mlp_model(
        X_train, y_train, X_test, y_test, 
        hidden_sizes=hidden_sizes, lr=lr, epochs=epochs, patience=patience
    )
    
    # 3. Visualize training
    print("\n[train] Training with quick view ...")
    plot_path = visualize_training(
        model, X_test, y_test, 
        train_losses, test_losses, model_dir
    )
    
    # 4. Save model
    print("\n[Save] Saving model...")
    model_file = save_model(
        model, scaler, X_data, model_dir
    )
    
    # 5. Return results
    print("\n[Train] Training completed!")
    print("="*50)
    
    return {
        'model': model,
        'scaler': scaler,
        'model_file': model_file,
        'plot_path': plot_path
    }

# -------------------------------
# 9. Inference Pipeline
# -------------------------------
def inference(model_path, 
              data, 
              output_dir=None):
    """
    Inference pipeline using saved model.
    
    Loads model, processes input data (file path, DataFrame, or dict),
    performs prediction, and optionally saves results.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model checkpoint
    data : str, pd.DataFrame, or dict
        Input data - can be file path (CSV/Excel), DataFrame, or dictionary
    save_dir : str, optional
        Directory to save prediction results
        
    Returns:
    --------
    pd.DataFrame : Input data with added 'Predicted_Power(uW)' column
    """
    print("="*50)
    print("Prediction processing")
    print("="*50)
    
    # 1. Load model
    print("\nStep 1/3: Loading model...")
    model, scaler = load_mlp_model(model_path)
    
    # 2. Prepare input data
    print("\nStep 2/3: Loading data...")
    if isinstance(data, str):
        # Read from file
        if data.endswith('.csv'):
            input_df = pd.read_csv(data)
        else:  
            input_df = pd.read_excel(data)
    elif isinstance(data, (dict, pd.DataFrame)):
        input_df = pd.DataFrame(data) if isinstance(data, dict) else data
    else:
        raise TypeError("Input data must be file path, dictionary, or DataFrame")
    
    # 3. Perform prediction
    print("\nStep 3/3: Processing prediction...")
    predictions = predict_with_model(model, scaler, input_df)
    
    # Add predictions to dataframe
    input_df['Predicted_Power(uW)'] = predictions
    
    # Optional: Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        pred_file = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        input_df.to_csv(pred_file, index=False)
        print(f"\nPredictions saved to: {pred_file}")
    
    print("\nPrediction completed!")
    print("="*50)
    
    return input_df


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="MLP Training")
    parser.add_argument("--data", required=True, help="path to input file (.xlsx or .csv) or directly pd.DataFrame/np.ndarray")
    parser.add_argument("--model_dir", default="./model", help="directory to save model")
    parser.add_argument("--test_size", type=float, default=0.2, help="test set proportion (0-1)")
    parser.add_argument("--random_state", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--hidden_sizes", nargs="+", type=int, default=[128] + [256] * 16 + [128], help="hidden layer sizes")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=80, help="early stopping patience")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    training_results = train(
        data=args.data, 
        model_dir=args.model_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience
    )
    
