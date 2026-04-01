import math

def sigmoid_weight(y: float, threshold: float = 10.0, scale: float = 2.0) -> float:
    """
    Calculate sigmoid-based transition weight.
    
    Uses logistic function to create smooth weighting between 0 and 1,
    transitioning around the threshold value.
    
    Parameters:
    -----------
    y : float
        Input value (typically model prediction)
    threshold : float, default 10.0
        Center point of transition (inflection point)
    scale : float, default 2.0
        Steepness factor, smaller values = sharper transition
        
    Returns:
    --------
    float : Weight between 0 and 1
            - Approaches 1 when y << threshold
            - 0.5 when y == threshold  
            - Approaches 0 when y >> threshold
    """
    return 1.0 / (1.0 + math.exp((y - threshold) / scale))


def ensemble_predict(mlp_pred: float, xgb_pred: float) -> float:
    """
    Ensemble prediction with smooth transition logic.
    
    Combines MLP and XGBoost predictions using a smooth transition strategy:
    - Low power regime (< 5): Trust MLP (typically better at low values)
    - High power regime (> 10): Trust XGB (typically better at high values)
    - Transition regime: Weighted combination using sigmoid interpolation
    
    Parameters:
    -----------
    mlp_pred : float
        Prediction from MLP model (Multi-Layer Perceptron)
    xgb_pred : float
        Prediction from XGBoost model (eXtreme Gradient Boosting)
        
    Returns:
    --------
    float : Final ensemble prediction
        
    Notes:
    ------
    Weight calculation uses asymmetric logic:
    - MLP weight based on MLP's own prediction value
    - XGB weight based on XGB's own prediction value
    This allows each model to "know" when it should contribute.
    """
    # Low power regime: MLP dominates (typically more accurate below threshold)
    if mlp_pred < 5.0:
        return mlp_pred
    
    # High power regime: XGB dominates (typically more accurate above threshold)
    elif xgb_pred > 10.0:
        return xgb_pred
    
    # Transition regime: smooth weighted combination
    else:
        # Calculate weights for each model
        w_mlp = sigmoid_weight(mlp_pred)      # MLP weight decreases as mlp_pred increases
        w_xgb = 1.0 - sigmoid_weight(xgb_pred)  # XGB weight increases as xgb_pred increases
        
        # Weighted ensemble (note: w_mlp + w_xgb may not equal 1.0 due to asymmetric design)
        return mlp_pred * w_mlp + xgb_pred * w_xgb


