import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score
)

@dataclass
class RegressionMetrics:
    """Container for regression metrics"""
    r2: float
    mse: float
    rmse: float
    mae: float
    explained_variance: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'r2': self.r2,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'explained_variance': self.explained_variance
        }
    
    def __str__(self) -> str:
        """String representation of metrics"""
        return (
            f"R² Score: {self.r2:.4f}\n"
            f"MSE: {self.mse:.4f}\n"
            f"RMSE: {self.rmse:.4f}\n"
            f"MAE: {self.mae:.4f}\n"
            f"Explained Variance: {self.explained_variance:.4f}"
        )

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> RegressionMetrics:
    """
    Compute regression metrics
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    sample_weight : np.ndarray, optional
        Sample weights
        
    Returns:
    --------
    RegressionMetrics
        Container with computed metrics
    """
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    
    return RegressionMetrics(
        r2=r2_score(y_true, y_pred, sample_weight=sample_weight),
        mse=mse,
        rmse=np.sqrt(mse),
        mae=mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
        explained_variance=explained_variance_score(
            y_true, y_pred, sample_weight=sample_weight
        )
    ) 