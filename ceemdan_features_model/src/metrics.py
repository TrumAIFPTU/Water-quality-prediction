import numpy as np
from sklearn.metrics import r2_score

def metric(pred, true):
    # Mean Absolute Error
    MAE = np.mean(np.abs(pred - true))
    
    # Root Mean Squared Error
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    
    # Mean Absolute Percentage Error (thêm epsilon để tránh chia 0)
    MAPE = np.mean(np.abs((pred - true) / (true + 1e-7))) * 100
    
    # R-squared
    R2 = r2_score(true.flatten(), pred.flatten())
    
    return MAE, MAPE, RMSE, R2