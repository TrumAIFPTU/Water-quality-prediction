import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def evaluate_model(model, loader, dataset, device):
    """Evaluate model"""
    model.eval()
    preds, actuals, events = [], [], []

    with torch.no_grad():
        for bx, by, bevent in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            preds.append(dataset.denormalize(out).cpu().numpy())
            actuals.append(dataset.denormalize(by).cpu().numpy())
            events.append(bevent.cpu().numpy())

    if len(preds) == 0: 
        return {}

    preds = np.concatenate(preds).flatten()
    actuals = np.concatenate(actuals).flatten()
    events = np.concatenate(events).flatten()

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-6))) * 100

    event_indices = np.where(events == 1)[0]
    if len(event_indices) > 0:
        sf_mae = mean_absolute_error(actuals[event_indices], preds[event_indices])
    else:
        sf_mae = 0.0

    return {
        "RMSE": rmse, 
        "MAE": mae, 
        "R2": r2, 
        "MAPE": mape,
        "SF_MAE": sf_mae,
        "N_Events": len(event_indices)
    }
