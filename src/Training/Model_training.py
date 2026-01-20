import time 
import torch
import numpy as np
import torch.nn as nn
from src.Parameters.parameter import RUNTIME_LOG

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, model_name="Model"):
    """⏱️ Training loop with RUNTIME TRACKING"""
    best_val_loss = float('inf')

    epoch_times = []

    for epoch in range(epochs):
        # ⏱️ START EPOCH TIMER
        epoch_start = time.time()

        model.train()
        train_loss = 0

        for bx, by, bevent in train_loader:
            bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by, bevent)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, bevent in val_loader:
                bx, by, bevent = bx.to(device), by.to(device), bevent.to(device)
                pred = model(bx)
                loss = criterion(pred, by, bevent)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # ⏱️ END EPOCH TIMER
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # ⏱️ LOG AVERAGE TRAINING TIME
    avg_epoch_time = np.mean(epoch_times)
    RUNTIME_LOG.append({
        "Stage": "Training",
        "Model": model_name,
        "Time_s": avg_epoch_time,
        "Unit": "s/epoch"
    })

    return model

def train_model_simple(model, train_loader, val_loader, criterion, optimizer, 
                      epochs, device, model_name="Model"):
    """Training loop for standard MSE (with runtime)"""
    best_val_loss = float('inf')
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0

        for bx, by, bevent in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by, bevent in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    avg_epoch_time = np.mean(epoch_times)
    RUNTIME_LOG.append({
        "Stage": "Training",
        "Model": model_name,
        "Time_s": avg_epoch_time,
        "Unit": "s/epoch"
    })

    return model