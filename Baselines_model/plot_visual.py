import matplotlib.pyplot as plt
import pandas as pd
import os
from src.path import SERIES_DIR,ROOT_DIR
from pathlib import Path
def plot_prediction(file_path,save_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(15, 5))
    
    # Vẽ 200 điểm đầu tiên để xem chi tiết
    plt.plot(df['Actual'], label='Actual', color='blue', linewidth=1.5)
    plt.plot(df['Predicted'], label='Predicted', color='red', linestyle='--', linewidth=1.2)
    
    plt.title(f"Comparison: {Path(file_path).stem}")
    plt.xlabel("Time steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)

def plot_all(target='EC'):
    filenames = os.listdir(SERIES_DIR/'USGs')
    os.makedirs(ROOT_DIR/'result/USGs',exist_ok=True)
    for file in filenames:
        if target in file:
            name = file.replace('.csv','.png')
            print(name)
            plot_prediction(os.path.join(SERIES_DIR/'USGs',file),os.path.join(ROOT_DIR/'result/USGs',name))

if __name__ == "__main__":
    plot_all()

