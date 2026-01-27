from PyEMD import CEEMDAN
import numpy as np

def run_ceemdan(data, trials=50, max_imfs=12):
    print(f"--> Đang chạy CEEMDAN (Trials: {trials})... Vui lòng chờ.")
    ceemdan = CEEMDAN(trials=trials, epsilon=0.2)
    
    imfs = ceemdan(data.reshape(-1), max_imf=max_imfs)
    
    return imfs