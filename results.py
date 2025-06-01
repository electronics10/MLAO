import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def pad_and_average(arrays):
    max_len = max(len(arr) for arr in arrays)
    padded_arrays = []

    for arr in arrays:
        pad_width = max_len - len(arr)
        padded = np.pad(arr, (0, pad_width), constant_values=np.nan)
        padded_arrays.append(padded)

    stacked = np.vstack(padded_arrays)
    return np.nanmean(stacked, axis=0)

data_dir = "./data"
files = [f for f in os.listdir(data_dir) if f.startswith("GA")]
files_ml = [f for f in os.listdir(data_dir) if f.startswith("ML")]
            
plt.figure(figsize=(10, 5))
ga = []
mlaga = []
for file in files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    ga.append(df.iloc[:, 0].values)
for file in files_ml:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    mlaga.append(df.iloc[:, 0].values)

ga = pad_and_average(ga)
mlaga = pad_and_average(mlaga)
plt.plot(ga, label='GA', linestyle='-.', color='blue')
plt.plot(mlaga, label='MLAO-GA', linestyle='-.', color='red')
plt.xlabel("Number of Simulations")
plt.ylabel("Fitness")
plt.title("Convergence Rate")
plt.legend()
plt.axhline(y = 170, linestyle=':', color = '#000')
plt.show()