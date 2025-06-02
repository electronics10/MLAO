import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# def pad_and_average(arrays):
#     max_len = max(len(arr) for arr in arrays)
#     padded_arrays = []

#     for arr in arrays:
#         pad_width = max_len - len(arr)
#         padded = np.pad(arr, (0, pad_width), constant_values=np.nan)
#         padded_arrays.append(padded)

#     stacked = np.vstack(padded_arrays)
#     return np.nanmean(stacked, axis=0)

def average(truncate = 2010):
    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.startswith("GA")]
    files_ml = [f for f in os.listdir(data_dir) if f.startswith("ML")]
    # files = []
    # files_ml = []
    # ub = 50
    # lb = 0
    # for f in os.listdir(data_dir):
    #     if f.startswith("GA"):
    #         num = int(f.split("_")[1].split(".")[0])
    #         if lb < num < ub: files.append(f)
    #     if f.startswith("ML"):
    #         num = int(f.split("_")[1].split(".")[0])
    #         if lb < num < ub: files_ml.append(f)

    ga = []
    mlaga = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        ga.append(df.iloc[:truncate, 0].values)
    for file in files_ml:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        mlaga.append(df.iloc[:truncate, 0].values)

    # ga = pad_and_average(ga)
    # mlaga = pad_and_average(mlaga)
    ga = np.average(np.array(ga), axis=0)
    mlaga = np.average(np.array(mlaga), axis=0)

    plt.plot(ga, label='GA', linestyle='-.', color='blue')
    plt.plot(mlaga, label='MLAO-GA', linestyle='-.', color='red')
    plt.xlabel("Number of Simulations")
    plt.ylabel("Average Fitness")
    plt.title(f"Antenna Fitness ({len(files)-1} cases)")
    plt.legend()
    # plt.axhline(y = 0.5, linestyle=':', color = '#000')
    plt.grid()
    plt.show()

def one_by_one():
    indices = list(map(int, input("index: ").split()))
    for i in indices:
        plt.figure(i)
        ga = pd.read_csv(f"data/GA_{i}.csv").iloc[:,0].values
        mlaga = pd.read_csv(f"data/MLAGA_{i}.csv").iloc[:,0].values
        plt.plot(ga, label='GA', linestyle='-.', color='blue')
        plt.plot(mlaga, label='MLAO-GA', linestyle='-.', color='red')
        plt.xlabel("Number of Simulations")
        plt.ylabel("Fitness")
        plt.title(f"Antenna Fitness (case {i})")
        plt.legend()
        plt.axhline(y = 0.5, linestyle=':', color = '#000')
    plt.show()

if __name__ == "__main__":
    one_by_one()
    # average()
