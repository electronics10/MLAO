import numpy as np
import csv

def dummy_cst_fitness(individual):
    ideal = np.zeros_like(individual)
    ideal[:, individual.shape[1]//2] = 1
    return -np.sum((individual - ideal) ** 2)

def calculate_fitness_from_csv(csv_file, target_curve, freq_range=(2.3, 2.5)):
    freqs, s11s = [], []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            freq, s11 = float(row[0]), float(row[1])
            if freq_range[0] <= freq <= freq_range[1]:
                freqs.append(freq)
                s11s.append(float(s11))
    s11s = np.array(s11s)
    target = np.interp(freqs, target_curve[:, 0], target_curve[:, 1])
    return -np.mean((s11s - target)**2)
