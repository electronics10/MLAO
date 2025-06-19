import numpy as np
import csv
import pandas as pd

def calculate_fitness(binary_array, target=(1.425,1.575), goal=-10):
    pop_index = int(''.join(map(str, binary_array)), 2) # binary back to decimal
    lfreq = target[0]
    hfreq = target[1]
    df = pd.read_csv(f"./s11/s11_{pop_index}.csv")
    j = 0
    n = 0
    freq = 0
    fitness = 0
    while freq < hfreq:
        freq = df.iloc[j, 0] # Read fequency
        if freq >= lfreq:
            s11 = df.iloc[j, 1] # Read s11
            # fitness += (goal - s11) # Record fitness # the larger the merrier
            fitness += max(goal, s11) 
            n += 1
        j += 1
    fitness = fitness/n/goal
    return fitness

pop_indices = np.arange(4096)

with open("fitness_table.csv", 'a', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for index in pop_indices:
        individual = [int(bit) for bit in format(index, '012b')]
        fitness = calculate_fitness(individual)
        spamwriter.writerow([index, fitness])
