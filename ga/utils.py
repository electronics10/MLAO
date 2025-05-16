import numpy as np

def initialize_population(pop_size, grid_size):
    return [np.random.randint(0, 2, size=(grid_size, grid_size)) for _ in range(pop_size)]

def mutate(individual, rate=0.01):
    mask = np.random.rand(*individual.shape) < rate
    return np.logical_xor(individual, mask).astype(int)

def crossover(p1, p2):
    mask = np.random.randint(0, 2, size=p1.shape)
    return np.where(mask, p1, p2)
