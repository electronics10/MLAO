import numpy as np
import pandas as pd
import os
from settings import CONVERGENCE, POPULATION_SIZE, SELECT_RATE, MUTATE_RATE

def calculate_fitness(binary_array, target=(2.4,2.5), goal=-6):
    pop_index = int(''.join(map(str, binary_array)), 2) # binary back to decimal
    # lfreq = target[0]
    # hfreq = target[1]
    # df = pd.read_csv(f"./s11/s11_{pop_index}.csv")
    # j = 0
    # n = 0
    # freq = 0
    # fitness = 0
    # while freq < hfreq:
    #     freq = df.iloc[j, 0] # Read fequency
    #     if freq >= lfreq:
    #         s11 = df.iloc[j, 1] # Read s11
    #         fitness += (goal - s11) # Record fitness # the larger the merrier
    #         n += 1
    #     j += 1
    # fitness = fitness/n*(target[1]-target[0])*500

    df = pd.read_csv("fitness_table.csv", header=None)
    fitness = float(df.iloc[pop_index, 0].split()[1])
    return fitness

def fitness_assign_and_sort(population):
    scored_pop = []
    pop_avg_fitness = 0
    for i in range(len(population)):
        individual = population[i]
        fitness = calculate_fitness(individual)
        scored_pop.append((individual, fitness))
        pop_avg_fitness += fitness
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    pop_avg_fitness = pop_avg_fitness/len(population)
    return scored_pop, pop_avg_fitness

def crossover(p1, p2):
    mask = np.random.randint(0, 2, size=p1.shape)
    return np.where(mask, p1, p2)

def mutate(individual, rate=0.1):
    mask = np.random.rand(*individual.shape) < rate
    return np.logical_xor(individual, mask).astype(int)

def run(seed = 2, store = False, show = False):
    # Create population
    np.random.seed(seed)
    pop_indices = np.random.randint(4096, size = POPULATION_SIZE)
    population = []
    for i in range(POPULATION_SIZE):
        individual = np.array([int(bit) for bit in format(pop_indices[i], '012b')])
        population.append(individual)
    population = np.array(population)

    # Evolve till spec satisfied
    generation = 0
    best_fitness = -1000
    fitness_record = []
    while len(fitness_record) < 2000:
        # Fitness Assignment
        scored_pop, pop_avg_fitness = fitness_assign_and_sort(population) # [(individual, fitness)]
        best_fitness = scored_pop[0][1]
        best_index = int(''.join(map(str, scored_pop[0][0])), 2)
        for i in range(POPULATION_SIZE): fitness_record.append(best_fitness)

        # Selection
        elites_size = int(POPULATION_SIZE*SELECT_RATE)
        elites = scored_pop[:elites_size] # [(individual, fitness)]

        new_pop = [] # new population
        for item in elites: new_pop.append(item[0]) # append elite individuals

        # Crossover and Mutation
        while len(new_pop) < POPULATION_SIZE:
            # Randomly select two parents from elites
            p_indices = np.random.randint(elites_size, size=2)
            p1 = np.array(elites[p_indices[0]][0])
            p2 = np.array(elites[p_indices[1]][0])
            child = mutate(crossover(p1, p2), rate = MUTATE_RATE)
            new_pop.append(child)
        population = np.array(new_pop)

        print(f"\nGen{generation}")
        print(f"Iterations: {(generation)*POPULATION_SIZE}")
        print(f"Pop avg Fitness: {pop_avg_fitness}")
        print(f"Best Fitness: {best_fitness} ({best_index})")

        generation += 1

    if store:
        # Store and Show data
        os.makedirs("./data", exist_ok=True)
        df = pd.DataFrame(fitness_record, columns = ['best_fitness'])
        df.to_csv(f'data/GA_{seed}.csv', index=False) 
    if show:
        import matplotlib.pyplot as plt
        plt.plot(fitness_record, label='GA', linestyle='-.', color='blue')
        plt.xlabel("Number of Simulations")
        plt.ylabel("Fitness")
        plt.title("Antenna Fitness")
        plt.legend()
        # plt.axhline(y = 0.5, linestyle=':', color = '#000')
        plt.show()

if __name__ == "__main__":
    run(seed = 2, store = False, show = True)
