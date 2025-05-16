from config import GRID_SIZE, POP_SIZE, GENERATIONS
from models.cnn import AntennaCNN
from ga.utils import initialize_population
from ga.evolution import evolve
from ga.fitness import dummy_cst_fitness
import matplotlib.pyplot as plt

if __name__ == "__main__":
    population = initialize_population(POP_SIZE, GRID_SIZE)
    model = AntennaCNN(grid_size=GRID_SIZE)
    final_pop = evolve(population, model, GENERATIONS, real_eval_fn=dummy_cst_fitness)

    best = final_pop[0]
    print("Best antenna design (binary matrix):")
    print(best)
    plt.imshow(best, cmap='gray')
    plt.title("Best Evolved Antenna")
    plt.colorbar()
    plt.show()
