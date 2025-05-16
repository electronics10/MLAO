import torch
import random
from ga.utils import crossover, mutate
from train.trainer import train_model

def evaluate(individual, model, threshold=0.1, use_surrogate=True, real_eval_fn=None):
    tensor = torch.tensor(individual.reshape(1, 1, *individual.shape)).float()
    if use_surrogate:
        pred, std = model.predict_with_uncertainty(tensor, n_samples=10)
        if std.item() < threshold:
            return pred.item()
    if real_eval_fn:
        return real_eval_fn(individual)
    return None

def evolve(population, model, generations, elite_fraction=0.2, mutation_rate=0.01, real_eval_fn=None):
    pop_size = len(population)
    elite_count = int(pop_size * elite_fraction)
    training_data = []

    for gen in range(generations):
        scored = [(ind, evaluate(ind, model, real_eval_fn=real_eval_fn)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        elites = [x[0] for x in scored[:elite_count]]

        for ind, fit in scored:
            if fit is not None:
                training_data.append((ind, fit))

        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = mutate(crossover(p1, p2), rate=mutation_rate)
            new_pop.append(child)

        population = new_pop
        print(f"Generation {gen+1}: Best fitness = {scored[0][1]:.4f}")

        if training_data:
            train_X = torch.tensor([x[0] for x in training_data]).float()
            train_y = torch.tensor([x[1] for x in training_data]).float()
            train_model(model, train_X, train_y)

    return population
