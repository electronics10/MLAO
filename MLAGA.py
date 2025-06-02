import numpy as np
import pandas as pd
import os
import csv
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from settings import CONVERGENCE, POPULATION_SIZE, SELECT_RATE, MUTATE_RATE

def crossover(p1, p2):
    mask = np.random.randint(0, 2, size=p1.shape)
    return np.where(mask, p1, p2)

def mutate(individual, rate=0.1):
    mask = np.random.rand(*individual.shape) < rate
    return np.logical_xor(individual, mask).astype(int)

class AntennaCNN(nn.Module):
    def __init__(self, grid_size=4):
        super(AntennaCNN, self).__init__()
        self.grid_size = grid_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        flat_size = 16 * grid_size * grid_size

        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.view(-1, 1, self.grid_size, self.grid_size).float()
        return self.fc_layers(self.conv_layers(x))

class AntennaMLP(nn.Module):
    def __init__(self):
        super(AntennaMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class Fitness_Evaluator():
    def __init__(self):
        self.fitness_record = []
        self.model_mse = 100
        self.ml_folder = "artifacts_CNN"
        self.data_path = "artifacts_CNN/training_data.csv"
        if os.path.exists(self.data_path): os.remove(self.data_path)
        print("legacy cleaned")
        self.model = AntennaCNN()
        self.threshold = 0.2
        self.switch = 0
        self.population_cache = None
        self.avg_fitness = 0

    def calculate_fitness(self, binary_array):
        pop_index = int(''.join(map(str, binary_array)), 2) # binary back to decimal
        df = pd.read_csv("fitness_table.csv", header=None)
        fitness = float(df.iloc[pop_index, 0].split()[1])
        return fitness

    def fitness_assign_and_sort(self, population):
        scored_pop = []
        pop_avg_fitness = 0
        # print("Evaluate by CST.")
        for i in range(POPULATION_SIZE):
            # Use CST to calculate fitness
            individual = population[i]
            fitness = self.calculate_fitness(individual)
            scored_pop.append((individual, fitness))
            pop_avg_fitness += fitness
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            # Update training data
            with open(self.data_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list(np.ravel(self.graph(individual))) + [fitness])
                writer.writerow(list(np.ravel(self.graph(individual)[::-1])) + [fitness]) # symmetry, double data size

        pop_avg_fitness = pop_avg_fitness/POPULATION_SIZE
        best_fitness = scored_pop[0][1]
        best_index = int(''.join(map(str, scored_pop[0][0])), 2)
        # print(f"Pop avg Fitness: {pop_avg_fitness}")
        # print(f"Best Fitness: {best_fitness} ({best_index})")
        self.train() # train model
        self.switch = 0
        return scored_pop, best_fitness

    def evaluate_fitness(self, population):
        best_fitness = -1000
        best_index = 0
        pop_avg_fitness = 0
        scored_pop = []
        # Logic for using model or CST solver
        max_MLE = 20
        if self.switch == 0: self.population_cache = population
        if self.model_mse < self.threshold and self.switch <= max_MLE: # Use CNN model
            if self.switch == max_MLE:
                scored_pop, best_fitness = self.fitness_assign_and_sort(population)
                for i in range(POPULATION_SIZE): pop_avg_fitness += scored_pop[i][1]
                pop_avg_fitness = pop_avg_fitness/POPULATION_SIZE
                if pop_avg_fitness < self.avg_fitness or best_fitness < self.fitness_record[-1] - 0.05: # very rough criterion:
                    print("model failed")
                    best_fitness = self.fitness_record[-1]
                    for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record previous fitness
                    scored_pop, best_fitness = self.fitness_assign_and_sort(self.population_cache)
                    for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
                    best_index = int(''.join(map(str, scored_pop[0][0])), 2)
                    print(f"Iterations: {len(self.fitness_record)}")
                    print(f"Best Fitness: {best_fitness} ({best_index})\n")
                else:
                    for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
                    best_index = int(''.join(map(str, scored_pop[0][0])), 2)
                    print("Model workeddddddddddddddddddddddddddd")
                    print(f"Iterations: {len(self.fitness_record)}")
                    print(f"Best Fitness: {best_fitness} ({best_index})\n")
            else:
                scored_pop = self.predict(population)
                scored_pop.sort(key=lambda x: x[1], reverse=True)
                best_fitness = scored_pop[0][1]
                # print("Evaluate by ML model.")
                best_index = int(''.join(map(str, scored_pop[0][0])), 2)
                for i in range(POPULATION_SIZE): pop_avg_fitness += scored_pop[i][1]
                pop_avg_fitness = pop_avg_fitness/POPULATION_SIZE
                # print(f"Predicted pop avg Fitness: {pop_avg_fitness}")
                # print(f"Predicted Best Fitness: {best_fitness} ({best_index})\n")
                self.switch += 1
        else:
            # print("Evaluate by ML model.")
            scored_pop, best_fitness = self.fitness_assign_and_sort(population)# Use CST
            for i in range(POPULATION_SIZE): self.fitness_record.append(best_fitness) # record fitness
            for i in range(POPULATION_SIZE): pop_avg_fitness += scored_pop[i][1]
            pop_avg_fitness = pop_avg_fitness/POPULATION_SIZE
            self.avg_fitness = pop_avg_fitness
            print(f"Iterations: {len(self.fitness_record)}")
            print(f"Best Fitness: {best_fitness} ({best_index})\n")
        return scored_pop, best_fitness

    def graph(self, binary_array): # Change 12 digits binary list into actual antenna topology
        binary_list = list(binary_array)
        binary_list.insert(5,1)
        binary_list.insert(6,1)
        binary_list.insert(9,1)
        binary_list.insert(10,1)
        binary_array = np.array(binary_list)
        return np.reshape(binary_array, (4,4))
    
    def predict(self, population):
        # Preprocess
        population16 = []
        for individual in population: population16.append(np.ravel(self.graph(individual))) # 16 digits
        population16 = np.array(population16)

        folder = self.ml_folder
        device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
        # Evaluation and plotting
        x_scaler = pickle.load(open(f"{folder}/x_scaler.pkl", "rb"))
        y_scaler = pickle.load(open(f"{folder}/y_scaler.pkl", "rb"))
        X_scaled = x_scaler.transform(population16)
        X_test = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        model = self.model.to(device)
        model.load_state_dict(torch.load(f"{folder}/model.pth")) # Load best model saved
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

        scored_pop = []
        for i in range(len(population)):
            scored_pop.append((population[i], y_pred[i][0]))
        
        return scored_pop

    def train(self): 
        folder = self.ml_folder
        os.makedirs(folder, exist_ok=True)
        # Device configuration
        device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
        # Load dataset
        data = pd.read_csv(f'{self.data_path}').values
        x_data = data[:, :16]
        y_data = data[:, 16:]
        # Normalize inputs and outputs separately
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_data = x_scaler.fit_transform(x_data)
        y_data = y_scaler.fit_transform(y_data)
        # Split training set and validation set
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        # Save scalers
        pickle.dump(x_scaler, open(f"{folder}/x_scaler.pkl", "wb"))
        pickle.dump(y_scaler, open(f"{folder}/y_scaler.pkl", "wb"))
        
        # Model, Loss, Optimizer
        model = self.model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # Early stopping settings
        early_stopping_patience = 70
        best_loss = float('inf')
        patience_counter = 0
        # Training loop
        num_epochs = 5000
        loss_list = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                loss_list.append([epoch, loss.item(), test_loss.item()])
            # if epoch % 50 == 0: print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
            # Early stopping
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                patience_counter = 0
            else: patience_counter += 1
            if patience_counter >= early_stopping_patience: break

        # Save the model
        torch.save(model.state_dict(), f'{folder}/model.pth')
        mse = test_loss.item()
        # print("mse: ", mse)
        self.model_mse = mse
    
    def store_and_show_fitness(self, seed): # Store and Show data
        os.makedirs("./data", exist_ok=True)
        df = pd.DataFrame(self.fitness_record, columns = ['best_fitness'])
        df.to_csv(f'data/MLAGA_{seed}.csv', index=False) 
        # plt.plot(self.fitness_record, label='MLAO-GA', linestyle='-.', color='red')
        # plt.xlabel("Number of Simulations")
        # plt.ylabel("Fitness")
        # plt.title("Convergence Rate")
        # plt.legend()
        # plt.axhline(y = 0.4, linestyle=':', color = '#000')
        # plt.show()


# if __name__ == "__main__":
def run(seed = 2):
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
    fitness_evaluator = Fitness_Evaluator()
    
    # while best_fitness < CONVERGENCE or fitness_evaluator.switch != 0:
    while len(fitness_evaluator.fitness_record) < 2000:
        # print(f"\nGen{generation}")

        # Fitness Assignment
        scored_pop, best_fitness = fitness_evaluator.evaluate_fitness(population)
        
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
        
        generation += 1

    fitness_evaluator.store_and_show_fitness(seed)
