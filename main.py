import GA
import MLAGA

for seed in range(54, 200):
    GA.run(seed)
    MLAGA.run(seed)
