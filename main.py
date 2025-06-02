import GA
import MLAGA

for seed in range(50):
    GA.run(seed)
    MLAGA.run(seed)
