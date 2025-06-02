import GA
import MLAGA

for seed in range(100):
    GA.run(seed)
    MLAGA.run(seed)
