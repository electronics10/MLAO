import GA
import MLAGA

for i in range(50):
    GA.run(seed = i, store = True)
    MLAGA.run(seed = i, store = True)
