import GA
import MLAGA

for i in range(50):
    GA.run(seed = i, store = True, show = False)
    MLAGA.run(seed = i, store = True, show = False)
