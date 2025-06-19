import GA
import MLAGA
import os

os.makedirs(f"./data", exist_ok=True)

for i in range(50):
    GA.run(seed = i, store = True, show = False)
    MLAGA.run(seed = i, store = True, show = False)
