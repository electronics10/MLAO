import pandas as pd
import matplotlib.pyplot as plt

indices = map(int, input("index: ").split())

for i in indices:
    df = pd.read_csv(f"./s11/s11_{i}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label=f'{i}')

plt.xlabel("Frequency")
plt.ylabel("S11")
plt.title("test")
plt.legend()
plt.grid()
plt.show()
