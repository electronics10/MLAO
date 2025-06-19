# Run
1. Use `./preprocess/pre.py` to obtain s11 of all antenna topologies in the whole 4096 design space and saved in folder `./s11`.
2. Run `fitness_table_generator.py` to obtain table of fitnesses for different topologies.
3. Run `main.py` to obtain 50 samples of GA and MLAO-GA optimization results of iteration-fitness files and saved as `./data/GA_i` or `./data/MLAGA_i`. (Running the file GA or MLAO alone will show one sample of fitness plot with random seed=2).

# Show Data
1. To show the result plots (fitness vs. iteration) in `data`, run function `one_by_one()` in `results.py` to specify which sample to be plotted and run function `average()` to plot the average fitness vs. iteration plot.
2. To show the topology and it's corresponding s11 plot in `./s11`, one can run `plot.py` and specify topology's decimal coding.
