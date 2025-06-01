from Antenna_Design import Controller
import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
    os.makedirs("./s11", exist_ok=True)
    os.makedirs("./CST_Antennas", exist_ok=True)
    antenna = Controller("CST_Antennas/MLAO.cst")
    antenna.set_frequency_solver()
    # antenna.set_base()
    # antenna.set_domain()
    # antenna.set_port(antenna.port[0], antenna.port[1])

    for i in range(4096):  # 0 to 4095
        # Clear legacy
        antenna.delete_results()

        # Generate 9 digit binary sequence
        binary_str = format(i, '012b')  # Convert to 12-digit binary string
        print(i, binary_str)
        binary_list = [int(bit) for bit in binary_str]
        # Make middle four blocks PEC to reduce design space from 2^16 to 2^12
        binary_list.insert(5,1)
        binary_list.insert(6,1)
        binary_list.insert(9,1)
        binary_list.insert(10,1)

        # Update antenna topology
        antenna.update_distribution(binary_list)
        antenna.start_simulate()

        # Obtain S11 for further inspection
        s11 = antenna.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        data.to_csv(f's11/s11_{i}.csv', index=False) # save to CSV
        print(f"S11 saved to 's11/s11_{i}.csv'")
