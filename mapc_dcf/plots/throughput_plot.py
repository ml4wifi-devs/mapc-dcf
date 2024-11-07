import json
from argparse import ArgumentParser
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap, calculate_ema
from mapc_dcf.constants import DATA_RATES

# plt.rcParams['text.usetex'] = False

def plot(json_data: Dict, csv_data: Optional[pd.DataFrame], mcs: Optional[int], save_path: str) -> None:

    color = get_cmap(1)[0]

    # Parse the data from the json file
    sim_time = json_data['Config']['simulation_length']
    warmup_time = json_data['Config']['warmup_length']
    data_rate_mean = json_data['DataRate']['Mean']
    data_rate_low = json_data['DataRate']['Low']
    data_rate_high = json_data['DataRate']['High']

    # Plot the warmup
    plt.axvline(warmup_time, color="black", linestyle="-", linewidth=0.5)
    if warmup_time > 0:
        plt.text(warmup_time * 0.5, 50, "Warmup", rotation=90, verticalalignment='center')
        plt.text(warmup_time * 1.1, 50, "Simtime", rotation=90, verticalalignment='center')

    # Plot the throughput for all runs
    if csv_data is not None:
        for run_number in range(1, 1 + json_data['Config']['n_runs']):
            df = csv_data
            df = df[(df["Collision"] == False) &  (df["RunNumber"] == run_number)].sort_values("SimTime")
            window_size = 5
            d_t = np.concatenate(([1.] * window_size, (df["SimTime"][5:].values - df["SimTime"][:-5].values)))
            d_data = df["AMPDUSize"].rolling(window_size).sum().fillna(0).values
            d_thr = d_data * 1e-6 / d_t
            df["dThr"] = d_thr
            xs = df["SimTime"].values
            ys = df["dThr"].values
            plt.plot(xs, ys, alpha=0.4, color=color, linewidth=0.5)

    # Plot the average throughput
    # - Define the x and y values
    res = 300
    xs = np.linspace(0, sim_time + warmup_time, res)
    ys = np.array([data_rate_mean] * res)
    ys_low = np.array([data_rate_low] * res)
    ys_high = np.array([data_rate_high] * res)

    # - Plot the data
    plt.plot(xs, ys, color="black", label=f"Average Throughput ({data_rate_mean:.3f} Mb/s)", linestyle="--", linewidth=0.5)
    plt.fill_between(xs, ys_low, ys_high, alpha=0.5, color="black", linewidth=0)

    # - Plot the MCS data rate
    if mcs is not None:
        plt.axhline(DATA_RATES[mcs], color="red", label=f"MCS {mcs} Data Rate ({DATA_RATES[mcs]:.3f})", linestyle="--", linewidth=0.5)

    # Setup the plot
    plt.xlabel('Time [s]')
    xticks = plt.xticks()[0]
    xticks = sorted(np.append(xticks, warmup_time))
    plt.xticks(xticks)
    plt.yticks(np.arange(0, 300 + 1, 50))
    plt.ylabel('Throughput [Mb/s]')
    plt.ylim(0, 300)
    plt.xlim(0, sim_time + warmup_time)
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-j', '--json_data',  type=str, required=True)
    args.add_argument('-c', '--csv_data',   type=str)
    args = args.parse_args()
    
    # Load the json data
    json_data = args.json_data
    with open(json_data, 'r') as f:
        json_data = json.load(f)
    
    # Get the mcs from config
    mcs = None
    try:
        mcs = json_data['Config']['scenario_params']['mcs']
    except KeyError:
        pass

    # Load the csv data
    csv_data = args.csv_data
    if csv_data is not None:
        csv_data = pd.read_csv(csv_data)
    
    # Create the save path
    save_path = args.json_data.replace('.json', '_thr.pdf')

    # Plot the data
    plot(json_data, csv_data, mcs, save_path)