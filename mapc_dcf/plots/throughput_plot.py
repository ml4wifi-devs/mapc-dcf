import json
from argparse import ArgumentParser
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap, calculate_ema
from mapc_dcf.constants import DATA_RATES

# plt.rcParams['text.usetex'] = False

def plot(json_data: Dict, df: Optional[pd.DataFrame], run_number: Optional[int], mcs: int, ema_alpha: float = 0.02):

    color = get_cmap(1)[0]

    # Parse the data from the json file
    sim_time = json_data['Config']['simulation_length']
    warmup_time = json_data['Config']['warmup_length']
    data_rate_mean = json_data['DataRate']['Mean']
    data_rate_low = json_data['DataRate']['Low']
    data_rate_high = json_data['DataRate']['High']

    # Plot the throughput for a specific run
    if df is not None:
        df = df[(df["Collision"] == False) &  (df["RunNumber"] == run_number)].sort_values("SimTime")
        df["dThr"] = np.concatenate(([0], df["AMPDUSize"][1:].values * 1e-6 / (df["SimTime"][1:].values - df["SimTime"][:-1].values)))
        xs = df["SimTime"].values
        ys = df["dThr"].values
        ys_smooth = calculate_ema(ys, alpha=ema_alpha)
        plt.plot(xs, ys, alpha=0.6, color=color, label="Instantaneous Throughput")
        plt.plot(xs, ys_smooth, color=color, linewidth=0.5)
        plt.axvline(warmup_time, color="red", linestyle="--")

    # Plot the average throughput
    # - Define the x and y values
    res = 300
    xs = np.linspace(0, sim_time + warmup_time, res)
    ys = np.array([data_rate_mean] * res)
    ys_low = np.array([data_rate_low] * res)
    ys_high = np.array([data_rate_high] * res)

    # - Plot the data
    plt.plot(xs, ys, color=color, label=f"Average Throughput ({data_rate_mean:.3f} Mb/s)", linestyle="--")
    plt.fill_between(xs, ys_low, ys_high, alpha=0.5, color="black", linewidth=0)

    # - Plot the MCS data rate
    plt.axhline(DATA_RATES[mcs], color="gray", label=f"MCS {mcs} Data Rate ({DATA_RATES[mcs]:.3f})", linestyle="--")

    # Setup the plot
    plt.xlabel('Time [s]')
    plt.yticks(np.arange(0, 155 + 1, 25))
    plt.ylabel('Throughput [Mb/s]')
    plt.ylim(0, 155)
    plt.xlim(0, sim_time + warmup_time)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'throughput.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-j', '--json_data',  type=str, required=True)
    args.add_argument('-c', '--csv_data',   type=str)
    args.add_argument('-r', '--run_number', type=int)
    args.add_argument('-m', '--mcs',        type=int, default=11)
    args = args.parse_args()

    # Check the arguments
    if args.csv_data is not None:
        assert args.run_number is not None, 'When the "--csv_data" argument is specified, a "--run_number" must aslo be specified.'
    
    # Load the json data
    json_data = args.json_data
    with open(json_data, 'r') as f:
        json_data = json.load(f)

    # Load the csv data
    csv_data = args.csv_data
    if csv_data is not None:
        csv_data = pd.read_csv(csv_data)

    # Plot the data
    plot(json_data, csv_data, args.run_number, args.mcs)