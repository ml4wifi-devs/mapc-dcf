from argparse import ArgumentParser

import json
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style

def plot(data: pd.DataFrame) -> None:

    # Plot the data
    plt.plot(data['NumAPs'], 1. - data['SuccessRateMean'], label='mapc-dcf')
    plt.fill_between(data['NumAPs'],  1. - data['SuccessRateLow'], 1. - data['SuccessRateHigh'], alpha=0.5)
    plt.xlabel('Number of APs')
    plt.ylabel('Collision probability')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('collisions-vs-density.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--csv_data', type=str, required=True)
    args = args.parse_args()

    data_df = pd.read_csv(args.csv_data).sort_values(by='NumAPs')
    plot(data_df)