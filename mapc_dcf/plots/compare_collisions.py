from argparse import ArgumentParser
from typing import Optional

import os
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap

plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.figsize'] = (3.5, 3.)


def plot(data_a: pd.DataFrame, data_b: pd.DataFrame, labels: list, reference_data: Optional[pd.DataFrame]):

    # Set color map
    colors = ['tab:blue', 'tab:orange']

    # Plot the mcs data
    for df, label, color in zip([data_a, data_b], labels, colors):
        xs = df["NumAPs"]
        plt.plot(xs, df['CollisionRateMean'], marker='.', label=label, color=color)
        plt.fill_between(xs,  df['CollisionRateLow'], df['CollisionRateHigh'], alpha=0.5, color=color, linewidth=0)

    # Plot the reference data
    if reference_data is not None:
        for i, row in reference_data.iterrows():
            if row[-1] == 'Analytical model':
                xs = list(range(1, 11))
                plt.plot(xs, row[:-1], marker='.', label=row[-1], linestyle='--', color='gray')
            else:
                continue
    
    # Setup the plot
    plt.xlabel('Number of APs')
    plt.ylabel('Collision probability')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'collision-comparison.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-a', '--data_a', type=str, required=True)
    args.add_argument('-b', '--data_b', type=str, required=True)
    args.add_argument('-l', '--labels', type=str, nargs='+', required=True)
    args.add_argument('-r', '--reference_data', type=str)
    args = args.parse_args()

    # Load the MAPC data
    data_a = pd.read_csv(args.data_a).sort_values(by='NumAPs')
    data_b = pd.read_csv(args.data_b).sort_values(by='NumAPs')

    # Load the reference data
    reference_data = args.reference_data
    if reference_data is not None:
        reference_df = pd.read_csv(reference_data)
        reference_df = reference_df[reference_df["Name"] != "DCF-SimPy"]
        reference_df = reference_df.iloc[:, :11]

    # Plot the data
    plot(data_a, data_b, args.labels, reference_df)