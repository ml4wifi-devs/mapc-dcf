from argparse import ArgumentParser
from typing import Optional, List

import os
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap

plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.figsize'] = (3.5, 3.)


def plot(labels: list, dataframes: List[pd.DataFrame], title: str):

    # Set color map
    colors = get_cmap(len(labels))

    # Plot the data
    for label, df, color in zip(labels, dataframes, colors):
        xs = df["Distance"]
        plt.plot(xs, df['CollisionRateMean'], marker='.', label=label, color=color)
        plt.fill_between(xs,  df['CollisionRateLow'], df['CollisionRateHigh'], alpha=0.5, color=color, linewidth=0)

    
    # Setup the plot
    plt.xlabel('Distance [m]')
    plt.ylabel('PER')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'per-distance.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-l', '--labels',         type=str, nargs='+', required=True)
    args.add_argument('-d', '--data',           type=str, nargs='+', required=True)
    args.add_argument('-t', '--title',          type=str, required=False)
    args = args.parse_args()

    # Get labels
    labels = args.labels

    # Load the MAPC data
    dataframes = []
    for data in args.data:
        dataframes.append(pd.read_csv(data).sort_values(by='Distance'))
    
    # Get the title
    title =  args.title if args.title is not None else 'PER vs Distance'

    # Plot the data
    plot(labels, dataframes, title)