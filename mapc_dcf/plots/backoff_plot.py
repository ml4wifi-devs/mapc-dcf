import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from mapc_mab.plots.config import *

set_style()

INTERVALS_MAP = {
    "4": "[0, 15)",
    "5": "[15, 31)",
    "6": "[31, 63)",
    "7": "[63, 127)",
    "8": "[127, 255)",
    "9": "[255, 511)",
    "10": "[511, 1023)"
}


def plot(data: dict, save_path: str) -> None:

    plt.figure(figsize=(5, 3))

    aps = data.keys()
    for ap in aps:
        backoffs = data[ap]
        plt.bar(backoffs.keys(), list(map(len, backoffs.values())), label=f'AP {ap}', alpha=0.5)

    plt.xlabel('Backoff Interval')
    plt.ylabel('Number of Occurrences')
    plt.yscale('log')
    plt.ylim(bottom=0.5)
    plt.xticks(range(0, 7), list(map(lambda x: INTERVALS_MAP[x], backoffs.keys())), rotation=45)
    plt.title('Distribution of Backoff Values')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--data', type=str, required=True, help='Path to the json results file')
    args = args.parse_args()

    save_path = args.data.split('.')[0] + '.pdf'

    # Load data from JSON file
    with open(args.data, 'r') as f:
        data = json.load(f)["Backoff"]

    # Plot the backoff distribution
    plot(data, save_path)
    