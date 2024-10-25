from argparse import ArgumentParser
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap

plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.figsize'] = (3.5, 3.)


def plot(labels: list, dataframes: List[pd.DataFrame], reference_data: Optional[pd.DataFrame], title: str, scatter: bool):

    # Set color map
    colors = get_cmap(len(labels))

    # Plot the mcs data
    for label, df, color in zip(labels, dataframes, colors):
        xs = df["NumAPs"]
        if scatter:
            plt.scatter(xs, df['CollisionRateMean'], marker='.', label=label, color=color)
            plt.errorbar(
                xs, df['CollisionRateMean'], yerr=[
                    df['CollisionRateMean'] - df['CollisionRateLow'], 
                    df['CollisionRateHigh'] - df['CollisionRateMean']
                ], fmt='none', color=color)
        else:
            plt.plot(xs, df['CollisionRateMean'], marker='.', label=label, color=color)
            plt.fill_between(xs,  df['CollisionRateLow'], df['CollisionRateHigh'], alpha=0.5, color=color, linewidth=0)

    # Plot the reference data
    if reference_data is not None:
        reference_data = reference_data.sort_values(by='NumAPs')
        reference_data = reference_data[reference_data['NumAPs'] <= max([df['NumAPs'].max() for df in dataframes])]
        plt.plot(reference_data['NumAPs'], reference_data['CollisionRateMean'], label='Reference', linestyle='--', color='red')
    
    # Setup the plot
    plt.xlabel('Number of APs')
    plt.ylabel('Collision probability')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'collision-comparison.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-l', '--labels',         type=str, nargs='+', required=True)
    args.add_argument('-d', '--data',           type=str, nargs='+', required=True)
    args.add_argument('-r', '--reference_data', type=str, required=False)
    args.add_argument('-t', '--title',          type=str, required=False)
    args.add_argument('-s', '--scatter',        action='store_true')
    args = args.parse_args()

    # Get labels
    labels = args.labels

    # Load the MAPC data
    dataframes = []
    for data in args.data:
        dataframes.append(pd.read_csv(data).sort_values(by='NumAPs'))

    # Load the reference data
    reference_data = args.reference_data
    if reference_data is not None:
        reference_df = pd.read_csv(reference_data)
    else:
        reference_df = None
    
    # Get the title
    title =  args.title if args.title is not None else 'Collision Probability in Dense Networks'

    # Plot the data
    plot(labels, dataframes, reference_df, title, args.scatter)