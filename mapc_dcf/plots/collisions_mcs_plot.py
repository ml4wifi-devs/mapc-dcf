from argparse import ArgumentParser
from typing import Dict

import os
import pandas as pd
import matplotlib.pyplot as plt

from mapc_dcf.plots import set_style, get_cmap

plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.figsize'] = (3.5, 3.)


def plot(mcs_data: Dict[int, pd.DataFrame], reference_data: pd.DataFrame):

    # Set color map
    colors = get_cmap(12)

    # Plot the mcs data
    xs = list(mcs_data.values())[0]["NumAPs"]
    for mcs, df in mcs_data.items():
        plt.plot(xs, df['CollisionRateMean'], marker='.', label=f'MCS {mcs}', color=colors[mcs])
        plt.fill_between(xs,  df['CollisionRateLow'], df['CollisionRateHigh'], alpha=0.5, color=colors[mcs], linewidth=0)

    # Plot the reference data
    for i, row in reference_data.iterrows():
        if row[-1] == 'Analytical model':
            plt.plot(xs, row[:-1], marker='.', label=row[-1], linestyle='--', color='gray')
        else:
            continue
    
    # Setup the plot
    plt.xlabel('Number of APs')
    plt.ylabel('Collision probability')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()
    plt.savefig(f'collision-mcs.pdf', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('-r', '--reference_data', type=str, required=True)
    args.add_argument('-d', '--results_dir', type=str, required=True)
    args = args.parse_args()
    results_dir = args.results_dir
    reference_data = args.reference_data

    # Load the MAPC data
    csv_data = os.listdir(results_dir)
    csv_data = sorted(csv_data, key=lambda x: int(x.split('.')[0]))
    data_dict = {int(file.split('.')[0]): pd.read_csv(os.path.join(results_dir, file)).sort_values(by='NumAPs') for file in csv_data}

    # Load the reference data
    reference_df = pd.read_csv(reference_data)
    reference_df = reference_df[reference_df["Name"] != "DCF-SimPy"]
    reference_df = reference_df.iloc[:, :11]

    # Plot the data
    plot(data_dict, reference_df)