import os
import json
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from mapc_research.plots.utils import confidence_interval


def combine_data(results_dir, save_path, ci, clear_dir):
    """
    Combines the json results files from the results_dir into a single csv file and saves it to save_path.
    If clear_dir is True, the json files are deleted after combining.
    The csv file has the following columns:
    - NumAPs
    - CollisionRateMean
    - CollisionRateLow
    - CollisionRateHigh

    The example json file is:
    {
        "DataRate": {
            "Mean": 44.40000000000002,
            "Low": 44.40000000000002,
            "High": 44.40000000000002
        },
        "CollisionRate": {
            "Mean": 1.0,
            "Low": 1.0,
            "High": 1.0
        }
    }
    """

    results = []
    for file in os.listdir(results_dir):
        
        if not file.endswith('.json'):
            continue

        with open(os.path.join(results_dir, file), 'r') as f:
            distance = int(file.split('.')[0][1:])
            data = json.load(f)
            data = data['CollisionRate']['Data']
            mean, low, high = confidence_interval(np.array(data), ci)
            results.append({
                'Distance': distance,
                'CollisionRateMean': mean,
                'CollisionRateLow': low,
                'CollisionRateHigh': high
            })

    if clear_dir:
        for file in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, file))
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--out_dir',    type=str, default='out')
    args.add_argument('-s', '--save_path',  type=str, default='out/combined.csv')
    args.add_argument('-i', '--ci',         type=float, default=0.99)
    args.add_argument('-c', '--clear_dir',  action='store_true')
    args = args.parse_args()

    combine_data(args.out_dir, args.save_path, args.ci, args.clear_dir)

