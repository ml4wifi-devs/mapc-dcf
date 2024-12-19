import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from mapc_dcf.channel import AMPDU
from mapc_dcf.utils import confidence_interval


class Logger:

    def __init__(self, simulation_length: float, warmup_length: float, results_path: str, dump_size: int = 1000) -> None:
        
        self.simulation_length = simulation_length
        self.warmup_length = warmup_length
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.header = ['SimTime', 'RunNumber', 'FrameID', 'Retransmission', 'Src', 'Dst', 'AMPDUSize', 'MCS', 'TxPower', 'CW', 'Backoff', 'Collision', 'NSuccess', 'NCollision']
        self.accumulator = []
        self.dump_size = dump_size
        self.dumped = False

        self.get_cw_exp = lambda backoff: int(np.floor(np.log2(max(backoff, 15)))) + 1

        self.results_dir = os.path.dirname(results_path)
        self.results_path_csv = results_path + '.csv'
        self.results_path_json = results_path + '.json'

        # Create the results files
        if os.path.exists(self.results_path_csv):
            logging.warning(f"logger: Overwriting file {self.results_path_csv}!")
            os.remove(self.results_path_csv)
        if os.path.exists(self.results_path_json):
            logging.warning(f"logger: Overwriting file {self.results_path_json}!")
            os.remove(self.results_path_json)


    def dump_acumulators(self, run_number: int):
        
        if not self.dumped:
            self.dump_file_path = os.path.join(self.results_dir, f"dump_{self.timestamp}_{run_number}.csv")
            self.dumped = True

        logging.warning(f"Dumping {len(self.accumulator)} rows to {self.dump_file_path}")
        
        for row in self.accumulator:
            self.dump_file = open(self.dump_file_path, 'a')
            self.dump_file.write(','.join(map(str, row)) + '\n')
            self.dump_file.close()
        
        self.accumulator = []


    def log(self, sim_time: float, run_number: int, frame: AMPDU, cw: int, backoff: int, n_success: int, n_collision: int) -> None:
        
        collision = n_success == 0
        self.accumulator.append([
            sim_time,
            run_number,
            frame.id,
            frame.retransmission,
            frame.src,
            frame.dst,
            frame.pdu_size * n_success,
            frame.mcs,
            frame.tx_power,
            cw,
            backoff,
            collision,
            n_success,
            n_collision
        ])

        if len(self.accumulator) >= self.dump_size:
            self.dump_acumulators(run_number)
    

    def _combine_dumps(self):
        
        results_csv = open(self.results_path_csv, 'w')
        results_csv.write(','.join(self.header) + '\n')

        for dump_file in [f for f in os.listdir(self.results_dir) if f.startswith(f'dump_{self.timestamp}')]:
            with open(os.path.join(self.results_dir, dump_file), 'r') as dump:
                results_csv.write(dump.read())
            os.remove(os.path.join(self.results_dir, dump_file))
    
    
    def _parse_results(self, config: dict):
        
        # TODO: Implement a more efficient way to load the results
        # Load the results (May be too large to load at once)
        results_csv = pd.read_csv(self.results_path_csv)

        # Calculate the data rates and collision rates for each run
        data_rates = []
        collision_rates = []
        for run in results_csv['RunNumber'].unique():

            # Filter the results for the current run after the warmup period
            run_df = results_csv[(results_csv['RunNumber'] == run) & (results_csv['SimTime'] > self.warmup_length)]

            # Calculate the data rate
            total_payload = run_df[run_df['Collision'] == False]['AMPDUSize'].sum()
            data_rates.append(total_payload / self.simulation_length / 1e6)

            # Calculate the collision rate
            total_collisions = run_df['Collision'].sum()
            collision_rates.append(total_collisions / len(run_df))

        # Calculate the confidence intervals
            # For the data rate
        data_rate_mean, data_rate_low, data_rate_high = confidence_interval(np.array(data_rates))
        data_rate_std = np.std(data_rates)

            # For the collision rate
        collision_rate_mean, collision_rate_low, collision_rate_high = confidence_interval(np.array(collision_rates))
        collision_rate_std = np.std(collision_rates)

            # For the backoffs
        backoffs = np.array(results_csv[results_csv['SimTime'] > self.warmup_length]['Backoff'].values)
        backoff_mean, backoff_low, backoff_high = confidence_interval(backoffs)
        backoff_std = np.std(backoffs)

        # Fill the json with the results
        results_json = {}
        results_json['DataRate'] = {
            'Mean': data_rate_mean,
            'Std': data_rate_std,
            'Low': data_rate_low,
            'High': data_rate_high,
            'Data': data_rates
        }
        results_json['CollisionRate'] = {
            'Mean': collision_rate_mean,
            'Std': collision_rate_std,
            'Low': collision_rate_low,
            'High': collision_rate_high,
            'Data': collision_rates
        }
        results_json['Backoffs'] = {
            'Mean': backoff_mean,
            'Std': backoff_std,
            'Low': backoff_low,
            'High': backoff_high,
        }
        results_json["Config"] = config

        with open(self.results_path_json, 'w') as file:
            json.dump(results_json, file, indent=4)


    def shutdown(self, config: dict) -> None:
        self._combine_dumps()
        self._parse_results(config)
            
