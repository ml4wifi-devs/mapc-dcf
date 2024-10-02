from typing import Optional

import os
import json
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

from mapc_mab.plots.utils import confidence_interval
from mapc_dcf.plots import plot_backoff_hist
from mapc_dcf.channel import WiFiFrame

class Logger:

    def __init__(self, results_path: str, dump_size: int = 1000) -> None:
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.header = ['SimTime', 'RunNumber', 'FrameID', 'Retransmission', 'Src', 'Dst', 'PayloadSize', 'MCS', 'Backoff', 'Collision']
        self.accumulator = []
        self.dump_size = dump_size
        self.dumped = False

        self.results_dir = os.path.dirname(results_path)
        self.results_path_csv = results_path.split('.')[0] + '.csv'
        self.results_path_json = results_path.split('.')[0] + '.json'

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


    def log(self, sim_time: float, run_number: int, frame: WiFiFrame, backoff: int, collision: bool):
        
        self.accumulator.append([
            sim_time,
            run_number,
            frame.id,
            frame.retransmission,
            frame.src,
            frame.dst,
            frame.size,
            frame.mcs,
            backoff,
            collision
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
    
    
    def shutdown(self):
        self._combine_dumps()
            
