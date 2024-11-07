import os

OUTPUT_DIR = 'out'
DIRS = ['random', 'residential', 'small_office']

if __name__ == '__main__':
    for d in DIRS:
        for f in os.listdir(f'{OUTPUT_DIR}/{d}'):
            if f.endswith('.json'):
                json_path = f'{OUTPUT_DIR}/{d}/{f}'
                csv_path = f'{OUTPUT_DIR}/{d}/{f.replace(".json", ".csv")}'
                os.system(f'python mapc_dcf/plots/throughput_plot.py -j {json_path} -c {csv_path}')