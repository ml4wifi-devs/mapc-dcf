import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import logging
from argparse import ArgumentParser

import jax
from chex import PRNGKey
import simpy
from tqdm import tqdm

from mapc_mab.envs.static_scenarios import *
from mapc_dcf.channel import Channel
from mapc_dcf.nodes import AccessPoint
from mapc_dcf.logger import Logger

def run_scenario(
        key: PRNGKey,
        n_runs: int,
        simulation_length: float,
        warmup_length: float,
        scenario: StaticScenario,
        logger: Logger
):  

    for run, _ in enumerate(tqdm(range(n_runs), desc='Repetition'), start=1):
        logging.info(f"Run {run}/{n_runs}")

        key, key_scenario = jax.random.split(key)
        des_env = simpy.Environment()
        channel = Channel(key_scenario, scenario.pos, walls=scenario.walls)
        aps = {ap_id: None for ap_id in list(scenario.associations.keys())}
        for ap in aps.keys():

            key_scenario, key_ap = jax.random.split(key_scenario)
            clients = jnp.array(scenario.associations[ap])
            mcs = scenario.mcs[ap].item()
            aps[ap] = AccessPoint(key_ap, ap, scenario.pos, mcs, clients, channel, des_env, logger)
            aps[ap].start_operation(run)
        
        des_env.run(until=warmup_length + simulation_length)

        # TODO to be removed once debugged or improve logger
        total = 0
        collisions = 0
        for ap in aps.keys():
            total_ap = aps[ap].dcf.total_frames
            collisions_ap = aps[ap].dcf.total_collisions
            print(f"Collisions:AP{ap}: {collisions_ap / total_ap:.3f} (of {total_ap})")
            total += total_ap
            collisions += collisions_ap
        print(f"Collisions: {collisions / total:.3f} (of {total})")

        del des_env

    logger.shutdown()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config_path',    type=str, default='default_config.json')
    args.add_argument('-r', '--results_path',   type=str, default='all_results')
    args.add_argument('-l', '--log_level',      type=str, default='warning')
    args = args.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

    with open(args.config_path, 'r') as file:
        config = json.load(file)
    
    key = jax.random.PRNGKey(config['seed'])

    logger = Logger(args.results_path, config['n_runs'], config['simulation_length'], config['warmup_length'], **config['logger_params'])
    scenario = globals()[config['scenario']](**config['scenario_params'])
    run_scenario(key, config['n_runs'], config['simulation_length'], config['warmup_length'], scenario, logger)
