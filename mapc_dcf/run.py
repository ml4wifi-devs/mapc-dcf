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

def run_scenario(
        key: PRNGKey,
        n_reps: int,
        simulation_length: float,
        scenario: StaticScenario,
):  
    
    for i, _ in enumerate(tqdm(range(n_reps), desc='Repetition'), start=1):
        logging.info(f"Repetition {i}/{n_reps}")

        key, key_channel = jax.random.split(key)
        des_env = simpy.Environment()
        channel = Channel(key_channel, scenario.pos)
        aps = list(scenario.associations.keys())
        for ap in aps:

            key, key_ap = jax.random.split(key)
            clients = jnp.array(scenario.associations[ap])
            mcs = scenario.mcs[ap].item()
            ap = AccessPoint(ap, scenario.pos, mcs, clients, channel, des_env, key_ap)
            ap.start_operation()
        
        des_env.run(until=simulation_length)
        del des_env


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-c', '--config', type=str, default='config.json')
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args.add_argument('-l', '--log', type=str, default='warning')
    args = args.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log.upper()))

    with open(args.config, 'r') as file:
        config = json.load(file)

    key = jax.random.PRNGKey(config['seed'])
    scenario = globals()[config['scenario']](**config['params'])
    run_scenario(key, config['n_reps'], config['simulation_length'], scenario)
