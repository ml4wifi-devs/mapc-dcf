# IEEE 802.11 Multi-AP Distributed Coordination Function (DCF) Simulator

`mapc-dcf` is an implementation of the IEEE 802.11 Distributed Coordination Function (DCF) with support for Multi-Access Point operation. It is a [SimPy](https://simpy.readthedocs.io/en/latest/)-based Discrete Event Simulator (DES) that models the channel and frame exchange.

## Features and Assumptions

- Supports Multi-AP operation.
- Supports the Spatial Reuse (SR) mechanism introduced in the IEEE 802.11ax standard.
- Supports the AMPDU aggregation mechanism.
- Models collision probability by calculating the interference matrix for each device.
- Monitors and logs the history of frame exchanges.
- Outputs a summary of the simulation.
- Supports scenarios with walls.
- Assumes instantaneous and error-free acknowledgments.
- Assumes an ideal MCS selection mechanism.

## Installation

The package can be installed using pip:

```bash
pip install mapc-dcf
```

## Usage

It is best to be used with scenarios defined as in our other [repository](https://github.com/ml4wifi-devs/mapc-optimal-research). In the following example, we show how to use the simulator with a *small_office_scenario* defined in the `mapc_research` package.

```python
from typing import Dict

import simpy
import jax
import jax.numpy as jnp
from mapc_research.envs.scenario_impl import *

from mapc_dcf import AccessPoint, Channel, Logger

# Define the simulation parameters
SIMULATION_LENGTH = 0.1     # seconds
WARMUP_LENGTH = 0.1         # seconds
RESULTS_PATH = "./results"  # Path to save the results (without extension!)
SEED = 42

# Set the random seed
key = jax.random.PRNGKey(SEED)

# Create the SimPy environment
des_env = simpy.Environment()

# Create the logger
logger = Logger(SIMULATION_LENGTH, WARMUP_LENGTH, RESULTS_PATH)

# Define the scenario
scenario = small_office_scenario(d_ap=20, d_sta=2, n_steps=1000)

# Create the channel
key, key_channel = jax.random.split(key)
channel = Channel(
    key=key_channel,
    sr=True,                # Spatial reuse enabled
    pos=scenario.pos,
    walls=scenario.walls    # Scenario with walls
)

# Define the access points
aps: Dict[int, AccessPoint] = {}
for ap_id in scenario.associations:
    key, key_ap = jax.random.split(key)
    clients = jnp.array(scenario.associations[ap_id])
    tx_power = scenario.tx_power[ap_id].item()
    aps[ap_id] = AccessPoint(
        key=key_ap,
        id=ap_id,
        position=scenario.pos,
        tx_power=tx_power,
        mcs=11, # MCS 11, this is legacy, it does not affect
        # the simulation as the MCS is selected ideally inside the DCF
        clients=clients,
        channel=channel,
        des_env=des_env,
        logger=logger
    )

# Start the simulation
for ap_id in aps:
    aps[ap_id].start_operation(run_number=1)   # Run number can be convenient in case of multiple runs
des_env.run(until=WARMUP_LENGTH + SIMULATION_LENGTH)

# Dump the results to the results file
logger.dump_accumulators(run_number=1)
logger.shutdown(config={"scenario": "Optional JSON description of the scenario"})

# Clean up the environment
del des_env
```

## How to reference `MAPC-DCF`?

```
@article{wojnar2025coordinated,
  author={Wojnar, Maksymilian and Ciężobka, Wojciech and Tomaszewski, Artur and Chołda, Piotr and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Haxhibeqiri, Jetmir and Hoebeke, Jeroen and Bellalta, Boris and Zubow, Anatolij and Dressler, Falko and Szott, Szymon},
  title={{Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks}}, 
  year={2025},
}
```
