from typing import Set, Optional

import random
import logging
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Array, PRNGKey
from intervaltree import Interval, IntervalTree

from mapc_dcf.constants import *
from mapc_sim.utils import logsumexp_db, tgax_path_loss

tfd = tfp.distributions


class WiFiFrame():

    def __init__(self, src: int, dst: int, mcs: int, size: int = FRAME_LEN_INT) -> None:
        self.src = src
        self.dst = dst
        self.mcs = mcs
        self.size = size
        self.duration = self.size / (DATA_RATES[mcs].item() * 1e6)
    
    def materialize(self, start_time: float, tx_power: float):
        self.start_time = start_time
        self.end_time = start_time + self.duration
        self.tx_power = tx_power

class Channel():

    def __init__(self, key: PRNGKey, pos: Array, walls: Optional[Array] = None) -> None:
        self.key = key
        self.pos = pos
        self.n_nodes = self.pos.shape[0]
        self.walls = walls if walls is not None else jnp.zeros((self.n_nodes, self.n_nodes))
        self.frames_history = IntervalTree()

    def is_idle(self, time: float, ap: int) -> bool:
        """
        Check if the signal level at the AP is below the CCA threshold, in other words, check if the channel is idle.

        Parameters
        ----------
        time : float
            Time at which to check the channel.
        ap : int
            AP index for which to check the channel.

        Returns
        -------
        bool
            Whether the channel is idle or not.
        """

        return random.random() < 0.95 # TODO Remove this line

        overlapping_frames = self.frames_history[time]

        tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
        mcs_at_time = jnp.zeros((self.n_nodes,))
        tx_power_at_time = jnp.zeros((self.n_nodes,))
        for frame in overlapping_frames:
            tx_matrix_at_time[frame.src, frame.dst] = 1
            mcs_at_time[frame.src] = frame.mcs
            tx_power_at_time[frame.src] = frame.tx_power
        
        sinr_at_ap = self._get_sinr(self.key, tx_matrix_at_time, tx_power_at_time)[ap].item()
        return sinr_at_ap < CCA_THRESHOLD

    def is_idle_for(self, time: float, duration: float, ap: int) -> bool:
        """
        Check if the signal level at the AP is below the CCA threshold for a given duration. In other words,
        check if the channel is idle for a given duration.

        Parameters
        ----------
        time : float
            Time at which to check the channel.
        duration : float
            Required duration for the channel to be idle.
        ap : int
            AP index for which to check the channel.

        Returns
        -------
        bool
            Whether the channel is idle for the given duration or not.
        """

        overlapping_frames = self.frames_history.overlap(max(0., time - duration), time)
        logging.info(f"AP{ap}: Overlapping frames: {overlapping_frames}")
        start_times = {interval.data.start_time for interval in overlapping_frames}
        end_times = {interval.data.end_time for interval in overlapping_frames}
        timepoints = jnp.array(sorted(list(start_times.union(end_times))))
        middlepoints = (timepoints[:-1] + timepoints[1:]) / 2
        for middlepoint in middlepoints:
            if not self.is_idle(middlepoint, ap):
                return False
        return True

    
    def send_frame(self, frame: WiFiFrame, start_time: float, tx_power: float) -> None:
        frame.materialize(start_time, tx_power)
        self.frames_history.add(Interval(start_time, frame.end_time, frame))

    def succesfully_transmitted(self, frame: WiFiFrame) -> bool:
        # TODO: Implement
        return random.random() < 0.5
    
    def _get_sinr(self, key: PRNGKey, tx: Array, tx_power: Array) -> Array:

        distance = jnp.sqrt(jnp.sum((self.pos[:, None, :] - self.pos[None, ...]) ** 2, axis=-1))
        distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

        signal_power = tx_power[:, None] - tgax_path_loss(distance, self.walls)

        interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
        a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
        b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
        interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

        sinr = signal_power - interference
        sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=DEFAULT_SIGMA).sample(seed=key)
        sinr = (sinr * tx).sum(axis=1)

        return sinr


    def _get_overlapping_frames(self, start_time: float, end_time: float) -> Set[WiFiFrame]:
        return self.frames_history[start_time:end_time]
        