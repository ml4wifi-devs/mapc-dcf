from typing import Set, Optional, Tuple

import logging
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Array, Scalar, PRNGKey
from intervaltree import Interval, IntervalTree

from mapc_dcf.constants import *
from mapc_dcf.utils import logsumexp_db, tgax_path_loss, timestamp

tfd = tfp.distributions


class AMPDU():

    ampdu_duration: float = TAU
    n_ampdu: int
    ampdu_size: int
    pdu_duration: float

    def __init__(self, id: int, src: int, dst: int, tx_power: float, mcs: int, sr: bool, payload_size: int = FRAME_LEN_INT) -> None:
        self.id = id
        self.src = src
        self.dst = dst
        self.mcs = mcs
        self.tx_power = tx_power
        self.pdu_size = payload_size
        self.sr = sr
    

    def materialize(self, start_time: float, tx_power: float, retransmission: int) -> None:
        """
        Materialize the WiFi frame by setting its start time, end time, and transmission power.
        End time is calculated based on the predefined frame duration. After materialization,
        the frame can be sent over the channel.

        Parameters
        ----------
        start_time : float
            Transmission start time.
        tx_power : float
            Transmission power.
        retransmission : int
            Retransmission number.
        """
        
        # Reduce the tx power if operating in spatial reuse mode
        if self.sr:
            self.tx_power = tx_power
        else:
            self.tx_power = DEFAULT_TX_POWER
        
        self.start_time = start_time
        self.end_time = start_time + self.ampdu_duration
        self.retransmission = retransmission


class Channel():

    def __init__(self, key: PRNGKey, sr: bool, pos: Array, walls: Optional[Array] = None) -> None:
        self.key = key
        self.pos = pos
        self.is_sr_on = sr
        self.cca_threshold = OBSS_PD_MAX if self.is_sr_on else CCA_THRESHOLD
        self.n_nodes = self.pos.shape[0]
        self.walls = walls if walls is not None else jnp.zeros((self.n_nodes, self.n_nodes))
        self.tx_history = IntervalTree()
        self.frames = {}
        

    def is_idle(self, time: float, ap: int, sender_tx_power: float) -> bool:
        """
        Check if the signal level at the AP is below the CCA threshold, in other words, check if the channel is idle.

        Parameters
        ----------
        time : float
            Time at which to check the channel.
        ap : int
            AP index for which to check the channel.
        sender_tx_power : float
            Transmission power of the sender.

        Returns
        -------
        bool
            Whether the channel is idle or not.
        """

        # Get frames that occupy the channel at the given time
        overlapping_frames = self.tx_history[time]

        if not overlapping_frames:
            return True

        # Set the transmission matrix and transmission power from the current frames in the channel
        tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
        tx_power_at_time = jnp.zeros((self.n_nodes,))
        for frame_interval in overlapping_frames:

            overlapping_frame: AMPDU = self.frames[frame_interval.data]
            tx_matrix_at_time = tx_matrix_at_time.at[overlapping_frame.src, overlapping_frame.dst].set(1)
            tx_power_at_time = tx_power_at_time.at[overlapping_frame.src].set(overlapping_frame.tx_power)
        
        # If ap was transmitting at the given time, the channel is busy
        if tx_matrix_at_time[ap].sum() > 0:
            return False
        
        # Set the transmission from AP to itself, to be used in the signal level calculation
        tx_matrix_at_time = tx_matrix_at_time.at[ap, ap].set(1)
        tx_power_at_time = tx_power_at_time.at[ap].set(sender_tx_power)
        
        # Get the energy level calculated from the interference matrix at the AP
        _, interference = self._get_signal_power_and_interference(tx_matrix_at_time, tx_power_at_time)
        energy_detected = interference[ap].item()

        # Channel is idle if the energy level at the AP is below the CCA threshold
        return energy_detected < self.cca_threshold


    def is_idle_for(self, time: float, duration: float, ap: int, sender_tx_power: float) -> bool:
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

        # Transform time and duration to low and high times
        low_time, high_time = time - duration, time

        # If the simulation does not last long enough, assume the channel busy
        if low_time < 0:
            return False

        # Get the overlapping frames within the given time interval
        overlapping_ids = self.tx_history.overlap(low_time, high_time)

        # We asses the channel as idle if it is idle in all the middlepoints of the given interval
        middlepoints, _ = self._get_middlepoints_and_durations(overlapping_ids, low_time, high_time)
        logging.debug(f"AP{ap}:t{time:.9f}\t Overlapping frames: {overlapping_ids}\n" + "\t"*8 + f"Middlepoints: {middlepoints}")
        for middlepoint in middlepoints:

            if not self.is_idle(middlepoint, ap, sender_tx_power):
                return False
        
        return self.is_idle(high_time, ap, sender_tx_power)

    
    def send_frame(self, frame: AMPDU, start_time: float, retransmission: int) -> None:
        """
        Send a WiFi frame over the channel.

        Parameters
        ----------
        frame : WiFiFrame
            The 802.11 frame to send.
        start_time : float
            The simulation time at which the frame transmission starts.
        tx_power : float
            The transmission power at which the frame is sent.
        """

        tx_power = self._get_reduced_tx_power(start_time, frame.src, frame.tx_power)
        frame.materialize(start_time, tx_power, retransmission)
        self.tx_history.add(Interval(start_time, frame.end_time, (frame.src, frame.id)))
        self.frames[(frame.src, frame.id)] = frame

        # Log the frame transmission
        logging.info(f"AP{frame.src}:{timestamp(start_time)}\t Sending frame to {frame.dst} with MCS {frame.mcs} and tx power {frame.tx_power}")


    def is_tx_successful(self, frame: AMPDU) -> int:
        """
        Check if a AMPDU transmission was successful. Return the number of successful transmissions.

        Parameters
        ----------
        frame : WiFiFrame
            The transmitted AMPDU WiFi frame.

        Returns
        -------
        int
            The number of successful transmissions within the AMPDU.
        """

        # Calculate the ideal MCS for the frame
        frame.mcs = self._get_ideal_mcs(frame, frame.end_time)
        
        # Based on the selected MCS, calculate AMPDU attributes
        frame.n_ampdu = int(jnp.round(DATA_RATES[frame.mcs] * 1e6 * TAU / frame.pdu_size).item())
        frame.ampdu_size = frame.n_ampdu * frame.pdu_size
        frame.pdu_duration = frame.pdu_size / (DATA_RATES[frame.mcs].item() * 1e6)

        
        # Get the overlapping frames with the transmitted frame
        frame_start_time, frame_end_time, frame_duration = frame.start_time, frame.end_time, frame.ampdu_duration
        overlapping_frames = self.tx_history.overlap(frame_start_time, frame_end_time) - {Interval(frame_start_time, frame_end_time, (frame.src, frame.id))}
        overlapping_frames_tree = IntervalTree(overlapping_frames)

        # Iterate over the AMPDU frames to check the success of each PDU. Stop if a PDU fails, and return the number of successful PDUs
        pdu_iter = 0
        n_successful_txs = 0
        while pdu_iter < frame.n_ampdu:
            pdu_start_time = frame_start_time + pdu_iter * frame.pdu_duration
            pdu_end_time = pdu_start_time + frame.pdu_duration

            # Calculate the middlepoints and durations in reference to the current PDU
            middlepoints, durations = self._get_middlepoints_and_durations(overlapping_frames, pdu_start_time, pdu_end_time)
        
            # Calculate the success probability at the middlepoints
            middlepoints_success_probs = []
            for middlepoint, duration in zip(middlepoints, durations):

                # Get the concurrent frames at the middlepoint
                middlepoint_overlapping_frames = overlapping_frames_tree[middlepoint]
                middlepoint_overlapping_frames = middlepoint_overlapping_frames.union({Interval(pdu_start_time, pdu_end_time, (frame.src, frame.id))})

                # Build the transmission matrix, MCS, and transmission power at the middlepoint
                tx_matrix_at_middlepoint = jnp.zeros((self.n_nodes, self.n_nodes))
                mcs_at_middlepoint = jnp.zeros((self.n_nodes,), dtype=int)
                tx_power_at_middlepoint = jnp.zeros((self.n_nodes,))
                for frame_interval in middlepoint_overlapping_frames:
                    
                    iter_frame = self.frames[frame_interval.data]
                    tx_matrix_at_middlepoint = tx_matrix_at_middlepoint.at[iter_frame.src, iter_frame.dst].set(1)
                    mcs_at_middlepoint = mcs_at_middlepoint.at[iter_frame.src].set(iter_frame.mcs)
                    tx_power_at_middlepoint = tx_power_at_middlepoint.at[iter_frame.src].set(iter_frame.tx_power)
                
                # Calculate the success probability at the current middlepoint
                self.key, key_per = jax.random.split(self.key)
                middlepoints_success_probs.append(self._get_success_probability(
                    key_per,
                    tx_matrix_at_middlepoint,
                    mcs_at_middlepoint,
                    tx_power_at_middlepoint,
                    frame.src
                ))
            middlepoints_success_probs = jnp.array(middlepoints_success_probs)
        
            # Aggregate the probabilities
            self.key, key_uniform = jax.random.split(self.key)
            success = jnp.all(jax.random.uniform(key_uniform, shape=middlepoints_success_probs.shape) < middlepoints_success_probs).item()
            n_successful_txs += int(success)
            pdu_iter += 1
        
        return int(n_successful_txs)
    

    def _get_middlepoints_and_durations(
            self,
            overlapping_frames_ids: Set[Interval],
            low_time: float,
            high_time: float
    ) -> Tuple[Array, Array]:
        
        start_times = {self.frames[interval.data].start_time for interval in overlapping_frames_ids if self.frames[interval.data].start_time > low_time}
        start_times = start_times.union({low_time})
        end_times = {self.frames[interval.data].end_time for interval in overlapping_frames_ids if self.frames[interval.data].end_time < high_time}
        end_times = end_times.union({high_time})
        timepoints = jnp.array(sorted(list(start_times.union(end_times))))
        durations = timepoints[1:] - timepoints[:-1]

        return (timepoints[:-1] + timepoints[1:]) / 2, durations
    

    def _get_signal_power_and_interference(self, tx: Array, tx_power: Array) -> Tuple[Array, Array]:

        distance = jnp.sqrt(jnp.sum((self.pos[:, None, :] - self.pos[None, ...]) ** 2, axis=-1))
        distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

        signal_power = tx_power[:, None] - tgax_path_loss(distance, self.walls)

        interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
        a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
        b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
        interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

        return signal_power, interference

    
    def get_signal_level(self, time: float, ap: int, sender_tx_power: float) -> Scalar:

        # TODO: The sender_tx_power probably doesn't matter here. The Interference at the sender does
        # not depend on the sender's transmission power. Can be removed from the function signature?

        # Get frames that occupy the channel at the given time
        overlapping_frames_ids = self.tx_history[time]

        # If no frames are occupying the channel, return the noise floor
        if not overlapping_frames_ids:
            return NOISE_FLOOR

        # Set the transmission matrix and transmission power from the current frames in the channel
        tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
        tx_power_at_time = jnp.zeros((self.n_nodes,))
        for frame_interval in overlapping_frames_ids:

            overlapping_frame: AMPDU = self.frames[frame_interval.data]
            tx_matrix_at_time = tx_matrix_at_time.at[overlapping_frame.src, overlapping_frame.dst].set(1)
            tx_power_at_time = tx_power_at_time.at[overlapping_frame.src].set(overlapping_frame.tx_power)
        
        # Set the transmission from AP to itself, to be used in the signal level calculation
        tx_matrix_at_time = tx_matrix_at_time.at[ap, ap].set(1)
        tx_power_at_time = tx_power_at_time.at[ap].set(sender_tx_power)

        # Calculate the energy detected (interference signal level) at the AP
        _, interference = self._get_signal_power_and_interference(tx_matrix_at_time, tx_power_at_time)

        return interference[ap].item()
    

    def _calculate_sinr(self, key: PRNGKey, signal_power: Array, interference: Array, tx: Array) -> Array:

        sinr = signal_power - interference
        sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=DEFAULT_SIGMA).sample(seed=key)
        sinr = (sinr * tx).sum(axis=1)

        return sinr
    

    def _get_success_probability(self, key: PRNGKey, tx: Array, mcs: Array, tx_power: Array, ap_src: int) -> Scalar:

        signal_power, interference = self._get_signal_power_and_interference(tx, tx_power)
        sinr = self._calculate_sinr(key, signal_power, interference, tx)
        sdist = tfd.Normal(loc=MEAN_SNRS[mcs], scale=2.)
        success_prob = sdist.cdf(sinr)

        return success_prob[ap_src].item()
    

    def _get_ideal_mcs(self, frame: AMPDU, start_time: float) -> int:
            
            self.key, key_mcs = jax.random.split(self.key)
            
            # Simulate the channel at the start time to obtain the ideal MCS
            tx_matrix_at_time = jnp.zeros((self.n_nodes, self.n_nodes))
            tx_power_at_time = jnp.zeros((self.n_nodes,))
            overlapping_frames_ids = self.tx_history[start_time]
            overlapping_frames_ids = overlapping_frames_ids.union({Interval(start_time, TAU, (frame.src, frame.id))})
            for frame_interval in overlapping_frames_ids:

                overlapping_frame: AMPDU = self.frames[frame_interval.data]
                tx_matrix_at_time = tx_matrix_at_time.at[overlapping_frame.src, overlapping_frame.dst].set(1)
                tx_power_at_time = tx_power_at_time.at[overlapping_frame.src].set(overlapping_frame.tx_power)
    
            signal_power, interference = self._get_signal_power_and_interference(tx_matrix_at_time, tx_power_at_time)
            sinr = self._calculate_sinr(key_mcs, signal_power, interference, tx_matrix_at_time)
            expected_data_rate = DATA_RATES[:, None] * tfd.Normal(loc=MEAN_SNRS[:, None], scale=2.).cdf(sinr)
            mcs = jnp.argmax(expected_data_rate, axis=0)[frame.src].item()

            return mcs
    
    def _get_reduced_tx_power(self, time: float, ap: int, sender_tx_power: float) -> float:

        energy_detected = self.get_signal_level(time, ap, sender_tx_power)
        logging.info(f"AP{ap}:{timestamp(time)}\t Energy detected: ED = {energy_detected:.2f}")
        if energy_detected <= OBSS_PD_MIN:
            return DEFAULT_TX_POWER
        elif OBSS_PD_MAX < energy_detected:
            logging.info(f"AP{ap}:{timestamp(time)}\t Energy detected above OBSS_PD_MAX = {OBSS_PD_MAX}!")
        return DEFAULT_TX_POWER - (energy_detected - OBSS_PD_MIN)

        