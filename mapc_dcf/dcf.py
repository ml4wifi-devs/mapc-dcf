from typing import Callable, Optional
from chex import PRNGKey

import logging
import jax
import jax.numpy as jnp
import simpy

from mapc_dcf.constants import *
from mapc_dcf.channel import Channel, WiFiFrame
from mapc_dcf.logger import Logger


class DCF():

    def __init__(
            self,
            key: PRNGKey,
            ap: int,
            des_env: simpy.Environment,
            channel: Channel,
            logger: Logger,
            frame_generator: Callable[[], WiFiFrame],
    ) -> None:
        self.key = key
        self.ap = ap
        self.des_env = des_env
        self.channel = channel
        self.logger = logger
        self.frame_generator = frame_generator
        self.retry_limit = RETRY_LIMIT if RETRY_LIMIT is not None else jnp.inf
        self.cw = 2**CW_EXP_MIN

        # TODO Temporary, to be removed
        self.total_attempts = 0
        self.total_collisions = 0
    

    def timestamp(self):
        t = f"{10**6 * self.des_env.now:.9f} us"
        leading_zeros = max(6 - len(t.split(".")[0]), 0)
        return "0" * leading_zeros + t
    

    def start_operation(self, run_number: int):
        self.run_number = run_number
        self.des_env.process(self._run())
    

    def _wait_for_one_slot(self):
        yield self.des_env.timeout(SLOT_TIME)
    

    def _wait_for_difs(self, frame: WiFiFrame):
        
        idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src, frame.tx_power)
        while not idle:
            logging.info(f"AP{self.ap}:{self.timestamp()}\t DIFS: waiting for a SLOT_TIME = 9 us")
            yield self.des_env.timeout(SLOT_TIME)
            idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src, frame.tx_power)
        logging.info(f"AP{self.ap}:{self.timestamp()}\t DIFS: Channel idle for DIFS = 34 us")
    

    def _freeze_backoff(self, frame: WiFiFrame, time_to_backoff: int):

        logging.info(f"AP{self.ap}:{self.timestamp()}\t Channel busy, freezing backoff at TTB = {time_to_backoff}")
        yield self.des_env.process(self._wait_for_difs(frame))
        logging.info(f"AP{self.ap}:{self.timestamp()}\t Channel idle, reactivating backoff at TTB = {time_to_backoff}")
        # and reactivated after the channel is sensed idle again for a guard period.


    def _try_sending(self, frame: WiFiFrame, retry_count: int):

        logging.info(f"AP{self.ap}:{self.timestamp()}\t Attempting to send frame. Retry count: {retry_count}")
        
        # If the retry limit is reached, the frame is dropped
        if retry_count > self.retry_limit:
            logging.info(f"AP{self.ap}:{self.timestamp()}\t Retry limit reached, dropping frame")
            return

        # Initialize a random backoff interval
        key_backoff, self.key = jax.random.split(self.key)
        initialized_backoff = jax.random.randint(key_backoff, shape=(1,), minval=0, maxval=self.cw).item()
        logging.info(f"AP{self.ap}:{self.timestamp()}\t TTB initialized as {initialized_backoff} from [0, {self.cw}) interval")

        # The bakoff countdown with the freeze-and-reactivation mechanism
        time_to_backoff = initialized_backoff
        while time_to_backoff > 0:
            logging.info(f"AP{self.ap}:{self.timestamp()}\t TTB = {time_to_backoff}")

            # The backoff time counter is decremented as long as the channel is sensed idle.
            if self.channel.is_idle(self.des_env.now, frame.src, frame.tx_power):
                yield self.des_env.process(self._wait_for_one_slot())
                if self.channel.is_idle(self.des_env.now - 0.5 * SLOT_TIME, frame.src, frame.tx_power):
                    time_to_backoff -= 1
                else:
                    yield self.des_env.process(self._freeze_backoff(frame, time_to_backoff))    
            # It is frozen when activities (i.e. packet transmissions) are detected on the channel
            else:
                yield self.des_env.process(self._freeze_backoff(frame, time_to_backoff))
                # and reactivated after the channel is sensed idle again for a guard period. 
        
        # The frame is sent to the channel
        logging.info(f"AP{self.ap}:{self.timestamp()}\t TTB reached zero (TTB = {time_to_backoff}) and the channel is idle. Sending frame to {frame.dst}")
        self.channel.send_frame(frame, self.des_env.now, retry_count)
        yield self.des_env.timeout(frame.duration + SIFS) # The SIFS is the lower bound of the ACK timeout

        # The channel returns the number of successful transmissions within the AMPDU
        successful_txs = self.channel.is_tx_successful(frame)

        # Log the transmission attempt
        self.total_attempts += 1
        self.logger.log(self.des_env.now, self.run_number, frame, self.cw, initialized_backoff, successful_txs, frame.n_ampdu - successful_txs)

        # If the packet transmission is successful, the size of the contention window resets to the minimum value
        if successful_txs > 0:
            logging.info(f"AP{self.ap}:{self.timestamp()}\t At leas one successfull TX, resetting CW to {self.cw}")
            self.cw = 2**CW_EXP_MIN
        # and is doubled on failure
        else:
            logging.info(f"AP{self.ap}:{self.timestamp()}\t Collision, increasing CW to {self.cw}")
            self.total_collisions += 1
            self.cw = min(2*self.cw, 2**CW_EXP_MAX)
            yield self.des_env.process(self._try_sending(frame, retry_count + 1))


    def _run(self):
        """
        The simplified 802.11 DCF algorithm. Diagram of the algorithm can be found in
        the documentation `\\docs\\diagrams\\DCF_simple.pdf`.
        """

        logging.info(f"AP{self.ap}:{self.timestamp()}\t DCF running")

        # Network is assumed to be saturated, there is always a frame to send
        while True:

            # Generate a frame
            frame = self.frame_generator()

            # Whenever a station has a packet to send it should defer its transmission for a DIFS guard period
            yield self.des_env.process(self._wait_for_difs(frame))
            yield self.des_env.process(self._try_sending(frame, 0))
