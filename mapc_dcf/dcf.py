from typing import Callable
from chex import PRNGKey

import logging
import jax
import jax.numpy as jnp
import simpy

from mapc_dcf.constants import *
from mapc_dcf.channel import Channel, WiFiFrame


class DCF():

    def __init__(
            self,
            ap: int,
            des_env: simpy.Environment,
            channel: Channel,
            frame_generator: Callable[[], WiFiFrame],
            key: PRNGKey
    ) -> None:
        self.ap = ap
        self.des_env = des_env
        self.channel = channel
        self.frame_generator = frame_generator
        self.cw = 2**CW_EXP_MIN
        self.tx_power = DEFAULT_TX_POWER
        self.key = key
    
    def start_operation(self):
        self.des_env.process(self.run())
    
    def wait_for_one_slot(self):
        yield self.des_env.timeout(SLOT_TIME)

    def run(self):

        logging.info(f"AP{self.ap}: DCF started")

        # While ready to send frames
        while True:
            frame = self.frame_generator()

            # Try sending the frame until transmitted successfully
            frame_sent_successfully = False
            while not frame_sent_successfully:
                
                # First condition: channel is idle
                channel_idle = False
                while not channel_idle:

                    # Wait for DIFS
                    while not channel_idle:
                        # TODO Verify the division by 10. Can we do it in a better way?
                        yield self.des_env.timeout(DIFS / 10)
                        channel_idle = self.channel.is_idle_for(self.des_env.now, DIFS, frame.src)
                    logging.info(f"AP{self.ap}: Channel idle for DIFS")
                    
                    # Initialize backoff counter
                    key_backoff, self.key = jax.random.split(self.key)
                    backoff_counter = jax.random.randint(key_backoff, shape=(1,), minval=2**CW_EXP_MIN, maxval=self.cw+1).item()

                    # Second condition: backoff counter is zero
                    while channel_idle and backoff_counter > 0:
                        logging.info(f"AP{self.ap}: Backoff counter: {backoff_counter}")

                        # If not, wait for one slot and check again both conditions
                        yield self.des_env.process(self.wait_for_one_slot())
                        backoff_counter -= 1
                        channel_idle = self.channel.is_idle(self.des_env.now, frame.src)
                
                logging.info(f"AP{self.ap}: Backoff counter zero. Sending frame")
                
                # If both conditions are met, send the frame
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line
                self.channel.send_frame(frame, self.des_env.now, self.tx_power)
                yield self.des_env.timeout(frame.duration)  # TODO Include ACK time
                frame_sent_successfully = self.channel.is_succesfully_transmitted(frame)
                yield self.des_env.timeout(SIFS)            # TODO Try to remove this line

                # Act according to the transmission result
                if frame_sent_successfully:
                    frame_sent_successfully = True
                    self.cw = 2**CW_EXP_MIN
                else:
                    self.cw = min(2*self.cw, 2**CW_EXP_MAX)
    