from typing import Callable
from chex import PRNGKey

import jax
import jax.numpy as jnp

from mapc_dcf.constants import *


class Channel():

    def __init__(self) -> None:
        pass

    def is_idle(self):
        # TODO: Implement
        pass

    def is_idle_for(self, duration):
        # TODO: Implement
        pass


class WiFiFrame():
    # TODO: Implement

    def __init__(self) -> None:
        pass


class DCF():

    def __init__(self, channel: Channel, frame_generator: Callable[[], WiFiFrame], key: PRNGKey) -> None:
        self.channel = channel
        self.frame_generator = frame_generator
        self.cw = 2**CW_EXP_MIN
        self.key = key
    
    def wait_for_one_slot(self):
        # TODO: Implement
        pass

    def send_frame(self, frame: WiFiFrame):
        # TODO: Implement
        pass

    def start_operation(self):

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
                        channel_idle = self.channel.is_idle_for(DIFS)
                    
                    # Initialize backoff counter
                    key_backoff, self.key = jax.random.split(self.key)
                    backoff_counter = jax.random.randint(key_backoff, shape=(1,), minval=2**CW_EXP_MIN, maxval=self.cw+1)

                    # Second condition: backoff counter is zero
                    while channel_idle and backoff_counter > 0:

                        # If not, wait for one slot and check again both conditions
                        self.wait_for_one_slot()
                        backoff_counter -= 1
                        channel_idle = self.channel.is_idle()
                
                # If both conditions are met, send the frame
                frame_sent_successfully = self.send_frame(frame)
                if frame_sent_successfully:

                    # Reset the contention window on success
                    frame_sent_successfully = True
                    self.cw = 2**CW_EXP_MIN
                else:

                    # Double the contention window on failure
                    self.cw = min(2*self.cw, 2**CW_EXP_MAX)
    