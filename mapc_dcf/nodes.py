import jax
from chex import Array, PRNGKey

from mapc_dcf.channel import Channel, WiFiFrame
from mapc_dcf.dcf import DCF

class AccessPoint():

    def __init__(self, id: int, channel: Channel, position: Array, mcs: int, clients: Array, key: PRNGKey) -> None:
        self.key, key_dcf = jax.random.split(key)
        self.id = id
        self.position = position
        self.mcs = mcs
        self.clients = clients
        self.dcf = DCF(channel, self.frame_generator, key_dcf)
    
    def frame_generator(self) -> WiFiFrame:
        self.key, key_frame = jax.random.split(self.key)
        dst = jax.random.choice(key_frame, self.clients).item()
        return WiFiFrame(self.id, dst, self.mcs)
    
    def start_operation(self):
        self.dcf.start_operation()
