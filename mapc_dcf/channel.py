from mapc_dcf.constants import DATA_RATES, FRAME_LEN_INT


class WiFiFrame():
    # TODO: Implement

    def __init__(self, src: int, dst: int, mcs: int, size: int = FRAME_LEN_INT) -> None:
        self.src = src
        self.dst = dst
        self.mcs = mcs
        self.size = size
        self.duration = self.size / (DATA_RATES[mcs] * 1e6)
    
    def set_start_time(self, start_time: float):
        self.start_time = start_time

class Channel():

    def __init__(self) -> None:
        pass

    def is_idle(self):
        # TODO: Implement
        pass

    def is_idle_for(self, duration):
        # TODO: Implement
        pass