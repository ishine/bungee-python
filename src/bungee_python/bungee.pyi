import numpy as np
from numpy.typing import NDArray

class Bungee:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        speed: float = 1.0,
        pitch: float = 1.0,
        log2_synthesis_hop_adjust: int = 0,
    ) -> None: ...
    def process(self, input: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def set_debug(self, enable:bool=False):...
    def get_debug()->bool:...