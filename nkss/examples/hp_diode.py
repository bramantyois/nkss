import numpy as np

from ..nkss import NKSS
from ..components import Port
from ..components import ReactiveComponent
from ..semiconductors.diodemodel import D1N914


class HPDiode(NKSS):
    def __init__(self, sample_rate=44100, name='DiodeClipperSS'):
        super().__init__(num_of_nodes=3, sample_rate=sample_rate, v_supply=9, name=name)

        self.converter_matrix = -np.ones((1, 1))

        self.assign_components()
        self.calc_system()

    def assign_components(self):
        self.nl_instances = D1N914()

        res1 = ReactiveComponent(value=2.2e3, neg=1, pos=0)
        self.set_resistors([res1])

        cap1 = ReactiveComponent(value=0.47e-6, neg=2, pos=1)
        cap2 = ReactiveComponent(value=0.01e-6, neg=self.ground_node, pos=2)
        self.set_capacitors(capacitors=[cap1, cap2])

        in_vol = Port(neg=self.ground_node, pos=0)
        self.set_input_sources([in_vol])

        diode_i = Port(neg=self.ground_node, pos=2)
        self.set_nl_sources([diode_i])

        out_vol = Port(pos=2, neg=self.ground_node)
        self.set_output_ports([out_vol])
    #
    # def compute_i(self, vn: np.ndarray) -> np.ndarray:
    #     return self.nl_instances.compute_i(vn)
    #
    # def compute_ji(self, vn: np.ndarray) -> np.ndarray:
    #     return self.nl_instances.compute_ji(vn)
