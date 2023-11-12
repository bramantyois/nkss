import numpy as np

from ..nkss import NKSS
from ..components import Port
from ..components import ReactiveComponent

from ..semiconductors.triodemodel import TB12AX7, TB12AX7Improved


class SimpleTube(NKSS):
    def __init__(self, sample_rate=44100, name='SimpleTube', use_improved_tube=False):
        super().__init__(num_of_nodes=5, sample_rate=sample_rate, v_supply=300, name=name)
        self.use_improved_tube = use_improved_tube
        self.assign_components()
        self.calc_system()

        self.converter_matrix = np.array([
            [-1, 0],
            [0, -1]])

    def assign_components(self):
        if self.use_improved_tube:
            self.nl_instances = TB12AX7Improved()
        else:
            self.nl_instances = TB12AX7()
            
        res1 = ReactiveComponent(100, pos=0, neg=1)
        res2 = ReactiveComponent(27e3, pos=2, neg=3)
        res3 = ReactiveComponent(750, pos=4, neg=self.ground_node)
        res4 = ReactiveComponent(1e6, pos=0, neg=self.ground_node)

        self.set_resistors([res1, res2, res3, res4])

        cap1 = ReactiveComponent(1e-3, pos=4, neg=self.ground_node)
        self.set_capacitors([cap1])

        in_port = Port(pos=0, neg=self.ground_node)
        supply_port = Port(pos=2, neg=self.ground_node)

        self.set_input_sources([in_port, supply_port])

        ia_port = Port(pos=4, neg=3)
        ig_port = Port(pos=4, neg=1)

        self.set_nl_sources([ia_port, ig_port])

        out_port = Port(pos=3, neg=self.ground_node)
        self.set_output_ports([out_port])
