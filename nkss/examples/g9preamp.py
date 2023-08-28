from src.nodalkstatespace.nodalkstatespace import NodalKStateSpace
from src.nodalkstatespace.port import Port
from src.nodalkstatespace.reactivecomponent import ReactiveComponent

from src.nodalkstatespace.triodemodel import TB12AX7

import matplotlib.pyplot as plt
import numpy as np


class G9PreampFirstStage(NodalKStateSpace):
    def __init__(self, sample_rate=44100, v_supply=250, name='TriodeHPSS'):
        super().__init__(num_of_nodes=6, sample_rate=sample_rate, v_supply=v_supply, name=name)

        self.assign_components()
        self.calc_system()

        self.converter_matrix = np.array([
            [-1, 0],
            [0, -1]])

    def assign_components(self):
        self.nl_instances = TB12AX7()

        # resistors
        res1 = ReactiveComponent(1e6, pos=1, neg=2)
        res2 = ReactiveComponent(47e3, pos=2, neg=self.ground_node)
        res8 = ReactiveComponent(2.2e3, pos=5, neg=2)
        res9 = ReactiveComponent(47e3, pos=3, neg=4)
        self.set_resistors([res1, res2, res8, res9])

        #capacitors
        cap2 = ReactiveComponent(220e-9, pos=0, neg=1)
        self.set_capacitors([cap2])

        # supplies
        in_port = Port(pos=0, neg=self.ground_node)
        supply_port = Port(pos=3, neg=self.ground_node)

        self.set_input_sources([in_port, supply_port])

        # non-linears
        ia_tube1 = Port(pos=5, neg=4)
        ig_tube1 = Port(pos=5, neg=1)
        self.set_nl_sources([ia_tube1, ig_tube1])

        # output
        out_port = Port(pos=4, neg=self.ground_node)
        self.set_output_ports([out_port])

    # def compute_i(self, vn: np.ndarray) -> np.ndarray:
    #     #v_neg = self.neg_mat@vn
    #     return self.nl_instances.compute_i(vn)
    #
    # def compute_ji(self, vn: np.ndarray) -> np.ndarray:
    #     #v_neg = self.neg_mat@vn
    #     return self.nl_instances.compute_ji(vn)



