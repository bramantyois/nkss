from operator import ne
import sys 
sys.path.insert(0, '../../')

from src.nodalkstatespace.nodalkstatespace import NodalKStateSpace
from src.nodalkstatespace.reactivecomponent import ReactiveComponent
from src.nodalkstatespace.port import Port

from src.nodalkstatespace.triodemodel import TB12AX7

import matplotlib.pyplot as plt
import numpy as np


class SRPPPower(NodalKStateSpace):
    def __init__(self, sample_rate=44100, v_supply=200, name='PultecPowerSRPP'):
        super().__init__(num_of_nodes=9, sample_rate=sample_rate, v_supply=v_supply, name=name)

        self.assign_components()
        self.calc_system()

        self.converter_matrix = -np.eye(4)

        self.nl_instance_a = TB12AX7()
        self.nl_instance_b = TB12AX7()

    def assign_components(self):

        res_9 = ReactiveComponent(
            value=220,
            neg=0,
            pos=self.ground_node)

        res_10 = ReactiveComponent(
            value=220,
            neg=0,
            pos=1)

        res_11 = ReactiveComponent(
            value=220,
            neg=2,
            pos=4)

        res_12 = ReactiveComponent(
            value=1100,
            neg=7,
            pos=3)

        res_13 = ReactiveComponent(
            value=100,
            neg=self.ground_node,
            pos=7)

        res_14 = ReactiveComponent(
            value=1.2e3,
            pos=6,
            neg=2
        )

        res_load = ReactiveComponent(
            value=150e3,
            neg=self.ground_node,
            pos=8
        )

        self.set_resistors([res_9, res_10, res_11, res_12, res_13, res_14, res_load])

       
        cap_1 = ReactiveComponent(
            value=20e-6,
            pos=3,
            neg=7)

        cap_2 = ReactiveComponent(
            value=0.33e-6,
            pos=6,
            neg=8)
            
        self.set_capacitors([cap_1, cap_2])

        in_source = Port(pos=0, neg=self.ground_node)
        supply = Port(pos=5, neg=self.ground_node)

        self.set_input_sources([in_source, supply])

        ia1 = Port(pos=3, neg=2)
        ig1 = Port(pos=3, neg=1)
        ia2 = Port(pos=6, neg=5)
        ig2 = Port(pos=6, neg=4)

        self.set_nl_sources([ia1, ig1, ia2, ig2])

        out1 = Port(pos=8, neg=self.ground_node)

        self.set_output_ports([out1])

    def compute_i(self, vn: np.ndarray) -> np.ndarray:
        it1 = self.nl_instance_a.compute_i(vn[:2, :])
        it2 = self.nl_instance_b.compute_i(vn[2:, :])

        return np.concatenate((it1, it2), axis=0)

    def compute_ji(self, vn: np.ndarray) -> np.ndarray:
        dit1 = self.nl_instance_a.compute_ji(vn[:2, :])
        dit2 = self.nl_instance_b.compute_ji(vn[2:, :])

        ret = np.zeros((4, 4))

        ret[:2, :2] = dit1
        ret[2:, 2:] = dit2

        return ret
