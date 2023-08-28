import sys 
sys.path.insert(0, '../../')

from src.nodalkstatespace.nodalkstatespace import NodalKStateSpace
from src.nodalkstatespace.reactivecomponent import ReactiveComponent
from src.nodalkstatespace.port import Port

from src.nodalkstatespace.triodemodel import TB12AX7

import matplotlib.pyplot as plt
import numpy as np


class SRPPPultecPower(NodalKStateSpace):
    def __init__(self, sample_rate=44100, v_supply=300, name='PultecPowerSRPP'):
        super().__init__(num_of_nodes=6, sample_rate=sample_rate, v_supply=v_supply, name=name)

        self.assign_components()
        self.calc_system()

        self.converter_matrix = -np.eye(4)

        self.nl_instance_a = TB12AX7()
        self.nl_instance_b = TB12AX7()

    def assign_components(self):
        res1 = ReactiveComponent(
            value=470e3,
            neg=self.ground_node,
            pos=1)

        res2 = ReactiveComponent(
            value=470,
            neg=self.ground_node,
            pos=2)

        res3 = ReactiveComponent(
            value=1e3,
            neg=3,
            pos=4)

        res4 = ReactiveComponent(
            value=1e9,
            neg=self.ground_node,
            pos=4)

        self.set_resistors([res1, res2, res3, res4])

        cap1 = ReactiveComponent(
            value=220e-9,
            neg=0,
            pos=1)

        cap2 = ReactiveComponent(
            value=470e-12 + 220e-9,
            neg=self.ground_node,
            pos=2)

        self.set_capacitors([cap1, cap2])

        in_source = Port(pos=0, neg=self.ground_node)
        supply = Port(pos=5, neg=self.ground_node)

        self.set_input_sources([in_source, supply])

        ia1 = Port(pos=2, neg=3)
        ig1 = Port(pos=2, neg=1)
        ia2 = Port(pos=4, neg=5)
        ig2 = Port(pos=4, neg=3)

        self.set_nl_sources([ia1, ig1, ia2, ig2])

        out1 = Port(pos=4, neg=self.ground_node)

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


if __name__ == '__main__':
    sr = 44100
    n_sec = 0.4

    max_freq = 1000
    min_freq = 0

    triode = SRPPPultecPower()
    triode.calc_init_values(
        un=np.array([
            [0],
            [250]]),
        xnm1=np.array([
            [0.5e-3],
            [0.3]]),
        vn=np.array([
            [-228],
            [21],
            [0.1],
            [0.1]]))

    print(triode.init_vn)

    num_samples = int(sr * n_sec)
    freqs = np.linspace(2 * np.pi * min_freq, 2 * np.pi * max_freq, num_samples)
    t = np.linspace(0, n_sec, num_samples)

    sine_sweep = 10*np.sin(freqs * t)

    out = triode.process_block(input_samples=sine_sweep)

    plt.subplot(2, 1, 1)
    plt.plot(out)
    plt.title('out')

    plt.subplot(2, 1, 2)
    plt.plot(sine_sweep)
    plt.title('sin')

    plt.tight_layout()
    plt.show()
