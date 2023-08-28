from src.nodalkstatespace.nodalkstatespace import NodalKStateSpace
from src.nodalkstatespace.reactivecomponent import ReactiveComponent
from src.nodalkstatespace.port import Port

from src.nodalkstatespace.transistormodel import TR5088

import matplotlib.pyplot as plt
import numpy as np


class TransistorBooster(NodalKStateSpace):
    def __init__(self, sample_rate, v_supply=9, name='TransistorBooster'):
        super().__init__(num_of_nodes=5, sample_rate=sample_rate, v_supply=v_supply, name=name)

        self.assign_components()

        self.calc_system()

        self.converter_matrix = np.array([[-1, 0], [0, -1]])

    def assign_components(self):
        self.nl_instances = TR5088()

        res1 = ReactiveComponent(470e3, pos=2, neg=1)
        res2 = ReactiveComponent(65e3, pos=1, neg=self.ground_node)
        res3 = ReactiveComponent(10e3, pos=2, neg=3)
        res4 = ReactiveComponent(3.9e3, pos=4, neg=self.ground_node)
        self.set_resistors([res1, res2, res3, res4])

        cap1 = ReactiveComponent(100e-9, pos=1, neg=0)
        cap2 = ReactiveComponent(4.7e-6, pos=4, neg=0)
        self.set_capacitors([cap1, cap2])

        in_port = Port(pos=0, neg=self.ground_node)
        supply_port = Port(pos=2, neg=self.ground_node)

        self.set_input_sources([in_port, supply_port])

        ib_port = Port(pos=4, neg=1)
        ic_port = Port(pos=4, neg=3)

        self.set_nl_sources([ib_port, ic_port])

        out_port = Port(pos=4, neg=self.ground_node)
        self.set_output_ports([out_port])


if __name__ == '__main__':
    sr = 44100
    n_sec = 0.1

    max_freq = 1000
    min_freq = 20

    triode = TransistorBooster(sample_rate=sr)
    triode.calc_init_values(
        un=np.array([
            [0],
            [9]]),
        xnm1=np.array([
            [0.6e-9],
            [0.6e-9]]),
        vn=np.array([
            [-0.7],
            [-1.4]]))

    num_samples = int(sr * n_sec)
    freqs = np.linspace(2 * np.pi * min_freq, 2 * np.pi * max_freq, num_samples)
    t = np.linspace(0, n_sec, num_samples)

    sine_sweep = 1e-3*np.sin(freqs * t)

    out = triode.process_block(input_samples=sine_sweep)

    plt.subplot(2, 1, 1)
    plt.plot(out)
    plt.title('out')

    plt.subplot(2, 1, 2)
    plt.plot(sine_sweep)
    plt.title('sin')

    plt.tight_layout()
    plt.show()
