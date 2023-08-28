from src.nodalkstatespace.nodalkstatespace import NodalKStateSpace
from src.nodalkstatespace.reactivecomponent import ReactiveComponent
from src.nodalkstatespace.port import Port

from src.nodalkstatespace.triodemodel import TB12AX7, TB12AX7Improved

import matplotlib.pyplot as plt
import numpy as np


class HPSSTriode(NodalKStateSpace):
    def __init__(self, sample_rate=44100, v_supply=300, name='TriodeHPSS', use_improved_tube=False):
        super().__init__(num_of_nodes=5, sample_rate=sample_rate, v_supply=v_supply, name=name)

        self.improved_tube = use_improved_tube
        
        self.assign_components()
        self.calc_system()

        self.converter_matrix = np.array([
            [-1, 0],
            [0, -1]])
        

    def assign_components(self):
        if self.improved_tube:
            self.nl_instances = TB12AX7()
        else:
            self.nl_instances = TB12AX7Improved()

        res1 = ReactiveComponent(550e3, pos=1, neg=self.ground_node)
        res2 = ReactiveComponent(2.2e3, pos=4, neg=self.ground_node)
        res3 = ReactiveComponent(100e3, pos=2, neg=3)

        self.set_resistors([res1, res2, res3])

        cap1 = ReactiveComponent(2.2e-9, pos=1, neg=0)
        #cap2 = ReactiveComponent(100e-12, pos=2, neg=3)
        cap3 = ReactiveComponent(10e-6, pos=4, neg=self.ground_node)
        self.set_capacitors([cap1, cap3])

        in_port = Port(pos=0, neg=self.ground_node)
        supply_port = Port(pos=2, neg=self.ground_node)

        self.set_input_sources([in_port, supply_port])

        ia_port = Port(pos=4, neg=3)
        ig_port = Port(pos=4, neg=1)

        self.set_nl_sources([ia_port, ig_port])

        out_port = Port(pos=3, neg=self.ground_node)
        self.set_output_ports([out_port])

    # def compute_i(self, vn: np.ndarray) -> np.ndarray:
    #     #v_neg = self.neg_mat@vn
    #     return self.nl_instances.compute_i(vn)
    #
    # def compute_ji(self, vn: np.ndarray) -> np.ndarray:
    #     #v_neg = self.neg_mat@vn
    #     return self.nl_instances.compute_ji(vn)


if __name__ == '__main__':
    sr = 44100
    n_sec = 0.1

    freq = 800

    max_amp = 10

    triode = HPSSTriode()
    triode.calc_init_values(
        un=np.array([
            [0],
            [300]]),
        xnm1=np.array([
            [0.6e-9],
            #[7.6e-3],
            [1.6]]),
        vn=np.array([
            [-218.],
            [1.7]]))

    num_samples = int(sr * n_sec)

    amps = np.linspace(0, max_amp, num_samples)
    t = np.linspace(0, n_sec, num_samples)

    sine_sweep = amps*np.sin(freq * t)

    out = triode.process_block(input_samples=sine_sweep)

    plt.subplot(2, 1, 1)
    plt.plot(out)
    plt.title('out')

    plt.subplot(2, 1, 2)
    plt.plot(sine_sweep)
    plt.title('sin')

    plt.tight_layout()
    plt.show()

