import numpy as np

from .solver import newton_raphson
from tqdm import tqdm


class NKSS:
    def __init__(self, num_of_nodes, sample_rate, v_supply, name):
        """
        Nodal-K State Space. This class is used to compute the state space representation of a circuit.
        
        :param num_of_nodes: number of nodes in the circuit
        :param sample_rate: desired sample rate of the simulation
        :param v_supply: supply voltage
        :param name: name of the circuit
        """
        self.num_of_nodes = num_of_nodes
        self.sample_rate = int(sample_rate)
        self.ground_node = num_of_nodes
        self.v_supply = v_supply
        self.name = name

        self.NR = None
        self.NX = None
        self.NU = None
        self.NN = None
        self.NO = None
        self.GR = None
        self.GX = None
        self.Z = None

        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.E = None
        self.F = None
        self.G = None
        self.H = None
        self.K = None
        self.S = None

        self.init_vn = None
        self.init_vc = None
        self.init_xnm1 = None

        self.converter_matrix = None

        self.pn_max = None
        self.pn_min = None

        self.num_of_capacitors = None
        self.num_of_resistors = None
        self.num_of_nonlinear = None
        self.num_of_input_sources = None
        self.num_of_output_ports = None

        self.nl_instances = None

        self.ERROR = 1e-9

    def set_resistors(self, resistors: list):
        """

        :param resistors:
        :return:
        """
        self.num_of_resistors = len(resistors)

        self.NR = np.zeros((self.num_of_resistors, self.num_of_nodes))
        self.GR = np.zeros((self.num_of_resistors, self.num_of_resistors))

        for i in range(self.num_of_resistors):
            neg = resistors[i].neg
            pos = resistors[i].pos

            if neg < self.num_of_nodes:
                self.NR[i, neg] = -1
            if pos < self.num_of_nodes:
                self.NR[i, pos] = 1

            self.GR[i, i] = 1. / resistors[i].value

    def set_capacitors(self, capacitors: list):
        self.num_of_capacitors = len(capacitors)

        self.NX = np.zeros((self.num_of_capacitors, self.num_of_nodes))
        self.GX = np.zeros((self.num_of_capacitors, self.num_of_capacitors))
        self.Z = np.zeros((self.num_of_capacitors, self.num_of_capacitors))

        for i in range(self.num_of_capacitors):
            neg = capacitors[i].neg
            pos = capacitors[i].pos

            if neg < self.num_of_nodes:
                self.NX[i, neg] = -1
            if pos < self.num_of_nodes:
                self.NX[i, pos] = 1

            self.GX[i, i] = 2. * capacitors[i].value * self.sample_rate
            self.Z[i, i] = 1

    def set_input_sources(self, sources: list):
        self.num_of_input_sources = len(sources)

        self.NU = np.zeros((self.num_of_input_sources, self.num_of_nodes))

        for i in range(self.num_of_input_sources):
            neg = sources[i].neg
            pos = sources[i].pos

            if neg < self.num_of_nodes:
                self.NU[i, neg] = -1
            if pos < self.num_of_nodes:
                self.NU[i, pos] = 1

    def set_output_ports(self, outputs: list):
        self.num_of_output_ports = len(outputs)

        self.NO = np.zeros((self.num_of_output_ports, self.num_of_nodes))

        for i in range(self.num_of_output_ports):
            pos = outputs[i].pos

            self.NO[i, pos] = 1

    def set_nl_sources(self, nl_sources: list):
        self.num_of_nonlinear = len(nl_sources)
        self.NN = np.zeros((self.num_of_nonlinear, self.num_of_nodes))

        for i in range(self.num_of_nonlinear):
            neg = nl_sources[i].neg
            pos = nl_sources[i].pos

            if neg == pos and neg < self.num_of_nodes:
                self.NN[i, neg] = 0
            else:
                if neg < self.num_of_nodes:
                    self.NN[i, neg] = -1
                if pos < self.num_of_nodes:
                    self.NN[i, pos] = 1

    def compute_s(self):
        nrt = np.transpose(self.NR)
        nxt = np.transpose(self.NX)
        nut = np.transpose(self.NU)

        aa = np.dot(nrt, np.dot(self.GR, self.NR))
        aa += np.dot(nxt, np.dot(self.GX, self.NX))

        a = np.concatenate((aa, nut), axis=1)

        bb = np.zeros((self.NU.shape[0], self.NU.shape[0]))
        b = np.concatenate((self.NU, bb), axis=1)

        s = np.concatenate((a, b), axis=0)
        return s

    def calc_system(self) -> None:
        s = self.compute_s()
        si = np.linalg.inv(s)

        oo = np.zeros((self.NX.shape[0], si.shape[0] - self.NX.shape[1]))
        nxo = np.concatenate((self.NX, oo), axis=1)
        nxot = np.transpose(nxo)

        i = np.eye(self.NU.shape[0])
        oo = np.zeros((i.shape[0], si.shape[1] - i.shape[0]))
        oi = np.concatenate((oo, i), axis=1)
        oit = np.transpose(oi)

        oo = np.zeros((self.NN.shape[0], si.shape[0] - self.NN.shape[1]))
        nno = np.concatenate((self.NN, oo), axis=1)
        nnosi = np.dot(nno, si)
        nnot = np.transpose(nno)

        oo = np.zeros((self.NO.shape[0], si.shape[0] - self.NO.shape[1]))
        noo = np.concatenate((self.NO, oo), axis=1)
        noosi = np.dot(noo, si)

        zgnx = np.dot(nxo, si)
        zgnx = np.dot(self.GX, zgnx)
        zgnx = np.dot(2 * self.Z, zgnx)

        self.A = np.dot(zgnx, nxot) - self.Z
        self.B = np.dot(zgnx, oit)
        self.C = np.dot(zgnx, nnot)

        self.D = np.dot(noosi, nxot)
        self.E = np.dot(noosi, oit)
        self.F = np.dot(noosi, nnot)

        self.G = np.dot(nnosi, nxot)
        self.H = np.dot(nnosi, oit)
        self.K = np.dot(nnosi, nnot)

    def compute_xn(self, xnm1: np.ndarray, un: np.ndarray, iin: np.ndarray) -> np.ndarray:
        """

        :param xnm1:
        :param un:
        :param iin:
        :return:
        """
        xn = np.dot(self.A, xnm1) + np.dot(self.B, un) + np.dot(self.C, iin)
        return xn

    def compute_yn(self, xnm1: np.ndarray, un: np.ndarray, iin: np.ndarray) -> np.ndarray:
        """

        :param xnm1:
        :param un:
        :param iin:
        :return:
        """
        yn = np.dot(self.D, xnm1) + np.dot(self.E, un) + np.dot(self.F, iin)
        return yn

    def compute_vn(self, xnm1: np.ndarray, un: np.ndarray, iin: np.ndarray) -> np.ndarray:
        """

        :param xnm1:
        :param un:
        :param iin:
        :return:
        """
        vn = np.dot(self.G, xnm1) + np.dot(self.H, un) + np.dot(self.K, iin)
        return vn

    def compute_pn(self, xnm1: np.ndarray, un: np.ndarray) -> np.ndarray:
        """

        :param xnm1:
        :param un:
        :return:
        """
        pn = np.dot(self.G, xnm1) + np.dot(self.H, un)
        return pn

    def compute_in(self, vn: np.ndarray, pn: np.ndarray) -> np.ndarray:
        """

        :param vn:
        :param pn:
        :return:
        """
        temp_vnl = self.converter_matrix@vn
        temp_vnl = newton_raphson(self.compute_f, self.compute_jf, init_states=temp_vnl, inputs=pn)

        return self.compute_i(temp_vnl)

    def compute_in_vn(self, vn: np.ndarray, pn: np.ndarray) -> tuple:
        """

        :param vn:
        :param pn:
        :return:
        """

        temp_vnl = self.converter_matrix@vn
        temp_vnl = newton_raphson(self.compute_f, self.compute_jf, init_states=temp_vnl, inputs=pn)

        iin = self.compute_i(temp_vnl)
        return iin, temp_vnl

    def process_block(self, input_samples: np.ndarray) -> np.ndarray:
        """

        :param input_samples:
        :return:
        """
        num_of_samples = input_samples.shape[0]
        output = np.zeros((num_of_samples, self.num_of_output_ports))

        vn = self.init_vn.copy()
        xnm1 = self.init_xnm1.copy()

        if self.num_of_input_sources == 1:
            uns = input_samples.reshape(1, -1)
        else:
            uns = np.concatenate((input_samples.reshape(1, -1), np.ones((1, num_of_samples)) * self.v_supply))

        for i in tqdm(range(num_of_samples)):
            un = uns[:, [i]]

            pn = self.compute_pn(xnm1=xnm1, un=un)
            iin = self.compute_in(vn=vn, pn=pn)
            output[i, :] = self.compute_yn(xnm1=xnm1, un=un, iin=iin)
            vn = self.compute_vn(xnm1=xnm1, un=un, iin=iin)
            xnm1 = self.compute_xn(xnm1=xnm1, un=un, iin=iin)

        return output

    def compute_f(self, vn: np.ndarray, pn: np.ndarray) -> np.ndarray:
        inl = self.compute_i(vn)
        f = (pn + self.K @ inl) - self.converter_matrix@vn
        return f

    def compute_jf(self, vn: np.ndarray, pn: np.ndarray) -> np.ndarray:
        d_inl = self.compute_ji(vn)
        jf = (self.K @ d_inl) - self.converter_matrix@np.eye(vn.shape[0])
        return jf

    def compute_i(self, vn: np.ndarray) -> np.ndarray:
        return self.nl_instances.compute_i(vn)

    def compute_ji(self, vn: np.ndarray) -> np.ndarray:
        return self.nl_instances.compute_ji(vn)

    def calc_init_values(self, un: np.ndarray, xnm1: np.ndarray, vn: np.ndarray, error=1e-6):
        out_error = 1
        past_output = np.ones((self.num_of_output_ports, 1))
        past_state = np.ones_like(xnm1)
        while out_error > error:
            pn = self.compute_pn(xnm1=xnm1, un=un)
            iin = self.compute_in(vn=vn, pn=pn)
            output = self.compute_yn(xnm1=xnm1, un=un, iin=iin)
            vn = self.compute_vn(xnm1=xnm1, un=un, iin=iin)
            xnm1 = self.compute_xn(xnm1=xnm1, un=un, iin=iin)

            out_error_state = float(np.max(np.abs(past_state-xnm1)))
            out_error_nl = float(np.max(np.abs(output - past_output)))

            out_error = np.max([out_error_nl, out_error_state])
            past_output = output
            past_state = xnm1
        self.init_vn = vn
        self.init_xnm1 = xnm1

    def assign_components(self):
        pass
