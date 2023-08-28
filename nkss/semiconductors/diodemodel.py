import numpy as np
from ..components import NLInstance


class DiodeModel(NLInstance):
    def __init__(self, iis=2.52e-9, vt=25e-3, name='diode'):
        """
        Diode clipper model with high pass filter
        :param iis: Is current
        :param vt: temperature voltage
        """
        super().__init__(name)

        self.iis = iis
        self.vt = vt

    def compute_i(self, vd):
        ret = 2. * self.iis * np.sinh(vd / self.vt)

        return ret.reshape((1, 1))

    def compute_ji(self, vd):
        ret = (2. * self.iis / self.vt) * np.cosh(vd / self.vt)
        return ret.reshape((1, 1))


class D1N914(DiodeModel):
    def __init__(self, iis=2.52e-9, vt=25e-3):
        super().__init__(iis=iis, vt=vt)


class D1N34(DiodeModel):
    def __init__(self, iis=1e-6, vt=25e-3):
        super().__init__(iis=iis, vt=vt)