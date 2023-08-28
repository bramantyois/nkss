from ..components import NLInstance
import math
import numpy as np


class TransistorModel(NLInstance):
    def __init__(self, iis=5.42e-14, vt=25e-3, br=1.19e1, bf=3e2):
        """
        Transistor model for state space model, might not suitable for nodal analysis
        :param iis:
        :param vt:
        :param br:
        :param bf:
        """
        self.iis = iis
        self.vt = vt
        self.br = br
        self.bf = bf
        self.ar = self.br/(1.+self.br)
        self.af = self.bf/(1.+self.bf)

    def compute_i(self, vn):
        """
        compute transistor current, refers to yeh et al
        :param vn: array of vbe and vce
        :return: array of ib and ic
        """
        vn = vn.astype(np.longdouble)
        
        vbe = vn[[0]]
        vce = vn[[1]]
        vbc = vbe - vce

        ib = self.iis * ((math.exp(vbe/self.vt) - 1)/self.bf - (math.exp(vbc/self.vt)-1)/self.br)
        ic = self.iis * ((math.exp(vbe/self.vt) - 1) - (math.exp(vbc/self.vt)-1)/self.ar)

        return np.array([[ib], [ic]])

    def compute_ji(self, vn):
        """
        compute jacobian of transistor current

        :param vn: array of vbe and vce
        :return: array of J ib and J ic
        """
        vn = vn.astype(np.longdouble)

        vbe = vn[[0]]
        vce = vn[[1]]
        vbc = vbe - vce

        ib_be = self.iis * math.exp(vbe/self.vt)/(self.bf*self.vt)
        ib_ce = self.iis * math.exp(vbc/self.vt)/(self.br*self.vt)

        ic_be = self.iis * math.exp(vbe/self.vt)/self.vt
        ic_ce = self.iis * math.exp(vbc/self.vt)/(self.ar*self.vt) # (dic/dvbc) * (dvbc/dvce)

        return np.array([[ib_be, ib_ce], [ic_be, ic_ce]])


class TR5088(TransistorModel):
    def __init__(self):
        super().__init__(iis=20.3e-15, bf=1430, br=4.0)
