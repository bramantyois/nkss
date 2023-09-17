import numpy as np
from ..components import NLInstance

# TODO: [NKSS-2] model is not stable. Need more investigation

class TransistorModel1(NLInstance):
    def __init__(self, iis=5.42e-14, vt=25e-3, br=1.19e1, bf=3e2):
        """
        Transistor model for state space model version 1. 
        Collector and emitter current are computed.
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
        
        vbc = vn[[0]]
        vbe = vn[[1]]

        ic = self.iis * ((np.exp(vbe/self.vt) - 1) - (np.exp(vbc/self.vt)-1)/self.ar)
        ie = self.iis * ((np.exp(vbe/self.vt) - 1)/self.af - (np.exp(vbc/self.vt)-1))

        return np.array([[ic], [ie]])

    def compute_ji(self, vn):
        """
        compute jacobian of transistor current

        :param vn: array of vbe and vce
        :return: array of J ib and J ic
        """
        vn = vn.astype(np.longdouble)
        
        vbc = vn[[0]]
        vbe = vn[[1]]

        ic_bc = -self.iis * np.exp(vbc/self.vt)/(self.ar*self.vt)
        ic_be = self.iis * np.exp(vbe/self.vt)/self.vt

        ie_bc = -self.iis * np.exp(vbc/self.vt)/self.vt
        ie_be = self.iis * np.exp(vbe/self.vt)/(self.af*self.vt)

        return np.array([[ic_bc, ic_be], [ie_bc, ie_be]])



class TransistorModel2(NLInstance):
    def __init__(self, iis=5.42e-14, vt=25e-3, br=1.19e1, bf=3e2):
        """
        Transistor model for state space model version 1. 
        Collector and base current are computed.
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
        vbe = vn[0][0]
        vce = vn[1][0]
        
        vbc = vbe - vce

        ib = self.iis * ((np.exp(vbe/self.vt) - 1)/self.bf - (np.exp(vbc/self.vt)-1)/self.br)
        #ic = self.iis * ((np.exp(vbe/self.vt) - 1) - (np.exp(vbc/self.vt)-1)/self.ar)
        ic = self.iis * ((np.exp(vbe/self.vt) - np.exp(vbc/self.vt)) - (np.exp(vbc/self.vt)-1)/self.br)

        ret = np.array([[ib], [ic]])
 
        return ret

    def compute_ji(self, vn):
        """
        compute jacobian of transistor current

        :param vn: array of vbe and vce
        :return: array of J ib and J ic
        """
        vbe = vn[0][0]
        vce = vn[1][0]
        vbc = vbe - vce

        ib_be = self.iis * np.exp(vbe/self.vt)/(self.bf*self.vt)
        ib_ce = self.iis * np.exp(vbc/self.vt)/(self.br*self.vt)

        ic_be = self.iis * np.exp(vbe/self.vt)/self.vt
        ic_ce = self.iis * (np.exp(vbc/self.vt)/self.vt + np.exp(vbc/self.vt)/(self.vt*self.br)) # (dic/dvbc) * (dvbc/dvce)

        ret = np.array([[ib_be, ib_ce], [ic_be, ic_ce]])
        
        return ret


class TR5088BaseCurrent(TransistorModel1):
    def __init__(self):
        super().__init__(iis=20.3e-15, bf=1430, br=4.0)
        

class TR5088EmitterCurrent(TransistorModel2):
    def __init__(self):
        super().__init__(iis=20.3e-15, bf=1430, br=4.0)