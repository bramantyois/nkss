from ..components import NLInstance
import numpy as np


def sign(val):
    if val > 0:
        return 1
    elif val == 0:
        return 0
    else:
        return -1


class TriodeModel(NLInstance):
    def __init__(self, mu=106, ex=1.46, kg=1572, kp=464, kvb=179.0, rg=2000, vt=2.5e-3):
        """
        Triode Model for state-space simulation
        :param mu:
        :param ex:
        :param kg:
        :param kp:
        :param kvb:
        :param rg:
        :param vt:
        """
        self.mu = mu
        self.ex = ex  # equal with KX
        self.kg = kg
        self.kp = kp
        self.kvb = kvb
        self.rg = rg
        self.vt = vt

    def compute_e1(self, vn):
        vak = vn[0]
        vgk = vn[1]

        e1 = (vak / self.kp) * np.log(1 + np.exp(self.kp * (1. / self.mu + vgk / np.sqrt(self.kvb + vak * vak))))
        return e1

    def compute_de1(self, vn, e1):
        res = np.zeros(2)
        if e1 >= 0:
            vak = vn[0]
            vgk = vn[1]

            temp_exp = np.exp(self.kp * (vgk / np.sqrt(self.kvb + vak * vak) + 1. / self.mu))

            res[0] = np.log(temp_exp + 1) / self.kp - vgk * vak * vak * temp_exp / (
                        np.power(self.kvb + vak * vak, 1.5) * (temp_exp + 1.))
            res[1] = vak * temp_exp / (np.sqrt(vak * vak + self.kvb) * (temp_exp + 1))
        return res

    def compute_ia(self, vn):
        e1 = self.compute_e1(vn)
        if e1 >= 0:
            ia = np.power(e1, self.ex) * (sign(e1)+1) / self.kg
        else:
            ia = 0
        return ia

    def compute_dia(self, vn, e1=None):
        res = np.zeros(2)
        if e1 is None:
            e1 = self.compute_e1(vn)
        if e1 >= 0:
            de1 = self.compute_de1(vn=vn, e1=e1)
            temp_val = (1 + sign(e1)) * self.ex * np.power(e1, self.ex - 1) / self.kg
            res[0] = temp_val * de1[0]
            res[1] = temp_val * de1[1]

        return res

    def compute_ig(self, vn):
        vgk = vn[1]
        ig = np.log(1 + np.exp(vgk / self.vt)) * self.vt / self.rg
        return ig

    def compute_dig(self, vn):
        res = np.zeros(2)
        vgk = vn[1]
        res[1] = np.exp(vgk / self.vt) / (self.rg * np.exp(vgk / self.vt) + self.rg)
        # res[1] = 1./(self.rg * np.exp(-vgk/self.vt) + self.rg)
        return res

    def compute_i(self, vn):
        vn = vn.astype(np.longdouble)
        res = np.zeros((2, 1))
        res[0, 0] = self.compute_ia(vn)  # ia
        res[1, 0] = self.compute_ig(vn)  # ig

        return res

    def compute_ji(self, vn):
        vn = vn.astype(np.longdouble)
        dia = self.compute_dia(vn).reshape((1, 2))
        dig = self.compute_dig(vn).reshape((1, 2))
        res = np.concatenate((dia, dig), axis=0)

        return res


class TriodeModelImproved(TriodeModel):
    def __init__(self, mu=106, ex=1.46, kg=1572, kp=464, kvb=179.0, rg=2000, vt=2.5e-3, kn=0.5, v_gamma=0.35, v_ct=0.49):
        super().__init__(mu=mu, ex=ex, kg=kg, kp=kp, kvb=kvb, rg=rg, vt=vt)
        self.kn = kn
        self.v_gamma = v_gamma
        self.v_ct = v_ct

    def compute_e1(self, vn):
        vak = vn[0]
        vgk = vn[1]

        e1 = (vak / self.kp) * np.log(1 + np.exp(self.kp * (1 / self.mu + (vgk + self.v_ct) / np.sqrt(self.kvb + vak*vak))))

        return e1

    def compute_de1(self, vn, e1):
        res = np.zeros(2)
        if e1 >= 0:
            vak = vn[0]
            vgk_ct = vn[1] + self.v_ct

            temp_exp = np.exp(self.kp * (vgk_ct / np.sqrt(self.kvb + vak * vak) + 1. / self.mu))

            res[0] = np.log(temp_exp + 1) / self.kp - vgk_ct * vak * vak * temp_exp / (
                        np.power(self.kvb + vak * vak, 1.5) * (temp_exp + 1.))
            res[1] = vak * temp_exp / (np.sqrt(vak * vak + self.kvb) * (temp_exp + 1))
        return res

    def compute_ig(self, vn):
        vgk = vn[1]

        if vgk < (self.v_gamma - self.kn):
            result = 0
        elif vgk > (self.v_gamma + self.kn):
            result = (vgk - self.v_gamma) / self.rg
        else:
            a = 1 / (4 * self.kn * self.rg)
            b = (self.kn - self.v_gamma) / (2 * self.kn * self.rg)
            c = (-1.0 * a * np.power(self.v_gamma - self.kn, 2.0)) - (b * (self.v_gamma - self.kn))
            result = a * np.power(vgk, 2.0) + b * vgk + c

        return result

    def compute_dig(self, vn):
        vgk = vn[1]

        result = np.zeros(2)

        if vgk < (self.v_gamma - self.kn):
            pass
        elif vgk > (self.v_gamma + self.kn):
            result[0] = 0
            result[1] = 1 / self.rg

        else:
            a = 1 / (4 * self.kn * self.rg)
            b = (self.kn - self.v_gamma) / (2 * self.kn * self.rg)

            result[0] = 0
            result[1] = a * 0.5 * vgk + b

        return result


class TB12AX7(TriodeModel):
    def __init__(self):
        super().__init__(mu=100, ex=1.4, kg=1060, kp=600, kvb=300, rg=2000)


class TB12AX7Improved(TriodeModelImproved):
    def __init__(self):
        super().__init__(mu=100, ex=1.4, kg=1060, kp=600, kvb=300, rg=2000)

