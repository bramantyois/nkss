import numpy as np


class Port:
    def __init__(self, pos, neg):
        """
        Class for port devices as voltage supply and voltage input
        :param pos: positive potential node
        :param neg: negative potential node
        """
        self.pos = pos
        self.neg = neg


class ReactiveComponent:
    def __init__(self, value, pos, neg):
        """
        Class for reactive component such as capacitor, inductor and resistor
        :param value: value for the component
        :param pos: positive potential node
        :param neg:negative potential node
        """
        self.value = value
        self.pos = pos
        self.neg = neg


class NLInstance:
    def __init__(self, name):
        self.name = name

    def compute_i(self, vn):
        return np.zeros_like(vn)

    def compute_ji(self, vn):
        return np.zeros((vn.shape[0], vn.shape[0]))
