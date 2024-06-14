import numpy as np


class UniformNoise(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        self.seed = (13917 * self.seed + 17) & ((1<<16) - 1)
        return self.seed / ((1 << 16) - 1)


class GaussianNoise(object):
    def __init__(self, seed):
        self.uniform_noise = UniformNoise(seed)

    def __call__(self):
        u1 = self.uniform_noise()
        u2 = self.uniform_noise()
        n = np.sqrt(-2*np.log(u1+1e-20)) * np.cos(2*np.pi*u2)
        return n