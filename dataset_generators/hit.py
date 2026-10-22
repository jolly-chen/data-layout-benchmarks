import numpy as np
import scipy.stats

NAME = "Hit"
COLUMNS = ["x", "y", "z", "sigma"]


def generate(ngen, seed):
    """
    Generate PtEtaPhiE vectors.
    Based on https://github.com/root-project/root/blob/654a2c8eadac3409a65dc017279666e1cd5ccb6d/math/mathcore/test/stress/VectorTest.h#L43
    """
    np.random.seed(seed)
    for i in range(ngen):
        x = (float)(np.random.randint(0, 1000)) / 10.0
        y = (float)(np.random.randint(0, 1000)) / 10.0
        z = (float)(np.random.randint(0, 1000)) / 10.0
        sigma = 0.01 + (np.random.randint(0, 100)) / 1000.0
        yield [x, y, z, sigma]
