import numpy as np
import scipy.stats

NAME = "Particle"
COLUMNS = ["pt", "eta", "phi", "e"]


def generate(ngen, seed):
    """
    Generate PtEtaPhiE vectors.
    Based on https://github.com/root-project/root/blob/654a2c8eadac3409a65dc017279666e1cd5ccb6d/math/mathcore/test/stress/VectorTest.h#L43
    """
    np.random.seed(seed)
    for i in range(ngen):
        pt = np.random.exponential(scale=10.0)
        eta = np.random.uniform(-5.0, 5.0)
        phi = np.random.rand() * 3.1415926535897931
        if i % 50 == 0:
            m = np.random.uniform(0.0, 10.0)
        else:
            m = scipy.stats.rel_breitwigner.rvs(1.0, 0.01)
        e = np.sqrt((m**2) + (pt * np.cosh(eta)) ** 2)
        yield [pt, eta, phi, e]
