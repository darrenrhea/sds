

import numpy as np





def make_random_rodrigues_vector():
    rando = np.random.randn(3)
    random_unit_vector = rando / np.linalg.norm(rando)
    random_angle = np.pi * np.random.rand()
    rodrigues = random_angle * random_unit_vector
    n = np.linalg.norm(rodrigues)
    assert n < np.pi
    return rodrigues