import numpy as np

def get_average_color_from_samples(samples) -> list:
    assert len(samples) > 0, "Cannot average zero things"
    r_total = 0.0
    g_total = 0.0
    b_total = 0.0
    for sample in samples:
        r = sample["r"]
        g = sample["g"]
        b = sample["b"]
        r_total += r
        g_total += g
        b_total += b
    r_average = r_total / len(samples)
    g_average = g_total / len(samples)
    b_average = b_total / len(samples)
    
    return [
        int(np.round(r_average)),
        int(np.round(g_average)),
        int(np.round(b_average))
    ]

    