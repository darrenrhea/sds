from typing import List, Tuple
import numpy as np

def choose_by_percent(
    value_prob_pairs: List[Tuple[any, float]],
):
    probs = [prob for _, prob in value_prob_pairs]
    probs = np.array(probs)
    probs /= probs.sum()
    u = np.random.rand()
    cumsum = np.cumsum(probs)
    for i, cum_prob in enumerate(cumsum):
        if u <= cum_prob:
            return value_prob_pairs[i][0]
    return value_prob_pairs[-1][0]
