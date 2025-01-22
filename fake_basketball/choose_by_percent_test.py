import pprint as pp
from choose_by_percent import (
     choose_by_percent
)
from collections import defaultdict


def choose_by_percent_test():
    value_prob_pairs = [
        ("a", 0.75),
        ("b", 0.20),
        ("c", 0.05),
    ]
    num_trials = 100000

    counts = defaultdict(int)
    for _ in range(num_trials):
        ans = choose_by_percent(
            value_prob_pairs=value_prob_pairs,
        )
        counts[ans] += 1
    

    pp.pprint(counts)

if __name__ == "__main__":
    choose_by_percent_test()