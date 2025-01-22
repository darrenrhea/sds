import random


def choose_name_from_name_to_prob(name_to_prob):
    """
    given a dictionary from key to probability of that key, returns a key at random.
    ```python
    for i in range(100):
    print(
        choose_name_from_name_to_prob(name_to_prob=dict(a=0.29, b=0.7, c=0.01)),
        end=""
    )
    ```
    """
    names = sorted([name for name in name_to_prob.keys()])
    probs = [name_to_prob[name] for name in names]
    assert sum(probs) <= 1.000001, f"so-called probabilities add up to way more than 1.0: {name_to_prob}"
    assert sum(probs) >= 0.999999
    name = random.choices(population=names, weights=probs, k=1)[0]
    return name
