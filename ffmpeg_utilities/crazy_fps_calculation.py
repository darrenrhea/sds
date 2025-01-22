import numpy as np


def crazy_fps_calculation(n, fps):
    """
    The program bc has an option 
    echo "scale=10; ($n-0.25)/60000*1001" | bc
    export n=1; echo "scale=10; ($n-0.25)/60000*1001" | bc
    """
    return (n - 0.25) / fps
    return np.round(np.round((n-0.25) / 60000, 10) * 1001, 10)


def test_crazy_fps_calculation():
    input_output_pairs = [
        (1, .0125125000),
        (60, .9968291333),
        (57, .9467791333)
    ]

    for x, y in input_output_pairs:
        assert crazy_fps_calculation(x) == y, f"{crazy_fps_calculation(x)=} but should be {y}"
    print("Tests passed")
