import numpy as np


def get_led_corners_for_dallas_mavericks():
    """
    This is very DAL specific.
    """
    origin = np.array(
        [
            0.18643665313720703,
            29.4282808303833,
            1.3350388407707214
        ],
        dtype=np.float64
    )
    x = np.array(
        [
            1.0,
            0.0,
            0.0
        ],
        dtype=np.float64
    )
    y = np.array(
        [
            0.0,
            0.16780186922657853,
            0.9858207406440921
        ],
        dtype=np.float64
    )  
    width = 50.34819984436035 / 2
    height = 1.8224297205803919 / 2

    bl = origin - x * width - y * height
    tl = origin - x * width + y * height
    br = origin + x * width - y * height
    tr = origin + x * width + y * height

    dct = {}
    dct["bl"] = list(bl)
    dct["tl"] = list(tl)
    dct["br"] = list(br)
    dct["tr"] = list(tr)
    return dct


if __name__ == "__main__":
    import pprint as pp
    pp.pprint(get_led_corners_for_dallas_mavericks())