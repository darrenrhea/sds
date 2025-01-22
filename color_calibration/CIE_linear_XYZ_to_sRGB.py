import numpy as np


def CIE_linear_XYZ_to_sRGB(xyz_values):
    """
    Matrix coefficients came from:
    https://www.color.org/sRGB.xalter

    https://github.com/markkness/ColorPy/blob/master/colorpy/colormodels.py
    and are different from the ones in the wikipedia article:

    Maybe they used simplified 
    http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    """
    M = np.array(
        [
            [3.2410, -1.5374, -0.4986],
            [-0.9692, 1.8760, 0.0416],
            [0.0556, -0.2040, 1.0570]
        ]
    ).T
    rgb = np.dot(xyz_values, M)
    # let us not take negative values to a power:
    # RuntimeWarning: invalid value encountered in power 1.055 * (rgb**(1/2.4)) - 0.055,
    rgb = rgb.clip(0, 1e12)

    # https://stackoverflow.com/questions/17536417/correct-conversion-from-rec-709-to-srgb
    # claims 0.0031308  is the right threshold to match 0.04045
    # again, everyone claims it doesn't matter at 8 bit literally at all,
    # and is negligible at 10 bit.

    threshold = 0.0031308  
    # threshold = 0.00304

    gamma_compressed = np.where(
        rgb > threshold,
        1.055 * (rgb**(1/2.4)) - 0.055,
        12.92 * rgb
    )

    # WHOA! the np.round makes it perfectly invertible!!??
    return np.round(gamma_compressed * 255).clip(0, 255)



