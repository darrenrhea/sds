import numpy as np
from PIL import Image
import numpy as np

def sRGB_to_CIE_linear_XYZ(rgb_values):
    """
    See the tests
    skimage.rgb2xyz
    or 
    https://www.color.org/sRGB.xalter


    """
    assert rgb_values.shape[1] == 3
    
    v = rgb_values / 255.0
    # according to https://github.com/w3c/wcag/issues/360
    # the threshold should be 0.04045, NOT 0.03928
    # BUT they also say that at 8 bit there is literally no difference
    # and at 10 bit a negligible difference.
    # http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    # backs this up.
    # https://en.wikipedia.org/wiki/SRGB

    # https://stackoverflow.com/questions/17536417/correct-conversion-from-rec-709-to-srgb

    threshold = 0.04045
    # threshold = 0.03928  # may be wrong despite wikipedia

    gamma_expanded = np.where(
        v > threshold,
        ((v + 0.055) / 1.055)**2.4,
        v / 12.92
    )

    # http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    # simple_gamma_companding = v**2.2  is a lot simpler but not as accurate

    v ** (1/2.2)
    M = np.array(  # matches https://en.wikipedia.org/wiki/SRGB
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    ).T
    xyz = np.dot(gamma_expanded, M)
    return xyz



# CHATGPT comes up with this instead,
# which is the same as the above, but with a different threshold 0.04045
# which is backed up by https://github.com/w3c/wcag/issues/360
# and more digits of precision in the matrix M.

def srgb_to_linear(srgb):
    a = 0.055
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + a) / (1 + a)) ** 2.4)

def linear_to_xyz(rgb):
    # Matrix to convert from linear RGB to XYZ, assuming sRGB color space
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    return np.dot(rgb, M.T)

def jpg_to_xyz(filename):
    # Load the image
    image = Image.open(filename)
    image = image.convert('RGB')
    
    # Normalize and convert to linear RGB
    srgb = np.array(image) / 255.0
    linear_rgb = srgb_to_linear(srgb)
    
    # Convert to XYZ
    xyz = linear_to_xyz(linear_rgb)
    
    return xyz

# Example usage
xyz_image = jpg_to_xyz('your_image.jpg')