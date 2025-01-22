

from scipy.spatial.transform import Rotation as R


def builtin_rotate_this_vector_by_this_rodrigues(vec, rodrigues):
    """
    Amazingly, even slower.
    Returns the result of rotating vector vec by the rotation defined by the angle-axis vector rodrigues.
    The Rodrigues vector's length is the angle (in radians) to rotate in right-hand rule about the axis.
    """

    r = R.from_rotvec(rodrigues)
    return r.apply(vec)

