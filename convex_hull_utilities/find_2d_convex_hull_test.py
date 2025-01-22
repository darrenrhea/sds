import numpy as np
from find_2d_convex_hull import (
     find_2d_convex_hull
)
"""

~/r/computer_vision/convex_hull_finder.py

~/r/triangle_intersecting_lattice/triangle_intersecting_lattice.py 

~/r/ad_insertion/triangle_lattice.py
"""




from shapely.geometry import MultiPoint

def test_find_2d_convex_hull_1():
    """
    Given points in the plane R^2 as a n x 2 numpy array,
    find the points of the convex hull.
    """
    xys = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0.5, 0.5],
            [1, 1],
        ],
        dytpe=np.float64
    )

    vertices = find_2d_convex_hull(xys=xys)

    print(vertices)

   
if __name__ == "__main__":
    test_find_2d_convex_hull_1()