import numpy as np
from shapely import get_coordinates
from shapely.geometry import MultiPoint

def find_2d_convex_hull(xys):
    """
    Given points in the plane R^2 as a n x 2 numpy array,
    find the points of the convex hull.
    """
    assert isinstance(xys, np.ndarray)
    assert xys.shape[1] == 2
    
    multi_point = MultiPoint(xys)
    # print(get_coordinates(multi_point))
    convex_hull_shapely = multi_point.convex_hull

    ans = get_coordinates(convex_hull_shapely)
    assert isinstance(ans, np.ndarray)
    assert ans.shape[1] == 2

    # we need to find the indices of the points in the convex hull:
    indicator = np.zeros(xys.shape[0], dtype=bool)
    indices = []
    for i in range(ans.shape[0]):
        for j in range(xys.shape[0]):
            if np.allclose(ans[i], xys[j]):
                indices.append(j)
                break
    indicator[indices] = True
    
    return ans, indicator