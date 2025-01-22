from find_2d_convex_hull import (
     find_2d_convex_hull
)
"""
given a triangle in R^2, determine the lattice squares that
it intersects with non-trivial area, and how much area
"""
import numpy as np
from scipy.spatial import ConvexHull
# http://mathworld.wolfram.com/TriangleInterior.html



def find_2d_convex_hull_demo():
    """
    Given points in the plane R^2,
    compute the area of the convex hull
    """
    points = np.random.rand(30, 2)   # random points in 2-D

  
    convex_hull_vertices = find_2d_convex_hull(points)
    
    # print(convex_hull_vertices)

    center = np.average(convex_hull_vertices, axis=0)
    

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots()
    plt.plot(points[:,0], points[:,1], 'o')
    xs = list(convex_hull_vertices[:, 0]) + [convex_hull_vertices[0, 0]]
    ys = list(convex_hull_vertices[:, 1]) + [convex_hull_vertices[0, 1]]
    plt.plot(xs, ys, 'r--', lw=2)
    plt.plot([center[0]], [center[1]], 'go')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    axes.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
   
    find_2d_convex_hull_demo()