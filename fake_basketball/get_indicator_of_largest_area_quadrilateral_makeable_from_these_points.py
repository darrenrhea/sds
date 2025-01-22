import numpy as np

import shapely

def get_indicator_of_largest_area_quadrilateral_makeable_from_these_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    Given a set of points in 2D as a numpy array of shape (num_points, 2),
    this returns an indicator vector
    [i.e. a boolean numpy array of shape (num_points,)]
    which indicates which 4 points make the biggest area quadrilateral
    over all choices of 4 points.
    """
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 2

    num_points = points.shape[0]
    assert num_points >= 4, "Error: Cannot make a quadrilateral from less than 4 points!"

    biggest_area_so_far = - 1.0

    max_iterations = 1000
    cntr = 0
    while True:
        if cntr >= max_iterations:
            break
        a = np.random.randint(0, num_points)
        b = np.random.randint(0, num_points)
        c = np.random.randint(0, num_points)
        d = np.random.randint(0, num_points)
        if len(set([a, b, c, d])) < 4:
            continue

        quad = np.array([
            points[a],
            points[b],
            points[c],
            points[d]
        ])

        area = (
            np.abs(
                np.cross(
                    quad[1] - quad[0],
                    quad[2] - quad[0]
                )
            )
            +
            np.abs(
                np.cross(
                    quad[2] - quad[0],
                    quad[3] - quad[0]
                )
            )
        ) / 2.0

        area = shapely.MultiPoint(quad).convex_hull.area

        if area > biggest_area_so_far:
            biggest_area_so_far = area
            best_a = a
            best_b = b
            best_c = c
            best_d = d

        # print(f"quad: {quad}, area: {area}")
        cntr += 1

    indicator = np.zeros(
        shape=(
            num_points,
        ),
        dtype=bool,
    )
    
    print(f"{biggest_area_so_far=}")

    indices = [best_a, best_b, best_c, best_d]
    indicator[indices] = True
    return indicator


