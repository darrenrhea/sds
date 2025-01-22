from get_indicator_of_largest_area_quadrilateral_makeable_from_these_points import (
     get_indicator_of_largest_area_quadrilateral_makeable_from_these_points
)
import numpy as np



def test_get_indicator_of_largest_area_quadrilateral_makeable_from_these_points_1():
    points = np.array(
        [
            [1883.618711680896, 300.99323662609044],
            [1919.1660557805724, 312.3450517419296],
            [1882.8185988383534, 310.4509640480561],
            [1877.4537784630097, 376.13905056783824],
            [1876.7201682525088, 385.452570685182],
            [1912.7654102988313, 387.67631065912303],
        ],
        dtype=np.float64
    )

    # points = np.array(
    #     [
    #         [0, 0],
    #         [0, 1],
    #         [1, 1],
    #         [0, 2],
    #         [1, 2],
    #     ],
    #     dtype=np.float64
    # )

    indicator = get_indicator_of_largest_area_quadrilateral_makeable_from_these_points(
        points=points
    )
    print(indicator)
    

if __name__ == "__main__":
    test_get_indicator_of_largest_area_quadrilateral_makeable_from_these_points_1()
    print("get_indicator_of_largest_area_quadrilateral_makeable_from_these_points passed")