from wdtlitl_where_does_this_line_intersect_this_line import (
     wdtlitl_where_does_this_line_intersect_this_line
)
import numpy as np


def get_polygon_points_for_brewcub_023094():
    """
    Mathieu needs the area above the walls to be white.
    We do this for video frame brewcub_023094,
    and will transport to other video frames via homography.
    """
    
    width = 1920
    height = 1080

    # pick two points on the top of the left brick wall:
    a = [8.0, 332.375]
    b = [902.625, 302.875]
    ab_line = [a, b]

    # pick two points on the top of the right brick wall:
    c = [1171.0, 298.625]
    d = [1918.125, 306.375]

    cd_line = [c, d]

    # find the "joint point" where one wall meets the other:
    joint_point = wdtlitl_where_does_this_line_intersect_this_line(
        ab_line,
        cd_line
    )

    # continue the line from b through a until it hits an x coordinate of -big_number:

    big_number = 4000

    # this will be a vertex of our giant polygon so we give it a name:
    very_far_left_and_quite_high = [-big_number,  -big_number]
    very_far_left_and_quite_low = [-big_number,   big_number]
    
    # This make the "locus of all points where x = -big_number" line:
    far_left_line = [
        very_far_left_and_quite_low,
        very_far_left_and_quite_high
    ]

    
    point_on_the_ab_line_that_is_far_left = (
        wdtlitl_where_does_this_line_intersect_this_line(
            ab_line,
            far_left_line
        )
    )
        
    very_far_right_and_quite_high = [big_number, -big_number]
    very_far_right_and_quite_low = [big_number, big_number]
    
    # This make the "locus of all points where x = big_number" line:
    far_right_line = [
        very_far_right_and_quite_low,
        very_far_right_and_quite_high
    ]

    point_on_the_cd_line_that_is_far_right = wdtlitl_where_does_this_line_intersect_this_line(
        far_right_line,
        cd_line
    )

    list_of_xys = np.array(
        [
            very_far_left_and_quite_high,
            point_on_the_ab_line_that_is_far_left,
            a,
            b,
            joint_point,
            c,
            d,
            point_on_the_cd_line_that_is_far_right,
            very_far_right_and_quite_high
        ],
        dtype=np.float64
    )
    return list_of_xys

   