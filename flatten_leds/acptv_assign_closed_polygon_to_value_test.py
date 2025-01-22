from wdtlitl_where_does_this_line_intersect_this_line import (
     wdtlitl_where_does_this_line_intersect_this_line
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from acptv_assign_closed_polygon_to_value import (
     acptv_assign_closed_polygon_to_value
)
from prii import (
     prii
)
import numpy as np


def test_acptv_assign_closed_polygon_to_value_1():
    """
    grayscale test.
    """
    
    width = 1920
    height = 1080
    

    list_of_xys = np.array(
        [
            [100, 540],
            [1930, 540],
            [1930, -30],
            [-10, -30],
        ],
        dtype=np.int32
    )

    image = np.zeros(
        shape=(height, width),
        dtype=np.uint8
    )

    acptv_assign_closed_polygon_to_value(
        list_of_xys=list_of_xys,
        value=255,
        victim_image_hw_and_maybe_c_np=image,
    )

    prii(image)


def test_acptv_assign_closed_polygon_to_value_2():
    """
    rgb_hw test.
    """
    
    width = 1920
    height = 1080
    
    radius = 400
    list_of_xys = np.array(
        [
            [
                960 + radius * np.cos(t),
                540 + radius * np.sin(t)
            ]
            
            for t in np.linspace(0, 2 * np.pi, 5 + 1)[:-1]
        ],
        dtype=np.float32
    )

    image = np.zeros(
        shape=(height, width, 3),
        dtype=np.uint8
    )

    image = get_original_frame_from_clip_id_and_frame_index(
        clip_id="brewcub",
        frame_index=23094,
    )
    
    value = [255, 255, 0]  # yellow

    acptv_assign_closed_polygon_to_value(
        list_of_xys=list_of_xys,
        value=value,
        victim_image_hw_and_maybe_c_np=image
    )

    prii(image)


def test_acptv_assign_closed_polygon_to_value_3():
    """
    grayscale test.
    """
    
    width = 1920
    height = 1080
    a = [8.0, 332.375]
    b = [902.625, 302.875]

    ab_line = [a, b]

    c = [1171.0, 298.625]
    d = [1918.125, 306.375]

    cd_line = [c, d]

    joint_point = wdtlitl_where_does_this_line_intersect_this_line(
        ab_line,
        cd_line
    )

    very_far_left_and_quite_high = [-3000,  -10000]
    
    far_left_line = [
        [-3000, 10000],
        very_far_left_and_quite_high
    ]

    point_off_left = wdtlitl_where_does_this_line_intersect_this_line(
        ab_line,
        far_left_line
    )
        
    very_far_right_and_quite_high = [3000, -10000]
    
    far_right_line = [
        [3000,  10000],
        very_far_right_and_quite_high
    ]

    point_off_right = wdtlitl_where_does_this_line_intersect_this_line(
        far_right_line,
        cd_line
    )


    list_of_xys = np.array(
        [
            very_far_left_and_quite_high,
            point_off_left,
            a,
            b,
            joint_point,
            c,
            d,
            point_off_right,
            very_far_right_and_quite_high
        ],
        dtype=np.int32
    )

   
    image = get_original_frame_from_clip_id_and_frame_index(
        clip_id="brewcub",
        frame_index=23094,
    )
    
    value = [255, 255, 0]  # yellow

    acptv_assign_closed_polygon_to_value(
        list_of_xys=list_of_xys,
        value=value,
        victim_image_hw_and_maybe_c_np=image
    )

    prii(image)




if __name__ == "__main__":
    test_acptv_assign_closed_polygon_to_value_1()
    test_acptv_assign_closed_polygon_to_value_2()
    test_acptv_assign_closed_polygon_to_value_3()