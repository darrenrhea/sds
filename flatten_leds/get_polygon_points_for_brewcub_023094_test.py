from get_polygon_points_for_brewcub_023094 import (
     get_polygon_points_for_brewcub_023094
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


def test_get_polygon_points_for_brewcub_023094_1():
    list_of_xys = get_polygon_points_for_brewcub_023094()

   
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
    test_get_polygon_points_for_brewcub_023094_1()