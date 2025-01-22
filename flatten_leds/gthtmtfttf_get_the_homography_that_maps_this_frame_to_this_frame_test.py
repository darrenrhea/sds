from gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame import (
     gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame
)


def test_gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame_1():

    gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame(
        clip_id="brewcub",
        src_frame_index=23094,  # we manually made the polygon to cover the crowd area
        dst_frame_index=41500,
    )

if __name__ == "__main__":
    test_gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame_1()