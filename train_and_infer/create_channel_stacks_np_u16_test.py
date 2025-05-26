from prii_hw_np_nonlinear_u16 import (
     prii_hw_np_nonlinear_u16
)
from create_padded_channel_stacks_np_u16 import (
     create_padded_channel_stacks_np_u16
)
import pprint
from get_local_file_paths_for_annotations import (
     get_local_file_paths_for_annotations
)
import numpy as np



def test_create_channel_stacks_np_u16_1():
    channel_names=["r", "g", "b", "floor_not_floor", "depth_map"]
    frame_height = 1088
    frame_width = 1920
    do_padding = True
    
    desired_labels = set(["depth_map", "floor_not_floor", "original"])

    video_frame_annotations_metadata_sha256 = "4bffcd3e6d1e6cdc0055fdf5004b498e1f07282ddeebe2d524a59d28726208d2"
    print(f"video_frame_annotations_metadata_sha256: {video_frame_annotations_metadata_sha256}")
    local_file_pathed_annotations = get_local_file_paths_for_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        max_num_annotations=3,
    )
   

    num_training_points = len(local_file_pathed_annotations)

    for a in local_file_pathed_annotations:
        pprint.pprint(local_file_pathed_annotations)
        assert a["local_file_paths"]["original"].is_file()
        assert a["local_file_paths"]["floor_not_floor"].is_file()
        assert a["local_file_paths"]["depth_map"].is_file()
        


    channel_stacks = create_padded_channel_stacks_np_u16(
        local_file_pathed_annotations=local_file_pathed_annotations,
        frame_height=frame_height,
        frame_width=frame_width,
        do_padding=do_padding,
        channel_names=["r", "g", "b", "floor_not_floor", "depth_map"],
    )

    assert len(channel_stacks) == num_training_points

    for channel_stack in channel_stacks:
        assert channel_stack.shape[0] == frame_height, f"{channel_stack.shape=} {frame_height=}"
        assert channel_stack.shape[1] == frame_width
        assert (
            channel_stack.shape[2] == len(channel_names)
        ), f"Given that {channel_names=}, aren't we expecting {len(channel_names)=} channels ?"
        assert channel_stack.dtype == np.uint16
        assert channel_stack.ndim == 3
        for i, channel_name in enumerate(channel_names):
            print(f"channel {channel_name}:")
            prii_hw_np_nonlinear_u16(channel_stack[:, :, i])
    

if __name__ == "__main__":
    test_create_channel_stacks_np_u16_1()