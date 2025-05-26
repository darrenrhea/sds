from make_a_channel_stack_u16_from_local_file_pathed_annotation import (
     make_a_channel_stack_u16_from_local_file_pathed_annotation
)
import numpy as np
from prii_hw_np_nonlinear_u16 import (
     prii_hw_np_nonlinear_u16
)
import pprint
from get_local_file_paths_for_annotations import (
     get_local_file_paths_for_annotations
)


def test_make_a_channel_stack_u16_from_local_file_pathed_annotation_1():
    """
    Try crazy orders, repeats of a channel twice, imputable channels like relevance, etc.
    """
    channel_names = ["r",  "floor_not_floor", "depth_map", "g", "b", "b", "relevance"]
    desired_labels = set(["depth_map", "floor_not_floor", "original"])

    video_frame_annotations_metadata_sha256 = "4bffcd3e6d1e6cdc0055fdf5004b498e1f07282ddeebe2d524a59d28726208d2"
        
    local_file_pathed_annotations = get_local_file_paths_for_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        max_num_annotations=1,
    )
    
    local_file_pathed_annotation = local_file_pathed_annotations[0]
    pprint.pprint(
        local_file_pathed_annotation
    )
    channel_stack = make_a_channel_stack_u16_from_local_file_pathed_annotation(
        local_file_pathed_annotation=local_file_pathed_annotation,
        channel_names=channel_names,
    )
   
    for a in local_file_pathed_annotations:
        pprint.pprint(local_file_pathed_annotations)
        assert a["local_file_paths"]["original"].is_file()
        assert a["local_file_paths"]["floor_not_floor"].is_file()
        assert a["local_file_paths"]["depth_map"].is_file()
           
    assert channel_stack.dtype == np.uint16
    assert channel_stack.ndim == 3
    assert channel_stack.shape[0] == 1080, f"{channel_stack.shape=}"
    assert channel_stack.shape[1] == 1920
    assert (
        channel_stack.shape[2] == len(channel_names)
    ), f"Given that {channel_names=}, aren't we expecting {len(channel_names)=} channels ?"
   
    for i, channel_name in enumerate(channel_names):
        print(f"{i}-ith channel is {channel_name}:")
        prii_hw_np_nonlinear_u16(channel_stack[:, :, i])
    

if __name__ == "__main__":
    test_make_a_channel_stack_u16_from_local_file_pathed_annotation_1()

