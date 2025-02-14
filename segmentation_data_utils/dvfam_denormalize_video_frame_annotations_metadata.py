import copy
from get_clip_id_to_info import (
     get_clip_id_to_info
)


def dvfam_denormalize_video_frame_annotations_metadata(
    video_frame_annotations_metadata
):
    """
    Expand the clip_id in each annotation to include the information about that clip.
    """
    clip_id_to_info = get_clip_id_to_info()
    denormalized_video_frame_annotations_metadata = []
    
    for x in video_frame_annotations_metadata:
        clip_id = x["clip_id"]
        clip_info = clip_id_to_info[clip_id]
        y = copy.deepcopy(x)
        y["clip_id_info"] = clip_info
        denormalized_video_frame_annotations_metadata.append(y)
    
    return denormalized_video_frame_annotations_metadata
   