from clip_id_to_frame_ranges_we_care_about_for_segmentation import (
     clip_id_to_frame_ranges_we_care_about_for_segmentation
)
from infer_arbitrary_frames_from_a_clip import (
     infer_arbitrary_frames_from_a_clip
)
from pathlib import Path

import better_json as bj


def test_infer_arbitrary_frames_from_a_clip_1():
    """
    Given a final_model_id, clip_id, and frame_ranges,
    which is a python list of single frames, or frame ranges [start, stop],
    or even frame ranges with a step [start, stop, step],
    infer the frames from the clip.
    """
    final_model_id = "w3k"

    clip_id_to_frame_ranges = bj.load(
        Path("~/r/clip_ids/clip_id_to_frame_ranges_we_care_about_for_segmentation.json5").expanduser()
    )
    
    clip_ids = sorted(list(clip_id_to_frame_ranges.keys()))
    
    for clip_id in clip_ids:
        frame_ranges = clip_id_to_frame_ranges_we_care_about_for_segmentation(
            clip_id=clip_id
        )

        print(f"for {clip_id=} doing frame_ranges: {frame_ranges}")
        
        infer_arbitrary_frames_from_a_clip(
            final_model_id=final_model_id,
            clip_id=clip_id,
            frame_ranges=frame_ranges
        )
  

def test_infer_arbitrary_frames_from_a_clip_2():
    infer_arbitrary_frames_from_a_clip(
        print_in_terminal=True,
        final_model_id="tw9", # fe1",
        clip_id="london20240208",
        original_suffix=".jpg",
        frame_ranges=[
           [0, 5000, 1000],
        ]
    )


if __name__ == "__main__":
    test_infer_arbitrary_frames_from_a_clip_2()
