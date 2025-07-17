import better_json as bj
from infer_from_id import (
     infer_from_id
)
from pathlib import Path


from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)

def make_frame_ranges_file(clip_id, frame_ranges):
    frame_ranges_file_path = Path("frame_ranges/temp.json5").resolve()
    obj = {
        "original_suffix": "_original.png",
        "input_dir": f"/media/drhea/muchspace/clips/{clip_id}/frames",
        "clip_id": clip_id,
        "frame_ranges": frame_ranges
    }

    bj.dump(
        fp=frame_ranges_file_path,
        obj=obj
    )
    return frame_ranges_file_path

   

def test_infer_from_id_1():
    """
    This test became infer abitrary frames from a clip.
    """
    final_model_id = "yur"

    clip_id = "munich2024-01-09-1080i-yadif"
    
    fixups = bj.load(
        "~/r/frame_attributes/bay-zal-2024-03-15-mxf-yadif_led.json5"
    )
    
    frame_indices = []
    for thing in fixups:
        frame_index = thing
        frame_indices.append(
            frame_index
        )

    # frame_ranges can contain integers:
    frame_ranges = frame_indices

    frame_ranges_file_path = make_frame_ranges_file(
        clip_id=clip_id,
        frame_ranges=frame_ranges
    )

    shared_dir = get_the_large_capacity_shared_directory()
    output_dir = shared_dir / "inferences"

    infer_from_id(
        final_model_id=final_model_id,
        model_id_suffix=final_model_id,
        frame_ranges_file_path=frame_ranges_file_path,
        output_dir=output_dir
    )
    
    print("Tell sarah or whoeverto:")

    for frame_index in frame_indices:
        print(f"led {frame_index}")

   
if __name__ == "__main__":
    test_infer_from_id_1()