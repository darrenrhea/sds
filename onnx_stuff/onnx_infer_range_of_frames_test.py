from instruct_how_to_download_file_to_laptop import (
     instruct_how_to_download_file_to_laptop
)
from make_evaluation_video import (
     make_evaluation_video
)
from load_json_file import (
     load_json_file
)
from color_print_json import (
     color_print_json
)
# pip install onnxruntime-gpu=1.22

# notice that this installs cudnn-9.1.1:
# conda install pytorch-cuda=12.4 pytorch=2.5.1

from print_green import (
     print_green
)
from onnx_infer_range_of_frames import (
     onnx_infer_range_of_frames
)
from get_mother_dir_of_frames_dir_from_clip_id import (
     get_mother_dir_of_frames_dir_from_clip_id
)
from pathlib import Path
import numpy as np


def logistic_sigmoid(x):
  return 1 / (1 + np.exp(-x))


def test_onnx_infer_range_of_frames_1():
    """
    aws s3 ls s3://infrastructurestack-pipelineassetsbed57a84-mlyzk550yglk/main/segmentation_models/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000036.onnx /shared/onnx/
    
    aws s3 cp s3://infrastructurestack-pipelineassetsbed57a84-mlyzk550yglk/main/segmentation_models/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000106.onnx /shared/onnx/
    """
    should_i_make_evaluation_video = True
    should_infer = True
    # should_i_make_evaluation_video = False
    # should_infer = False

    # onnx_file_path = Path(
    #    "/shared/onnx/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000036.onnx"
    # )
    
    # final_model_id = "nba2024finalgame5epoch36"


    onnx_file_path = Path(
       "/shared/onnx/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000106.onnx"
    )

    final_model_id = "nba2024finalgame5epoch106"

    obj = load_json_file(
        Path("~/r/frame_attributes/summer_league_evaluationvideos.json5").expanduser()
    )
    color_print_json(obj)
    out_video_file_paths = []
    for k in range(len(obj)):
        clip_id, first_frame_index, last_frame_index = obj[k]

        frames_dir = get_mother_dir_of_frames_dir_from_clip_id(clip_id) / "clips" / clip_id / "frames"
        original_suffix = "_original.jpg"
        if should_infer:
            onnx_infer_range_of_frames(
                onnx_file_path=onnx_file_path,
                final_model_id=final_model_id,
                clip_id=clip_id,
                frames_dir=frames_dir,
                original_suffix=original_suffix,
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                step=1,
                show_in_iterm2=False,
                is_logistic_sigmoid_baked_in=True
           )
        print_green(
            f"Inferred under {onnx_file_path} with clip_id={clip_id}, first_frame_index={first_frame_index}, last_frame_index={last_frame_index}"
        )
    
    
    
        shared_dir = Path("/shared")
        what_is_normal_color = "foreground"
        fill_color = "green"
        out_video_file_path = shared_dir / "show_n_tell" / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{final_model_id}_{what_is_normal_color}_fill{fill_color}.mp4"

        fps = 59.94
        original_suffix = "_original.jpg"
        mother_dir_of_frames_dir = get_mother_dir_of_frames_dir_from_clip_id(
            clip_id=clip_id
        )
        frames_dir = mother_dir_of_frames_dir / "clips" / clip_id / "frames"
        masks_dir = shared_dir / "inferences"
        draw_frame_numbers = True
        if should_i_make_evaluation_video:
            make_evaluation_video(
                original_suffix=original_suffix,
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                clip_id=clip_id,
                model_id=final_model_id,
                fps=fps,
                what_is_normal_color=what_is_normal_color,
                fill_color=fill_color,
                out_video_file_path=out_video_file_path,
                draw_frame_numbers=draw_frame_numbers,
            )


        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )

        out_video_file_paths.append(out_video_file_path)

    print_green("Here is a repeat of all the downloadables:")

    for out_video_file_path in out_video_file_paths:
        instruct_how_to_download_file_to_laptop(
            file_path=out_video_file_path
        )

if __name__ == "__main__":
    test_onnx_infer_range_of_frames_1()
