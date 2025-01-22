"""
python foreground_segment_video.py example_config.json
"""

import sys
import time
import better_json as bj
from pathlib import Path
from CameraParameters import CameraParameters
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import pprint as pp
from cover_subrectangles import cover_subrectangles
import copy
from get_subrectangles_that_need_masking import get_subrectangles_that_need_masking
from score_batch import score_batch
from get_the_torch_device_and_model import get_the_torch_device_and_model
from colorama import Fore, Style
# for x in `seq 10350 10450` ; do scp $(printf "lam:/home/drhea/awecom/data/clips/swinney1/frames/swinney1_%06d.jpg" $x) .; done
# for x in `seq 10350 10450` ; do scp $(printf "lam:/home/drhea/awecom/data/clips/swinney1/tracking_attempts/chaz_locked/swinney1_%06d_camera_parameters.json" $x) .; done

def score_this_batch_putting_the_answers_here(
    torch_device,
    model,
    threshold,
    batch_of_color_tiles,
    batch_of_segmented_tiles
):
    """In the real-deal, the neural network UNet will suck up this batch of 64 color tiles,
    segment each into a binary mask, and stick those 64 masked tiles into batch_of_segmented_tiles.
    Right now we are happy to have as a placeholder something that sort-of grayscales the 64 rgb tiles
    to make 64 fake answers.
    """
    # batch_of_segmented_tiles[:, :, :] = batch_of_color_tiles.astype(np.float).mean(axis=3).clip(0, 255).astype(np.uint8)
    # return

    score_batch(
        torch_device=torch_device,
        model=model,
        threshold=threshold,
        batch_of_color_tiles=batch_of_color_tiles,
        batch_of_segmented_tiles=batch_of_segmented_tiles  # answer comes back here
    )


video_frames_processed_counter = 0

def process_batch(
    clip_id,
    tracking_attempt_id,
    masking_attempt_id,
    torch_device,
    model,
    threshold,
    batch_occupancy,  # usually 64, but that last remainder batch may have less
    batch_of_color_tiles,
    batch_of_segmented_tiles,
    batch_index_to_frame_index_and_tile,
    answers_currently_being_worked_on,  # this is the return value
    how_many_tiles_remain_to_be_solved_for_frame,
    save_to_file
):
    global video_frames_processed_counter
    score_this_batch_putting_the_answers_here(
        torch_device=torch_device,
        model=model,
        threshold=threshold,
        batch_of_color_tiles=batch_of_color_tiles,
        batch_of_segmented_tiles=batch_of_segmented_tiles
    )
    # now that the tiles have been scored, put the answers back to the frames they came from: 
    for batch_index in range(batch_occupancy):
        tile_answer = batch_of_segmented_tiles[batch_index]
        dct = batch_index_to_frame_index_and_tile.get(batch_index)
        if dct is None:
            print(f"breaking because {batch_index} is not a key in batch_index_to_frame_index_and_tile")
            break
        frame_index = dct["frame_index"]
        tile = dct["tile"]
        i_min = tile["i_min"]
        i_max = tile["i_max"]
        j_min = tile["j_min"]
        j_max = tile["j_max"]
        answers_currently_being_worked_on[frame_index][i_min:i_max, j_min:j_max] = tile_answer
        how_many_tiles_remain_to_be_solved_for_frame[frame_index] -= 1
        if how_many_tiles_remain_to_be_solved_for_frame[frame_index] == 0:
            if save_to_file:
                answer_pil = PIL.Image.fromarray(
                    answers_currently_being_worked_on[frame_index] * 255
                )
                answer_path = Path(
                    f"~/awecom/data/clips/{clip_id}/masking_attempts/{masking_attempt_id}/{clip_id}_{frame_index:06d}_nonfloor.png"
                ).expanduser()
                answer_pil.save(str(answer_path))
                print(f"pri {answer_path}")
            video_frames_processed_counter += 1
            del answers_currently_being_worked_on[frame_index]
        


def main():
    config_file_path = Path(sys.argv[1]).expanduser()
    config = bj.load(config_file_path)
    clip_id = config["clip_id"]
    masking_attempt_id = config["masking_attempt_id"]
    tracking_attempt_id = config.get("tracking_attempt_id")
    if tracking_attempt_id is None:
        print(f"{Fore.YELLOW}WARNING: no camera tracking so masking the entire video frame{Style.RESET_ALL}")
    
    masking_attempts_dir = Path(
        f"~/awecom/data/clips/{clip_id}/masking_attempts"
    ).expanduser()
    masking_attempts_dir.mkdir(exist_ok=True)

    mask_output_dir = Path(
        f"~/awecom/data/clips/{clip_id}/masking_attempts/{masking_attempt_id}"
    ).expanduser()
    mask_output_dir.mkdir(exist_ok=True)
    save_to_file = True
    first_frame_index = config["first_frame"]
    last_frame_index = config["last_frame"]
    ads = config["ads"]
    model_id = config["model_id"]
    model_architecture = config["model_architecture"]
    assert model_architecture in ["UNet1", "UNet2"]
    
    threshold = 0.5
    torch_device, model = get_the_torch_device_and_model(
        model_id=model_id,
        model_architecture=model_architecture
    )

    photo_width = 1920
    photo_height = 1080
    nn_input_width = 224
    nn_input_height = 224

    batch_of_color_tiles = np.zeros(
        shape=(64, nn_input_height, nn_input_width, 3),
        dtype=np.uint8
    )
    
    batch_of_segmented_tiles = np.zeros(
        shape=(64, nn_input_height, nn_input_width),
        dtype=np.uint8
    )

    batch_occupancy = 0  # the schoolbus starts out empty
    batch_entry_to_frame = dict() 
    answers_currently_being_worked_on = dict()  # memory leak worries
    how_many_tiles_remain_to_be_solved_for_frame = dict()
    batch_index_to_frame_index_and_tile = dict()
    run_start_time = time.time()
    for frame_index in range(first_frame_index, last_frame_index + 1):
        stop_time = time.time()
        frames_per_second = video_frames_processed_counter / (stop_time - run_start_time)
        print(f"De facto frames_per_second = {frames_per_second}")

        print(f"Doing frame_index {frame_index}")
        video_frame_path = Path(
            f"~/awecom/data/clips/{clip_id}/frames/{clip_id}_{frame_index:06d}.jpg"
        ).expanduser()

        image_to_numpy_start_time = time.time()
        video_frame_pil = PIL.Image.open(str(video_frame_path))
        video_frame_np = np.array(video_frame_pil)
        image_to_numpy_stop_time = time.time()
        print(f"Loading image: {image_to_numpy_stop_time - image_to_numpy_start_time}")

        
        # it can take a while to develop the segmentation of a frame:
        answers_currently_being_worked_on[frame_index] = np.zeros(
            shape=(photo_height, photo_width),
            dtype=np.uint8
        )

        if tracking_attempt_id is not None:
            camera_parameters_path = Path(
                f"~/awecom/data/clips/{clip_id}/tracking_attempts/{tracking_attempt_id}/{clip_id}_{frame_index:06d}_camera_parameters.json"
            ).expanduser()

            camera_parameters_json = bj.load(camera_parameters_path)
            camera_parameters = CameraParameters.from_dict(camera_parameters_json)

            list_of_subrectangles = get_subrectangles_that_need_masking(
                camera_parameters=camera_parameters,
                ads=ads,
                drawable=None
            )
        else:
            list_of_subrectangles = [
                {'i_max': 1080, 'i_min': 0, 'j_max': 1920, 'j_min': 0}
            ]

        tiles = cover_subrectangles(
            photo_width=photo_width,
            photo_height=photo_height,
            nn_input_width=nn_input_width,
            nn_input_height=nn_input_height,
            list_of_subrectangles=list_of_subrectangles
        )

        how_many_tiles_remain_to_be_solved_for_frame[frame_index] = len(tiles)
        
        # for some frames, there are no tiles to be scores,
        # and their empty answer needs to be dropped to file:
        if how_many_tiles_remain_to_be_solved_for_frame[frame_index] == 0:
            answer_pil = PIL.Image.fromarray(
                answers_currently_being_worked_on[frame_index] * 255
            )
            answer_path = Path(
                f"~/awecom/data/clips/{clip_id}/masking_attempts/{masking_attempt_id}/{clip_id}_{frame_index:06d}_nonfloor.png"
            ).expanduser()
            answer_pil.save(str(answer_path))
            print(f"pri {answer_path}")
            del answers_currently_being_worked_on[frame_index]
       
        for tile in tiles:
            i_min = tile["i_min"]
            i_max = tile["i_max"]
            j_min = tile["j_min"]
            j_max = tile["j_max"]
            batch_of_color_tiles[batch_occupancy, :, :, :] = video_frame_np[i_min:i_max, j_min:j_max, :]
            batch_index_to_frame_index_and_tile[batch_occupancy] = dict(
                tile=copy.deepcopy(tile),
                frame_index=frame_index
            )
            batch_occupancy += 1

            if batch_occupancy == 64:  # its full, ship it
                process_batch(
                    clip_id=clip_id,
                    tracking_attempt_id=tracking_attempt_id,
                    masking_attempt_id=masking_attempt_id,
                    torch_device=torch_device,
                    model=model,
                    threshold=threshold,
                    batch_occupancy=batch_occupancy,
                    batch_of_color_tiles=batch_of_color_tiles,
                    batch_of_segmented_tiles=batch_of_segmented_tiles,
                    batch_index_to_frame_index_and_tile=batch_index_to_frame_index_and_tile,
                    answers_currently_being_worked_on=answers_currently_being_worked_on,
                    how_many_tiles_remain_to_be_solved_for_frame=how_many_tiles_remain_to_be_solved_for_frame,
                    save_to_file=save_to_file
                )
                batch_occupancy = 0
    
    print(f"batch_occupancy = {batch_occupancy}")

    process_batch(
        clip_id=clip_id,
        tracking_attempt_id=tracking_attempt_id,
        masking_attempt_id=masking_attempt_id,
        torch_device=torch_device,
        model=model,
        threshold=threshold,
        batch_occupancy=batch_occupancy,
        batch_of_color_tiles=batch_of_color_tiles,
        batch_of_segmented_tiles=batch_of_segmented_tiles,
        batch_index_to_frame_index_and_tile=batch_index_to_frame_index_and_tile,
        answers_currently_being_worked_on=answers_currently_being_worked_on,
        how_many_tiles_remain_to_be_solved_for_frame=how_many_tiles_remain_to_be_solved_for_frame
    )


        
if __name__ == "__main__":
    main()
