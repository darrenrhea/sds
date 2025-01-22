"""
python foreground_segment_video.py example_config.json
"""

import sys
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


def process_batch(
    clip_id,
    model_id,
    torch_device,
    model,
    threshold,
    batch_occupancy,  # usually 64, but that last remainder batch may have less
    batch_of_color_tiles,
    batch_of_segmented_tiles,
    batch_index_to_frame_index_and_tile,
    answers_currently_being_worked_on,  # this is the return value
    how_many_tiles_remain_to_be_solved_for_frame
):
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
            
            answer_pil = PIL.Image.fromarray(
                answers_currently_being_worked_on[frame_index] * 255
            )
            answer_path = Path(
                f"{clip_id}_{frame_index:06d}_segmentedby_{model_id}.png"
            ).expanduser()
            answer_pil.save(str(answer_path))
            print(f"pri {answer_path}")
            del answers_currently_being_worked_on[frame_index]
        
usage_message = """\
Usage:

    python foreground_segment_a_whole_frame.py <clip_id> <model_id> <frame_index>

    for example since the video frame 

    ~/awecom/data/clips/gsw1/frames/gsw1_150878.jpg

    exists and the model

    /home/drhea/r/trained_models/gsw1_trained_on_2f_7h_relevant.tar

    exists, you can do this:

    python foreground_segment_a_whole_frame.py gsw1 gsw1_trained_on_2f_7h_relevant 150878
    
    then do something to view the result like:
    
    pri gsw1_150878_segmentedby_gsw1_trained_on_2f_7h_relevant.png
"""

def main():
    if len(sys.argv) < 4:
        print(usage_message)
        sys.exit(1)
    
    valid_clip_ids = ["swinney1", "swinney2", "swinney3", "gsw1"]   
    clip_id = sys.argv[1]
    assert clip_id in valid_clip_ids, f"ERROR: bad clip_id? only valid clip_ids are {valid_clip_ids}"
    model_id = sys.argv[2]
    
    try:
        single_frame_index = int(sys.argv[3])
    except:
        print(f"ERROR: bad frame_index {sys.argv[3]}")
        print(usage_message)
        sys.exit(1)
    
    threshold = 0.5
    torch_device, model = get_the_torch_device_and_model(
        model_id=model_id
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

    for frame_index in [single_frame_index]:
        print(f"Doing frame_index {frame_index}")
        video_frame_path = Path(
            f"~/awecom/data/clips/{clip_id}/frames/{clip_id}_{frame_index:06d}.jpg"
        ).expanduser()

        video_frame_pil = PIL.Image.open(str(video_frame_path))
        video_frame_np = np.array(video_frame_pil)
        
        # it can take a while to develop the segmentation of a frame:
        answers_currently_being_worked_on[frame_index] = np.zeros(
            shape=(photo_height, photo_width),
            dtype=np.uint8
        )

        list_of_subrectangles = [
            dict(i_min=0, i_max=1080, j_min=0, j_max=1920)
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
                f"{clip_id}_{single_frame_index:06d}_segmentedbymodel_{model_id}.png"
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
                    model_id=model_id,
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
                batch_occupancy = 0
    
    print(f"batch_occupancy = {batch_occupancy}")

    process_batch(
        clip_id=clip_id,
        model_id=model_id,
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
