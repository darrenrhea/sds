# Suppose we are given the original RGB image/video-frame, as well as two segmentation answers
# a_mask_np_uint8 and b_mask_np_uint8
# as gray-scale PNG images, i.e. where each pixel is labeled between 0 or 255,
# where 255 means 100% foreground and 0 means 100% background.
# We want to compare the two segmentation answers and output the results as follows:
# show, maybe in green, the things that were added from foreground in b_mask_np_uint8 compared to a.
# To contextualize the above, show in color the original image with this green set blinking on and off.
# Similarly,
# show, maybe in red, the things that were removed from the foreground in b_mask_np_uint8 compared to a.
# show blinking on and off in color the original image with this red set blinking on and off.
import numpy as np
from pathlib import Path
from get_possible_clip_ids import get_possible_clip_ids
import pprint as pp
import imageio as iio
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
import pygifsicle  # pip install pygifsicle
import cv2
from rsync_utils import download_via_rsync
from image_openers import open_alpha_channel_image_as_a_single_channel_grayscale_image

def three_channel_ify_grayscale(
    grayscale_np_uint8: np.ndarray,   # height x width grayscale image as numpy array of uint8
) -> np.ndarray:
    assert grayscale_np_uint8.ndim == 2
    shared_height = grayscale_np_uint8.shape[0]
    shared_width = grayscale_np_uint8.shape[1]
    assert grayscale_np_uint8.dtype == np.uint8
    result = np.zeros((shared_height, shared_width, 3), dtype=np.uint8)
    result[:, :, 0] = grayscale_np_uint8
    result[:, :, 1] = grayscale_np_uint8
    result[:, :, 2] = grayscale_np_uint8
    assert result.ndim == 3
    assert result.shape[0] == shared_height
    assert result.shape[1] == shared_width
    assert result.shape[2] == 3
    assert result.dtype == np.uint8
    return result
    

def save_sequence_of_images_as_movie_old(
    sequence_of_images: list[np.ndarray],   # list of 3 channel Height x Width x Channel original image as numpy array of uint8,
    output_file_path: Path,   # path to output image
):
    # https://imageio.readthedocs.io/en/stable/reference/userapi.html#migration-from-v2
    shared_height = sequence_of_images[0].shape[0]
    shared_width = sequence_of_images[0].shape[1]

    with iio.imopen(output_file_path, io_mode="w", plugin="pillow") as writer:
        for rgb_np_uint8 in sequence_of_images:
            assert rgb_np_uint8.ndim == 3
            assert rgb_np_uint8.shape[0] == shared_height
            assert rgb_np_uint8.shape[1] == shared_width
            assert rgb_np_uint8.shape[2] == 3
            writer.write(rgb_np_uint8, duration=500, loop=1)
            # print_image_in_iterm2(rgb_np_uint8=rgb_np_uint8)
    # pygifsicle.optimize(output_gif_path)



def save_sequence_of_images_as_movie(
    sequence_of_images: list[np.ndarray],   # list of 3 channel Height x Width x Channel original image as numpy array of uint8,
    output_file_path: Path,   # path to output image
):
    # https://imageio.readthedocs.io/en/stable/reference/userapi.html#migration-from-v2
    shared_height = sequence_of_images[0].shape[0]
    shared_width = sequence_of_images[0].shape[1]

    for rgb_np_uint8 in sequence_of_images:
        assert rgb_np_uint8.ndim == 3
        assert rgb_np_uint8.shape[0] == shared_height
        assert rgb_np_uint8.shape[1] == shared_width
        assert rgb_np_uint8.shape[2] == 3

    # Each video has a frame per second which is number of frames in every second
    frame_per_second = 60

    w, h = None, None
    for frame in sequence_of_images: # files_and_duration:
        duration = 0.5
        if w is None:
            # Setting up the video writer
            h, w, _ = frame.shape
            # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
            writer = cv2.VideoWriter(
                filename=str(output_file_path),
                fourcc=fourcc,
                fps=frame_per_second,
                frameSize=(w, h)
            )

        # Repeating the frame to fill the duration
        for repeat in range(int(np.round(duration * frame_per_second))):
            bgr_np_uint8 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_np_uint8)

    writer.release()




def draw_indicated_subset_with_color_on_rgb_image(
    indicator_np_uint8: np.ndarray,   # height x width mask as numpy array of uint8
    color: list[int], # what color to use for the subset
    original_image_rgb_np_uint8: np.ndarray,   # 3 channel Height x Width x Channel original image as numpy array of uint8
) -> np.ndarray:   # 3 channel Height x Width x Channel original image as numpy array of uint8
    assert len(color) == 3
    for c in color:
        assert 0 <= c <= 255
        assert isinstance(c, int)
    assert original_image_rgb_np_uint8.ndim == 3
    assert original_image_rgb_np_uint8.shape[2] == 3
    shared_height = original_image_rgb_np_uint8.shape[0]
    shared_width = original_image_rgb_np_uint8.shape[1]
    assert indicator_np_uint8.ndim == 2
    assert original_image_rgb_np_uint8.shape[0] == indicator_np_uint8.shape[0]
    assert original_image_rgb_np_uint8.shape[1] == indicator_np_uint8.shape[1]
    assert original_image_rgb_np_uint8.shape[0] == shared_height
    assert original_image_rgb_np_uint8.shape[1] == shared_width
    assert indicator_np_uint8.dtype == np.uint8
    assert original_image_rgb_np_uint8.dtype == np.uint8

    where_to_color = indicator_np_uint8[:, :] > 0
    result = original_image_rgb_np_uint8.copy()
    result[where_to_color, :] = color
    assert result.dtype == np.uint8
    assert result.shape == original_image_rgb_np_uint8.shape
    return result


def compare_two_segmentation_answers(
    original_image_rgb_np_uint8: np.ndarray,   # 3 channel Height x Width x Channel original image as numpy array of uint8
    a_mask_np_uint8: np.ndarray,   # height x width segmentation answer a_mask_np_uint8 as numpy array of uint8 
    b_mask_np_uint8: np.ndarray,   # height x width segmentation answer b_mask_np_uint8 as numpy array of uint8
    blink_period_in_seconds: float,   # blink period in seconds
    output_gif_path: Path,   # path to output image
) -> None:
    assert isinstance(output_gif_path, Path), f"{output_gif_path=} is not a pathlib Path object"
    # Before getting into the details of the GIF video creation, let's create a sequence of images:
    added_indicator_np_uint8 = np.logical_and(
        a_mask_np_uint8 < 128,
        b_mask_np_uint8 > 128
    ).astype(np.uint8) * 255

    subtracted_indicator_np_uint8 = np.logical_and(
        a_mask_np_uint8 > 128,
        b_mask_np_uint8 < 128
    ).astype(np.uint8) * 255

    with_green = draw_indicated_subset_with_color_on_rgb_image(
        indicator_np_uint8=added_indicator_np_uint8,
        color=[0, 255, 0],   # green
        original_image_rgb_np_uint8=original_image_rgb_np_uint8,
    )

    with_red = draw_indicated_subset_with_color_on_rgb_image(
        indicator_np_uint8=subtracted_indicator_np_uint8,
        color=[255, 0, 0],   # red
        original_image_rgb_np_uint8=original_image_rgb_np_uint8,
    )

    grayscale_a = three_channel_ify_grayscale(
        grayscale_np_uint8=a_mask_np_uint8
    )

    grayscale_b = three_channel_ify_grayscale(
        grayscale_np_uint8=b_mask_np_uint8
    )
    
    sequence_of_images = (
        [
            original_image_rgb_np_uint8,
            with_green
        ] * 5
        +
        [
            original_image_rgb_np_uint8,
            with_red
        ] * 5
        +
        [
            original_image_rgb_np_uint8,
            grayscale_a
        ] * 5
        +
        [
            original_image_rgb_np_uint8,
            grayscale_b
        ] * 5
    )

    save_sequence_of_images_as_movie(
        sequence_of_images=sequence_of_images,
        output_file_path=output_gif_path
    )
   



def compare_two_segmentations_of_frame(
    frame_index: int,
    a_model_id: str,
    b_model_id: str
):
    a_mask_remote_abs_path = Path(
        f"/mnt/nas/volume1/videos/baseball/masks/20200725PIT-STL-CFCAM-PITCHCAST_inning1/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}_{a_model_id}.png"
    )

    b_mask_remote_abs_path = Path(
        f"/mnt/nas/volume1/videos/baseball/masks/20200725PIT-STL-CFCAM-PITCHCAST_inning1/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}_{b_model_id}.png"
    )

    original_image_remote_abs_path = Path(
        f"/mnt/nas/volume1/videos/baseball/clips/20200725PIT-STL-CFCAM-PITCHCAST_inning1/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}.jpg"
    )

    a_mask_abs_path = Path(
        f"~/baseball/masks/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}_{a_model_id}.png"
    ).expanduser()

    b_mask_abs_path = Path(
        f"~/baseball/masks/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}_{b_model_id}.png"
    ).expanduser()

    original_image_abs_path = Path(
        f"~/baseball/clips/20200725PIT-STL-CFCAM-PITCHCAST_inning1/frames/20200725PIT-STL-CFCAM-PITCHCAST_inning1_{frame_index:06d}.jpg"
    ).expanduser()

    download_via_rsync(
        src_machine="lam",
        src_path=a_mask_remote_abs_path,
        dst_path=a_mask_abs_path,
        verbose=True
    )

    download_via_rsync(
        src_machine="lam",
        src_path=b_mask_remote_abs_path,
        dst_path=b_mask_abs_path,
        verbose=True
    )

    download_via_rsync(
        src_machine="lam",
        src_path=original_image_remote_abs_path,
        dst_path=original_image_abs_path,
        verbose=True
    )


    original_image_rgb_np_uint8 = iio.v3.imread(uri=original_image_abs_path)
    assert original_image_rgb_np_uint8.ndim == 3
    assert original_image_rgb_np_uint8.shape[2] == 3

    shared_height = original_image_rgb_np_uint8.shape[0]
    shared_width = original_image_rgb_np_uint8.shape[1]

    current_directory = Path.cwd()  # NOTE: already absolute
    gifs_directory = Path.cwd() / "gifs"
    gifs_directory.mkdir(exist_ok=True)
    
    a_mask_np_uint8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(abs_file_path=a_mask_abs_path)
    b_mask_np_uint8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(abs_file_path=b_mask_abs_path)
    
    assert a_mask_np_uint8.ndim == 2, f"{a_mask_np_uint8.shape}"
    assert a_mask_np_uint8.shape[0] == shared_height
    assert a_mask_np_uint8.shape[1] == shared_width

    assert b_mask_np_uint8.ndim == 2
    assert b_mask_np_uint8.shape[0] == shared_height
    assert b_mask_np_uint8.shape[1] == shared_width

    output_gif_path = gifs_directory / f"{frame_index:06d}_{a_model_id}_to_{b_model_id}.mp4"

    compare_two_segmentation_answers(
        original_image_rgb_np_uint8=original_image_rgb_np_uint8,
        a_mask_np_uint8=a_mask_np_uint8,
        b_mask_np_uint8=b_mask_np_uint8,
        blink_period_in_seconds=0.5,
        output_gif_path=output_gif_path,
    )
    print(f"Suggest you:\nopen {output_gif_path}")



if __name__ == "__main__":
    a_model_id = "81ce-6671-4170-b8a7-no-overlap"
    b_model_id = "effs20231008"

    for frame_index in range(1000, 48552, 1000):
        compare_two_segmentations_of_frame(
            frame_index=frame_index,
            a_model_id=a_model_id,
            b_model_id=b_model_id
        )
    print("DONE")