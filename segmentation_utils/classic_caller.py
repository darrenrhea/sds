import sys
import numpy as np
import PIL
import PIL.Image
from pathlib import Path
import numpy as np
import better_json
from colorama import Fore, Style
from FullFrameSegmenter import FullFrameSegmenter
import os
import time


if __name__ == "__main__":
    num_frames_per_gpu_batch = 1
    json_config_path = sys.argv[1]
    json_config_file = Path(json_config_path).expanduser()
    config_dict = better_json.load(json_config_file)
    clip_id = config_dict["clip_id"]
    if "clip_id" in os.environ:
        clip_id = os.environ["clip_id"]
        print(f"{Fore.YELLOW}Overriding clip_id = {clip_id} because environment variable clip_id is set to that.{Style.RESET_ALL}")

    color_original_pngs_dir = Path(f"~/awecom/data/clips/{clip_id}/frames").expanduser()

    first_frame_index = config_dict["first_frame_index"]
    if "first_frame_index" in os.environ:
        first_frame_index = int(os.environ["first_frame_index"])
        print(f"{Fore.YELLOW}Overriding first_frame_index = {first_frame_index} via environment variable{Style.RESET_ALL}")

    last_frame_index = config_dict["last_frame_index"]
    if "last_frame_index" in os.environ:
        last_frame_index = int(os.environ["last_frame_index"])
        print(f"{Fore.YELLOW}Overriding last_frame_index = {last_frame_index} via environment variable{Style.RESET_ALL}")

    num_frames = last_frame_index - first_frame_index + 1  # num_frames: the total number of video frames that get processed

    gpu_substring = config_dict["gpu_substring"]
    which_gpu = config_dict["which_gpu"]

    model_name = config_dict["model_name"]

    if "model_name" in os.environ:
        model_name = os.environ["model_name"]
        print(f"{Fore.YELLOW}Overriding model_name = {model_name} because environment variable model_name is set to that.{Style.RESET_ALL}")


    masking_attempt_id = config_dict["masking_attempt_id"]

    if "masking_attempt_id" in os.environ:
        masking_attempt_id = os.environ["masking_attempt_id"]
        print(f"{Fore.YELLOW}Overriding masking_attempt_id = {masking_attempt_id} because environment variable masking_attempt_id is set to that.{Style.RESET_ALL}")


    increment_frame_index_by = config_dict["increment_frame_index_by"]

    save_color_information_into_masks = config_dict["save_color_information_into_masks"]
    if "save_color_information_into_masks" in os.environ:
        save_color_information_into_masks_str = os.environ["save_color_information_into_masks"]
        if save_color_information_into_masks_str.lower() in ["0", "false", "f"]:
            save_color_information_into_masks = False
        elif save_color_information_into_masks_str in ["1", "true", "t"]:
            save_color_information_into_masks = True
        else:
            print(f"{Fore.RED}save_color_information_into_masks environment variable is set to {save_color_information_into_masks_str} which is not understood.{Style.RESET_ALL}")
            sys.exit(1)

        print(f"{Fore.YELLOW}Overriding save_color_information_into_masks = {save_color_information_into_masks} because environment variable save_color_information_into_masks is set to that.{Style.RESET_ALL}")

    architecture = config_dict["architecture"]
    nn_input_width = config_dict["nn_input_width"]
    nn_input_height = config_dict["nn_input_height"]
    original_width = config_dict["original_width"]
    original_height = config_dict["original_height"]
    downsample_factor = config_dict["downsample_factor"]
    input_extension = config_dict.get("input_extension", "jpg")  # bmp is faster to read and write, but huge

    # This particluar caller outpputs to a directory:
    masking_attempts_dir = Path(f"~/awecom/data/clips/{clip_id}/masking_attempts").expanduser()
    masking_attempts_dir.mkdir(exist_ok=True)
    out_dir = Path(f"~/awecom/data/clips/{clip_id}/masking_attempts/{masking_attempt_id}").expanduser()
    out_dir.mkdir(exist_ok=True)

    segmenter = FullFrameSegmenter(
        original_width=original_width,
        original_height=original_height,
        num_frames_per_gpu_batch=num_frames_per_gpu_batch,
        gpu_substring=gpu_substring,
        which_gpu=which_gpu,
        downsample_factor=downsample_factor,
        nn_input_width=nn_input_width,
        nn_input_height=nn_input_height,
        model_name=model_name,
        architecture=architecture,
    )


    # "group_starts_at_frame_index" steps by num_frames_per_gpu_batch through the frame range:
    for group_starts_at_frame_index in range(first_frame_index, last_frame_index + 1, num_frames_per_gpu_batch):
        start_time = time.time()
        print(f"Processing a group of {num_frames_per_gpu_batch} frames starting at = {group_starts_at_frame_index}")


        # this should be the 4 images that make up the group of 4 images:
        image_paths = [
            color_original_pngs_dir / f"{clip_id}_{group_starts_at_frame_index + delta:06d}.{input_extension}"
            for delta in range(num_frames_per_gpu_batch)
        ]

        # check that at least the first of the 4 exists, for the last group the other 3 may not exist:
        assert image_paths[0].is_file(), f"The first of the 4, {image_paths[0]}, does not exist!"

        pil_images = []
        for within_the_group_frame_index, image_path in enumerate(image_paths):
            # the index of the image we are cutting into tiles:
            frame_index = group_starts_at_frame_index + within_the_group_frame_index
            if image_path.is_file():
                img_pil = PIL.Image.open(str(image_path)).convert("RGB")  # 1920x1080
                pil_images.append(img_pil)
                

        segmented_frames_np_ihw = segmenter.infer_full_size_rgb_pil_frames(pil_frames=pil_images)
        assert segmented_frames_np_ihw.shape == (num_frames_per_gpu_batch, 1080, 1920)
            # write the 4 segmented images out to disk:
        for within_the_group_frame_index in range(num_frames_per_gpu_batch):
            frame_index = group_starts_at_frame_index + within_the_group_frame_index
            if frame_index > last_frame_index:  # that last group of 4 can have less than 4 images
                continue

            if save_color_information_into_masks:
                mask_hw_uint8 = segmented_frames_np_ihw[within_the_group_frame_index, :, :].copy()
                mask_pil = PIL.Image.fromarray(mask_hw_uint8)
                upscaled_mask_pil = mask_pil.resize((original_width, original_height), PIL.Image.ANTIALIAS)

                final_hwc_rgba_uint8 = np.zeros(shape=(original_height, original_width, 4), dtype=np.uint8)
                final_hwc_rgba_uint8[:, :, :3] = np.array(pil_images[within_the_group_frame_index].convert("RGB"))
                final_hwc_rgba_uint8[:, :, 3] = np.array(upscaled_mask_pil)
                final_pil = PIL.Image.fromarray(final_hwc_rgba_uint8)

                # save it to file:
                out_file_name = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
                # even full color goes fast if you turn down compression level:
                final_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)
                print(f"pri {out_file_name}")
            else:
                out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
                # out_hw_grayscale_uint8[:, :] = binary_prediction[within_the_group_frame_index, :, :] * 255
                out_hw_grayscale_uint8[:, :] = segmented_frames_np_ihw[within_the_group_frame_index, :, :]
                small_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
                full_size_pil = small_pil.resize((original_width, original_height), Image.ANTIALIAS)
                # save it to file:
                out_file_name = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
                full_size_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)  # fast because it is black white
                print(f"pri {out_file_name}")

        stop_time = time.time()
        duration = stop_time - start_time
        images_per_second = num_frames_per_gpu_batch / duration
        print(f"Going at {images_per_second} images per second")