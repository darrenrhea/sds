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
from pathlib import Path
import pprint as pp

def make_list_of_input_output_path_pairs(
   config_dict
):
    requests = config_dict["requests"]
   
    model_name = config_dict["model_name"]
   
    list_of_input_output_path_pairs = []
    for request in requests:
        active = request.get("active", True)
        if not active:
            print(f"{Fore.YELLOW}WARNING: SKIPPING REQUEST")
            pp.pprint(request)
            print(f"{Style.RESET_ALL}")
            continue
        else:
            print(f"{Fore.GREEN}DOING REQUEST")
            pp.pprint(request)
            print(f"{Style.RESET_ALL}")

        save_color_information_into_masks = request.get("save_color_information_into_masks", False)

        only_multiples_of = request.get("only_multiples_of", 1)

        clip_id = request["clip_id"]
        clip_dir = Path(f"/awecom/data/clips/{clip_id}")
        in_dir = clip_dir / "frames"

        frame_ranges = request["frame_ranges"]

        masking_attempt_id = model_name + "_" + ("color" if save_color_information_into_masks else "bw")
        out_dir = clip_dir / "masking_attempts" / masking_attempt_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for f, l in frame_ranges:
            frame_indices = [x for x in range(f, l+1) if x % only_multiples_of == 0]
            for frame_index in frame_indices:
                in_file = in_dir / f"{clip_id}_{frame_index:06d}.jpg"
                if not in_file.is_file():
                    print(f"{Fore.YELLOW}WARNING:\n    {in_file}\ndoes not exist, skipping.\n{Style.RESET_ALL}")
                    continue
                out_file = out_dir / f"{clip_id}_{frame_index:06d}_nonfloor.png"
                list_of_input_output_path_pairs.append(
                    (in_file, out_file, save_color_information_into_masks)
                )

    return list_of_input_output_path_pairs


def list_of_paths_caller(config_dict, list_of_input_output_path_pairs):
    """
    Trying to design a more flexible caller.
    This still takes in a config_dict to set the model information.
    It takes in a list of pairs of paths. Each pair is the input image path followed by a path where you want to write the answer.
    """
    gpu_substring = config_dict["gpu_substring"]
    which_gpu = config_dict["which_gpu"]
    model_name = config_dict["model_name"]
    architecture = config_dict["architecture"]
    nn_input_width = config_dict["nn_input_width"]
    nn_input_height = config_dict["nn_input_height"]
    original_width = config_dict["original_width"]
    original_height = config_dict["original_height"]
    downsample_factor = config_dict["downsample_factor"]

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


    for group_starts_at_index in range(0, len(list_of_input_output_path_pairs), num_frames_per_gpu_batch):
        start_time = time.time()
       
        short_list_of_input_output_path_pairs = list_of_input_output_path_pairs[
            group_starts_at_index
            :
            group_starts_at_index + num_frames_per_gpu_batch
        ]

        image_paths = [x[0] for x in short_list_of_input_output_path_pairs]
        output_paths = [x[1] for x in short_list_of_input_output_path_pairs]

        pil_images = []
        for image_path in image_paths:
            img_pil = PIL.Image.open(str(image_path)).convert("RGB")
            pil_images.append(img_pil)
                

        segmented_frames_np_ihw = segmenter.infer_full_size_rgb_pil_frames(pil_frames=pil_images)
        # check that it returned the right kind of thing:
        assert segmented_frames_np_ihw.shape == (num_frames_per_gpu_batch, 1080, 1920)
        assert segmented_frames_np_ihw.dtype == np.uint8

        # write the 4 segmented images out to disk:
        for within_the_group_frame_index in range(len(pil_images)):
            save_color_information_into_masks = short_list_of_input_output_path_pairs[within_the_group_frame_index][2]
            in_file_name = image_paths[within_the_group_frame_index]
            out_file_name = output_paths[within_the_group_frame_index]
            mask_hw_uint8 = segmented_frames_np_ihw[within_the_group_frame_index, :, :].copy()
            mask_pil = PIL.Image.fromarray(mask_hw_uint8)
            
            if save_color_information_into_masks:
                if downsample_factor == 1:
                    upscaled_mask_pil = mask_pil
                else:
                    upscaled_mask_pil = mask_pil.resize((original_width, original_height), PIL.Image.ANTIALIAS)
                final_hwc_rgba_uint8 = np.zeros(shape=(original_height, original_width, 4), dtype=np.uint8)
                final_hwc_rgba_uint8[:, :, :3] = np.array(pil_images[within_the_group_frame_index].convert("RGB"))
                final_hwc_rgba_uint8[:, :, 3] = np.array(upscaled_mask_pil)
                final_pil = PIL.Image.fromarray(final_hwc_rgba_uint8)
                # even full color goes fast if you turn down compression level:
                final_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)
            else:
                if downsample_factor == 1:
                    mask_pil.save(out_file_name, "PNG")  # fast because it is black white
                else:
                    full_size_pil = mask_pil.resize((original_width, original_height), PIL.Image.ANTIALIAS)
                    full_size_pil.save(out_file_name, "PNG")  # fast because it is black white
            
            print(f"pri {in_file_name}")
            print(f"pri {out_file_name}")

                

        stop_time = time.time()
        duration = stop_time - start_time
        images_per_second = num_frames_per_gpu_batch / duration
        print(f"Going at {images_per_second} images per second")


if __name__ == "__main__":
    json_config_path = sys.argv[1]
    json_config_file = Path(json_config_path).expanduser()
    config_dict = better_json.load(json_config_file)
    assert config_dict["config_type"] == "arbitrary_frame_ranges"

    list_of_input_output_path_pairs = make_list_of_input_output_path_pairs(
        config_dict=config_dict
    )
    
    pp.pprint(list_of_input_output_path_pairs)

    num_frames_per_gpu_batch = 1
   
    list_of_paths_caller(
        config_dict=config_dict,
        list_of_input_output_path_pairs=list_of_input_output_path_pairs
    )
