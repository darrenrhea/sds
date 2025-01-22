import sys
import numpy as np
import torch
import torch.nn.functional as F
import time
from fastai.vision.all import *
import PIL
import PIL.Image
import my_pytorch_utils
from pathlib import Path
import numpy as np
import math
import better_json


num_frames_per_gpu_batch = 1

# This program takes in num_frames_per_gpu_batch = 4 image files at a time,
# each of which is originally 1920x1080
# but they are downsampled by 2 in both dimensions immediately to 960x540.

# Then each 960x540 image gets covered by 6 320x280 "neural network input tiles."
# That is a total of 24 different 320x280 NN inputs, which together form a batch tensor
# of size 24 x 3 x 280 x 320 (Batch x RGBChan x Height x Width).

# This batch gets put through the NN, and we get segmentations for the tiles.
# Then we have to reassemble the tiles into a segmentation for the 960x540,
# which is then upsampled to a segmentation of the fullsize 1920x1080.


# Iterate over all frames and collect batches. The size of each batch
# is based on the desired neural network input width and height.
# Once a collection of batches is obtained, each batch in the
# collection is processed by the neural network.
# Note that we need to maintain the association between
# any patch inside a batch with the original frame that it came from
# so that we can reconstruct an answer for each frame from the
# answers of all patches that constitute it, even if they
# span multiple batches.

json_config_path = sys.argv[1]
json_config_file = Path(json_config_path).expanduser()
config_dict = better_json.load(json_config_file)
video_name = config_dict["video_name"]
color_original_pngs_dir = Path(f"~/awecom/data/clips/{video_name}/frames").expanduser()

first_frame_index = config_dict["first_frame_index"]
last_frame_index = config_dict["last_frame_index"]
num_frames = last_frame_index - first_frame_index + 1  # num_frames: the total number of video frames that get processed

gpu_substring = config_dict["gpu_substring"]
which_gpu = config_dict["which_gpu"]

model_name = config_dict["model_name"]
masking_attempt_id = config_dict["masking_attempt_id"]
increment_frame_index_by = config_dict["increment_frame_index_by"]
save_color_information_into_masks = config_dict["save_color_information_into_masks"]
architecture = config_dict["architecture"]
nn_input_width = config_dict["nn_input_width"]
nn_input_height = config_dict["nn_input_height"]
original_width = config_dict["original_width"]
original_height = config_dict["original_height"]
downsample_factor = config_dict["downsample_factor"]

# Before covering it by tiles, we downsample by the downsampling_factor:
small_height = original_height // downsample_factor
small_width = original_width // downsample_factor

assert which_gpu in [
    0,
    1,
], "ERROR: which_gpu must be 0 or 1 to identify which of the two gpus selected by gpu_substring you want."

# WARNING: this next line seems crazy specific to gsw1, but it is actually necessary
# Because fastai is essentially forcing us to make a dataloader as-if we were training
# despite that we are only doing inference.
# What has to be in this directory for it to work?
path = Path("~/r/gsw1/224_224_one_third_downsample_croppings").expanduser()
# path = Path('~/r/gsw1/280_320_croppings').expanduser()
# path = Path(f'~/r/brooklyn_nets_barclays_center/320_280_one_half_downsampled_croppings').expanduser()
model_path = Path("~/r/trained_models").expanduser()

torch_device = my_pytorch_utils.get_the_correct_gpu(gpu_substring, which_copy=which_gpu)
fnames = list(path.glob("*_color.png"))  # list of input frame paths
codes = np.array(["nonfloor", "floor"])


def label_func(fn):
    return path / f"{fn.stem[:-6]}_nonfloor{fn.suffix}"


dls = SegmentationDataLoaders.from_label_func(
    path=path,
    bs=32,
    fnames=fnames,
    label_func=label_func,
    codes=codes,
    valid_pct=0.1,
    seed=42,  # random seed
)

if architecture == "resnet34":
    arch = resnet34
elif architecture == "resnet18":
    arch = resnet18
learner = unet_learner(dls=dls, arch=arch)
model = learner.load(model_path / model_name)
model = model.to(torch_device)
model.eval()  # we aren't training, only infering
torch.cuda.synchronize()

threshold = 0.5
input_extension = "jpg"  # bmp is faster to read and write, but huge


width = small_width
height = small_height
num_patches_per_frame = math.ceil(small_width / nn_input_width) * math.ceil(small_height / nn_input_width)
print(f"num patches per frame {num_patches_per_frame}")
j_stride = nn_input_width  # no overlap: the tiles are 0:320, 320:640, 640:960 as far as j.
i_stride = 260  # Because nn_input_height - 20 = 260.  TODO: generalize this.
# vertically the top row tiles have i range over 0:280,
# then the next row of tiles has i range over in 260:540.

masking_attempts_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts").expanduser()
masking_attempts_dir.mkdir(exist_ok=True)
out_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts/{masking_attempt_id}").expanduser()
out_dir.mkdir(exist_ok=True)
start_time = time.time()
num_images_scored = 0
lefts = [x for x in range(0, width - nn_input_width, j_stride)] + [width - nn_input_width]
uppers = [y for y in range(0, height - nn_input_height, i_stride)] + [height - nn_input_height]
upper_left_corners_of_all_tiles = [(left, upper) for left in lefts for upper in uppers]

print(f"The {width}x{height} image will be covered by these nn_input_tiles:")
for left, upper in upper_left_corners_of_all_tiles:
    right = left + nn_input_width
    lower = upper + nn_input_height
    print(f"nn_input_tile = image[{upper}:{lower}, {left}:{right}]")

# At some point we found that you can get fairly decent speed by infering 24 320x280 tiles per batch:
print(f"num_frames_per_gpu_batch {num_frames_per_gpu_batch}")

# for each frame and each pixel in a frame, total_score stores the score for that pixel.
total_score = np.zeros(shape=[num_frames_per_gpu_batch, height, width], dtype=np.int)
num_times_scored_np = np.zeros([num_frames_per_gpu_batch, height, width], dtype=np.int)
original_image_array = np.zeros(shape=(num_frames_per_gpu_batch, original_height, original_width, 3), dtype=np.int8)

# we infer 24 different tiles, each of which is a part of a particular video frame.
# this stores which videoframe that tile came from so that we can reassemble:
tile_index_to_frame_index = {}

# we have num_frames_per_gpu_batch * 6 tiles = 24 tiles.  Each has bounds [upper, lower, left, right]
# so that it was cut out via tile = image[upper:lower, left:right]
tile_index_to_tile_bounds = {}

num_tiles_per_batch = num_frames_per_gpu_batch * num_patches_per_frame
print(f"num_tiles_per_batch = {num_tiles_per_batch}")

batch_of_tiles = np.zeros(shape=(num_tiles_per_batch, 3, nn_input_height, nn_input_width), dtype=np.float32)

# we are going to do the simplest thing that gives decent speed,
# a group of 4 960x540 images will be inferred at a time
# by putting a single tensor, batch_of_tiles, through the neural network:


# "group_starts_at_frame_index" steps by num_frames_per_gpu_batch through the frame range:
for group_starts_at_frame_index in range(first_frame_index, last_frame_index + 1, num_frames_per_gpu_batch):
    print(f"Processing a group of {num_frames_per_gpu_batch} frames starting at = {group_starts_at_frame_index}")

    tile_index_to_within_the_group_frame_index = dict()

    # this should be the 4 images that make up the group of 4 images:
    image_paths = [
        color_original_pngs_dir / f"{video_name}_{group_starts_at_frame_index + delta:06d}.{input_extension}"
        for delta in range(num_frames_per_gpu_batch)
    ]

    # check that at least the first of the 4 exists, for the last group the other 3 may not exist:
    assert image_paths[0].is_file(), f"The first of the 4, {image_path}, does not exist!"

    tile_index = 0  # this should go from 0 to 23 since there are 4 * 6 = 24 tiles to put through the NN

    # within_the_group_frame_index ranges from 0 to 3:
    for within_the_group_frame_index, image_path in enumerate(image_paths):
        # the index of the image we are cutting into tiles:
        frame_index = group_starts_at_frame_index + within_the_group_frame_index
        if image_path.is_file():
            img_pil = PIL.Image.open(str(image_path)).convert("RGB")  # 1920x1080
            smaller_pil = img_pil.resize((small_width, small_height), Image.ANTIALIAS)  # 960x540
            hwc_np_uint8 = np.array(smaller_pil)  # as a hwc numpy array
            original_image_array[within_the_group_frame_index, :, :, :] = np.array(img_pil)
        else:
            hwc_np_uint8 = np.zeros((small_width, small_height), dtype=np.uint8)  # as a hwc numpy array
            original_image_array[within_the_group_frame_index, :, :, :] = 0

        chw_np_uint8 = np.transpose(hwc_np_uint8, axes=(2, 0, 1))

        # convert the image to float32s ranging over [0,1]:
        chw_np_float32 = chw_np_uint8[:, :, :].astype(np.float32) / 255.0

        # normalize it like AlexNet:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (chw_np_float32 - mean[..., None, None]) / std[..., None, None]

        # chop out the 6 tiles, stick them into the batch:
        for (left, upper) in upper_left_corners_of_all_tiles:
            right = left + nn_input_width
            lower = upper + nn_input_height
            batch_of_tiles[tile_index, :, :, :] = normalized[:, upper:lower, left:right]
            tile_index_to_frame_index[tile_index] = frame_index
            tile_index_to_within_the_group_frame_index[tile_index] = within_the_group_frame_index
            tile_index_to_tile_bounds[tile_index] = [upper, lower, left, right]
            tile_index += 1
    # At this point, the batch tensor should be ready to go through

    # with torch.no_grad():
    #     with torch.cuda.amp.autocast():
    # predict that batch
    start_time = time.time()
    xb_cpu = torch.tensor(batch_of_tiles)
    xb = xb_cpu.to(torch_device)
    out = model(xb)
    log_probs_torch = F.log_softmax(out.type(torch.DoubleTensor), dim=1)
    probs_times_255_gpu = torch.exp(log_probs_torch[:, 1, :, :]) * 255
    probs_times_255 = probs_times_255_gpu.detach().cpu().numpy().astype(np.uint8)
    stop_time = time.time()
    duration = stop_time - start_time
    images_per_second = num_frames_per_gpu_batch / duration
    print(f"Going at {images_per_second} images per second")

    for tile_index in range(num_tiles_per_batch):
        upper, lower, left, right = tile_index_to_tile_bounds[tile_index]
        within_the_group_frame_index = tile_index_to_within_the_group_frame_index[tile_index]
        total_score[within_the_group_frame_index, upper:lower, left:right] = probs_times_255[tile_index, :, :]

    # write the 4 segmented images out to disk:
    for within_the_group_frame_index in range(num_frames_per_gpu_batch):
        frame_index = group_starts_at_frame_index + within_the_group_frame_index
        if frame_index > last_frame_index:  # that last group of 4 can have less than 4 images
            continue

        if save_color_information_into_masks:
            mask_hw_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
            # mask_hw_uint8[:, :] = binary_prediction[within_the_group_frame_index, :, :] * 255
            mask_hw_uint8[:, :] = total_score[within_the_group_frame_index, :, :]

            mask_pil = PIL.Image.fromarray(mask_hw_uint8)
            upscaled_mask_pil = mask_pil.resize((original_width, original_height), Image.ANTIALIAS)

            final_hwc_rgba_uint8 = np.zeros(shape=(original_height, original_width, 4), dtype=np.uint8)
            final_hwc_rgba_uint8[:, :, :3] = original_image_array[within_the_group_frame_index, :, :, :]
            final_hwc_rgba_uint8[:, :, 3] = np.array(upscaled_mask_pil)
            final_pil = PIL.Image.fromarray(final_hwc_rgba_uint8)

            # save it to file:
            out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
            # even full color goes fast if you turn down compression level:
            final_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)
            print(f"pri {out_file_name}")
        else:
            out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
            # out_hw_grayscale_uint8[:, :] = binary_prediction[within_the_group_frame_index, :, :] * 255
            out_hw_grayscale_uint8[:, :] = total_score[within_the_group_frame_index, :, :]
            small_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
            full_size_pil = small_pil.resize((original_width, original_height), Image.ANTIALIAS)
            # save it to file:
            out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
            png_save_start = time.time()
            full_size_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)  # fast because it is black white
            png_save_stop = time.time()
            print(f"Took {png_save_stop - png_save_start} seconds to save png")

            print(f"pri {out_file_name}")
