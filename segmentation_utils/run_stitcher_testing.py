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
which_gpu = config_dict['which_gpu']
model_name = config_dict['model_name']
increment_frame_index_by = config_dict['increment_frame_index_by']
save_color_information_into_masks = config_dict['save_color_information_into_masks']
architecture = config_dict['architecture']
nn_input_width = config_dict['nn_input_width']
nn_input_height = config_dict['nn_input_height']
photo_width = config_dict['photo_width']
photo_height = config_dict['photo_height']


assert which_gpu in [0, 1, 2, 3], "gpu id must be 0 or 1, see nvidia-smi"

path = Path('~/brooklyn_nets_barclays_center/fastai_vs_deeplabv3_400x400').expanduser()
model_path = Path("~/r/trained_models").expanduser()

torch_device = my_pytorch_utils.get_the_correct_gpu("NVIDIA RTX A5000", which_copy=which_gpu)
# torch_device = my_pytorch_utils.get_the_correct_gpu("NVIDIA Quadro RTX 8000", which_copy=which_gpu)
fnames = list(path.glob('*_color.png')) # list of input frame paths
codes = np.array(["nonfloor", "floor"])
def label_func(fn):
    return path/f"{fn.stem[:-6]}_nonfloor{fn.suffix}"
dls = SegmentationDataLoaders.from_label_func(
    path=path, 
    bs=32, 
    fnames=fnames,
    label_func=label_func, 
    codes=codes,
    valid_pct=0.1,
    seed=42, # random seed
)
# architecture = sys.argv[7]
if architecture == "resnet34":
    arch = resnet34
elif architecture == "resnet18":
    arch = resnet18
learner = unet_learner(dls=dls, arch=arch)
model = learner.load(model_path/model_name)
model = model.to(torch_device)
torch.cuda.synchronize()
# model.eval()
threshold = 0.5
input_extension = "png" # bmp is faster to read and write, but huge
original_height = 1080
original_width = 1920

i_min = 0
j_min = 0
j_max = 1920
i_max = 1080
# j_max = photo_width
# i_max = photo_height
width = j_max - j_min
height = i_max - i_min
num_patches_per_frame = math.ceil(j_max/nn_input_width)*math.ceil(i_max/nn_input_width)
print(f"num patches per frame {num_patches_per_frame}")
stride = nn_input_width

originals_dir = Path(f'~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_test').expanduser()
out_dir = Path(f"~/r/segmentation_utils/fastai_temp").expanduser()
out_dir.mkdir(exist_ok=True)
start_time = time.time()
num_images_scored = 0
num_frames = len([file_name for file_name in originals_dir.iterdir() if str(file_name).endswith('color.png')])
print(f"num frames {num_frames}")
tot_num_patches = num_frames*num_patches_per_frame
print(f"tot num patches {tot_num_patches}")
patches = np.zeros(
        shape=(tot_num_patches, 3, nn_input_height, nn_input_width),
        dtype=np.float32
)
patches_counter = 0
lefts = [x for x in range(0, width - nn_input_width, stride)] + [width - nn_input_width]
uppers = [y for y in range(0, height - nn_input_height, stride)] + [height - nn_input_height]
all_chunks = [(left, upper) for left in lefts for upper in uppers]
# for each frame and each pixel in a frame, total score stores the score for that pixel.
total_score = torch.zeros([num_frames, height, width], dtype=torch.int16)
num_times_scored = torch.zeros([num_frames, height, width], dtype=torch.int32)
original_image_array = np.zeros(shape=(num_frames, height, width, 3), dtype=np.int8)
batch_number_to_frame_name_and_patch = {}
# print(f"first frame index type {type(first_frame_index)}")
# batch_number_to_frame_index_and_patch[patches_counter][first_frame_index] = []
# print(f"dictionary {batch_number_to_frame_index_and_patch}")

# print(f"chunking time {chunk_end - chunk_start}")
frame_index_to_batch_indices = dict()
batch_number_to_frame_index_and_patch = dict()
frame_name_to_index = dict()
frame_index_to_name = dict()
frame_counter = 0
for test_file in originals_dir.iterdir():
    if str(test_file).endswith("color.png"):
        test_file_name = str(test_file).split("/")[-1].rsplit('_', 1)[0]
        frame_name_to_index[test_file_name] = frame_counter
        frame_index_to_name[frame_counter] = test_file_name
        print(f"processing frame {test_file_name}")
        frame_index_to_batch_indices[frame_counter] = []
        image_path = originals_dir / f"{test_file_name}_color.{input_extension}"
        print(f"pri {image_path}")

        assert image_path.is_file(), f"{image_path} does not exist!"
        pil_start = time.time()
        img_pil = PIL.Image.open(str(image_path)).convert("RGB")
        pil_end = time.time()
        print(f"pil image open time {pil_end - pil_start}")
    
        hwc_np_uint8 = np.array(img_pil)

        channel_change_start = time.time()
        chw_np_uint8 = np.transpose(
            hwc_np_uint8,
            axes=(2, 0, 1)
        )
        channel_change_end = time.time()
        # print(f"channel change time {channel_change_end - channel_change_start}")

        convert_to_float_32_start = time.time()
        chw = chw_np_uint8[:, i_min:i_max, j_min:j_max].astype(np.float32) / 255.0  # float [0,1] the whole image
        # chw = np.transpose(np_float32,
        #                    axes=(2, 0, 1))  # transpose the whole image to chw
        convert_to_float_32_end = time.time()
        # print(f"convert to float32 time {convert_to_float_32_end - convert_to_float_32_start}")
            
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalize_start = time.time()
        normalized = (chw - mean[..., None, None]) / std[..., None, None]
        normalize_end = time.time()
        # print(f"normalize time {normalize_end - normalize_start}")
        normalized_size_mb = sys.getsizeof(normalized)/math.pow(10, 6)
        print(f"normalized size {normalized_size_mb} MB")
        print(f"for 3000 frames this would amount to about {3000*normalized_size_mb/math.pow(10, 3)} GB")

        # load up as much as a full batch:
        for (left, upper) in all_chunks:
            right = left + nn_input_width
            lower = upper + nn_input_height
            # batch[cntr, :, :, :] = normalized[:, upper:lower, left:right]
            patches[patches_counter, :, :, :] = normalized[:, upper:lower, left:right]
            frame_index_to_batch_indices[frame_counter].append(patches_counter)
            batch_number_to_frame_index_and_patch[patches_counter] = []
            batch_number_to_frame_index_and_patch[patches_counter].append(frame_counter)
            batch_number_to_frame_index_and_patch[patches_counter].append([upper, lower, left, right])
            patches_counter += 1
        
        frame_counter += 1

print(f"patches size {sys.getsizeof(patches)/math.pow(10, 6)} MB")
# print(patches.shape)

batch_size = 24
num_batches = math.ceil(tot_num_patches/batch_size)
# print(f"all batches length {len(patches)} and dictionary length {len(batch_number_to_frame_index_and_patch)}")


#print("exiting")
#sys.exit(0)

model.eval()
for_loop_start = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast():
        for i in range(num_batches):
            stride_counter = 0
            batch_start_index = i*batch_size
            batch_end_index = (i+1)*batch_size
            if i == 1:
                tot_process_start = time.time()
            # print(f"batch start index {batch_start_index}")
            # print(f"batch end index {batch_end_index}")
            batch = patches[batch_start_index:batch_end_index, :, :, :]
            # print(f"batch shape {batch.shape}")           
            # predict that batch
            tensor_start = time.time()
            xb_cpu = torch.tensor(batch)
            torch.cuda.synchronize()
            tensor_end = time.time()
            # print(f"convert to tensor time {tensor_end - tensor_start}")
            gpu_start = time.time()
            xb = xb_cpu.to(torch_device)
            torch.cuda.synchronize()
            gpu_end = time.time()
            # print(f"gpu load time {gpu_end - gpu_start}")
            solve_start = time.time()
            out = model(xb)
            torch.cuda.synchronize()
            solve_end = time.time()
            print(f"Inference time {(solve_end - solve_start)}")
            tensor_log_start = time.time()
            log_probs_torch = F.log_softmax(out.type(torch.DoubleTensor), dim=1)
            torch.cuda.synchronize()
            tensor_log_end = time.time()
            print(f"tensor log time {tensor_log_end - tensor_log_start}")
            threshold_start = time.time()
            chunk_binary_prediction = log_probs_torch[:, 1, :, :] > np.log(threshold)
            # chunk_binary_prediction_np = chunk_binary_prediction.detach().cpu().numpy().astype(np.uint8)
            threshold_end = time.time()
            print(f"Inference and thresholding time {(threshold_end - tensor_start)}")
            out_start = time.time()
            cntr = 0
            for patch_index in range(batch_start_index, min(batch_end_index, tot_num_patches)):
                # print(f"patch index {patch_index}")
                frame_index = batch_number_to_frame_index_and_patch[patch_index][0]
                upper = batch_number_to_frame_index_and_patch[patch_index][1][0]
                lower = batch_number_to_frame_index_and_patch[patch_index][1][1]
                left = batch_number_to_frame_index_and_patch[patch_index][1][2]
                right = batch_number_to_frame_index_and_patch[patch_index][1][3]
                # print(f"{patch_index} {upper} {lower} {left} {right}")
                total_score[frame_index, upper: upper + nn_input_height, left: left + nn_input_width] += chunk_binary_prediction[cntr, :, :]
                num_times_scored[frame_index, upper: upper + nn_input_height, left: left + nn_input_width] += 1
                stride_counter += 1
                cntr += 1
            out_end = time.time()
            if i == num_batches - 2:
                tot_process_end = time.time()
            print(f"Inference, thresholding, and saving time {out_end - tensor_start}")

for_loop_end = time.time()
for_loop_time = for_loop_end - for_loop_start
total_time = tot_process_end - tot_process_start
print(f"Time to batch process all {num_frames} frames: {total_time} seconds, avg rate of {num_frames/total_time} frames/second")
print(f"Time to for loop over {num_frames} frames: {for_loop_time} seconds, avg rate of {num_frames/for_loop_time} frames/second")
# print(f"num times scored shape {num_times_scored.shape}")
assert torch.min(
        num_times_scored) >= 1, "ERROR: not every pixel was scored at least once!"
fraction_of_votes = total_score / num_times_scored
binary_prediction = (fraction_of_votes >= 0.5).detach().cpu().numpy().astype(np.uint8)  # any positive??!!
assert np.all(
    np.logical_or(
        binary_prediction == 0,
        binary_prediction == 1
    )
)

for frame_index in range(binary_prediction.shape[0]):

    if (save_color_information_into_masks):
        create_pil_start = time.time()
        out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        out_hwc_rgba_uint8[:, :, :3] = original_image_array[frame_index, :, :, :]
        out_hwc_rgba_uint8[:, :, 3] = binary_prediction[frame_index, :, :] * 255
        out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
        out_pil = out_pil.resize((original_width, original_height), Image.ANTIALIAS)
        create_pil_stop = time.time()
        print(f"Create PIL took {create_pil_stop - create_pil_start}")
        # save it to file:
        out_file_name = out_dir / f"{frame_index_to_name[frame_index]}.jpg"
        # even full color goes fast if you turn down compression level:
        png_save_start = time.time()
        # out_pil.save(out_file_name, "PNG", optimize=False, compress_level=0)
        out_pil.save(out_file_name, "JPEG")
        png_save_stop = time.time()
        print(f"Took {png_save_stop - png_save_start} seconds to save png")
    else:
        create_pil_start = time.time()
        out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
        out_hw_grayscale_uint8[:, :] = binary_prediction[frame_index, :, :] * 255
        out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
        out_pil = out_pil.resize((original_width, original_height), Image.ANTIALIAS)
        create_pil_stop = time.time()
        print(f"Create PIL took {create_pil_stop - create_pil_start}")
        # save it to file:
        out_file_name = out_dir / f"{frame_index_to_name[frame_index]}.jpg"
        png_save_start = time.time()
        out_pil.save(out_file_name, "JPEG")  # fast because it is black white
        png_save_stop = time.time()
        print(f"Took {png_save_stop - png_save_start} seconds to save png")
        print(f"Wrote: {out_file_name}")

    print(f"See {out_file_name}")
