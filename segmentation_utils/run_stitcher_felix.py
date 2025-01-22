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

# time python run_stitcher_downsampled_batch_frames.py gsw1 147000 347000 0 fastai_224p_res34_32e_10f fastai_resnet34_224 resnet34 > out34_16.txt

model_path = Path(sys.argv[1]).expanduser()
image_path = Path(sys.argv[2]).expanduser()
out_dir = Path(sys.argv[3]).expanduser()
out_dir.mkdir(exist_ok=True)
print(type(out_dir))
which_gpu = 0
save_color_information_into_masks = True
architecture = "resnet34"
nn_input_width = 320
nn_input_height = 280
photo_width = 1920
photo_height = 1080
torch_device = my_pytorch_utils.get_the_correct_gpu("NVIDIA RTX A5000", which_copy=which_gpu)
# torch_device = my_pytorch_utils.get_the_correct_gpu("NVIDIA Quadro RTX 8000", which_copy=which_gpu)

path = Path('~/r/gsw1/224_224_one_third_downsample_croppings').expanduser()
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
if architecture == "resnet34":
    arch = resnet34
elif architecture == "resnet18":
    arch = resnet18
learner = unet_learner(dls=dls, arch=arch)
model = learner.load(model_path)
model = model.to(torch_device)
torch.cuda.synchronize()
threshold = 0.5

i_min = 0
j_min = 0
j_max = photo_width
i_max = photo_height
width = j_max - j_min
height = i_max - i_min
stride = nn_input_width

img_pil = PIL.Image.open(str(image_path)).convert("RGB")
hwc_np_uint8 = np.array(img_pil)

batch_size = 24

height = i_max - i_min
width = j_max - j_min
lefts = [x for x in range(0, width - nn_input_width, nn_input_width)] + [width - nn_input_width]
uppers = [y for y in range(0, height - nn_input_height, nn_input_height)] + [height - nn_input_height]

num_times_scored = torch.zeros([height, width], dtype=torch.int32)
total_score = torch.zeros([height, width], dtype=torch.int16)

chw_np_uint8 = np.transpose(
        hwc_np_uint8,
    axes=(2, 0, 1)
)
chw = chw_np_uint8[:, i_min:i_max, j_min:j_max].astype(np.float32) / 255.0  # float [0,1] the whole image
    
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (chw - mean[..., None, None]) / std[..., None, None]

all_chunks = [(left, upper) for left in lefts for upper in uppers]

batch = np.zeros(shape=(batch_size, 3, nn_input_height, nn_input_width), dtype=np.float32)

batches = [
    all_chunks[x:x + batch_size]
    for x in range(0, len(all_chunks), batch_size)
]

model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast():
        stride_counter = 0
        for batch_of_chunks in batches:
            bs = len(batch_of_chunks)  # for the remainder, it can be smaller than batch_size
            assert bs <= batch_size
            # load up as much as a full batch:
            for cntr, (left, upper) in enumerate(batch_of_chunks):
                right = left + nn_input_width
                lower = upper + nn_input_height
                batch[cntr, :, :, :] = normalized[:, upper:lower, left:right]
            # predict that batch
            xb_cpu = torch.tensor(batch)
            xb = xb_cpu.to(torch_device)
            out = model(xb)
            log_probs_torch = F.log_softmax(out.type(torch.DoubleTensor), dim=1)
            chunk_binary_prediction = log_probs_torch[:, 1, :, :] > np.log(threshold)

            for cntr, (left, upper) in enumerate(batch_of_chunks):
                total_score[upper:upper + nn_input_height, left:left + nn_input_width] += chunk_binary_prediction[cntr, :, :]
                num_times_scored[upper:upper + nn_input_height, left:left + nn_input_width] += 1
                stride_counter += 1

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
i_max%nn_input_height
if (save_color_information_into_masks):
    out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
    out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
    out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
    out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
    # save it to file:
    # even full color goes fast if you turn down compression level:
    out_pil.save("nets1progfeed_560500_nonfloor.png", "PNG", optimize=False, compress_level=0)
else:
    out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
    out_hw_grayscale_uint8[:, :] = binary_prediction * 255
    out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
    # save it to file:
    out_pil.save(out_dir, "PNG")  # fast because it is black white

print(f"pri {out_dir}")
