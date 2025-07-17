import onnx
from colorama import Fore, Style
from pathlib import Path
import onnxruntime as ort
import numpy as np
from print_image_in_iterm2 import print_image_in_iterm2
import PIL
import argparse


def logistic_sigmoid(x):
  return 1 / (1 + np.exp(-x))

argp = argparse.ArgumentParser(
    description="""
Infer an ONNX model on an image.
Do something like:

python onnx_infer.py \
--onnx_file_path \
/shared/onnx/u3fasternetsimagemaskmotionbluratntaematterl1classweights5frames307_epoch000240.onnx
--is_logistic_sigmoid_baked_in True \
--clip_id \
u3fasternetsimagemaskmotionbluratntaematterl1classweights5frames307_epoch000240
--frame_index 153403
"""
)
argp.add_argument("--onnx_file_path", type=str)
argp.add_argument("--clip_id", type=str)
argp.add_argument("--original_suffix", type=str)
argp.add_argument("--frame_index", type=int, default=153403)
argp.add_argument('--is_logistic_sigmoid_baked_in', required=True, type=str)  # binary flags are weird
opt = argp.parse_args()

original_suffix = opt.original_suffix
valid_original_suffixes = ["_original.png", ".png", ".jpg"]
assert (
  original_suffix in valid_original_suffixes
), f"{original_suffix=} seems invalid. It should be one of {valid_original_suffixes}"

if opt.is_logistic_sigmoid_baked_in.lower() in ["true", "t", "yes", "y", "1"]:
    is_logistic_sigmoid_baked_in = True
else:
    is_logistic_sigmoid_baked_in = False

assert isinstance(is_logistic_sigmoid_baked_in, bool)
print(f"{is_logistic_sigmoid_baked_in=}")

onnx_file_path = Path(opt.onnx_file_path).resolve()
frame_index = opt.frame_index
clip_id = opt.clip_id


assert (
  onnx_file_path.is_file()
), f"{Fore.RED}ERROR:\n{onnx_file_path}\nis not an extant file!{Style.RESET_ALL}"

onnx_model = onnx.load(onnx_file_path)
try:
    onnx.checker.check_model(onnx_model)
except Exception as e:
    print(f"{Fore.RED}ERROR:Something is wrong with {onnx_file_path}{Style.RESET_ALL}")
    raise e

print(f"{Fore.GREEN}{onnx_file_path} is a valid ONNX model{Style.RESET_ALL}")
frames_dir = Path(f"/media/drhea/muchspace/clips/{clip_id}/frames")


image_file_path = frames_dir / f"{clip_id}_{frame_index:06d}{original_suffix}"
assert image_file_path.is_file(), f"{image_file_path} is not a file"

image_pil = PIL.Image.open(image_file_path)

rgb_hwc_np_u8 = np.array(image_pil)
print_image_in_iterm2(rgb_np_uint8=rgb_hwc_np_u8)
padded_rgb_hwc_np_u8 = np.zeros((1088, 1920, 3), dtype=np.uint8)
padded_rgb_hwc_np_u8[:rgb_hwc_np_u8.shape[0], :rgb_hwc_np_u8.shape[1], :] = rgb_hwc_np_u8

# Convert to NCHW under alexnet /imagenet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalized_rgb_hwc_np_f32 = (
    (padded_rgb_hwc_np_u8.astype(np.float32) / 255 - mean) / std
).astype(np.float32)

normalized_rgb_chw_np_f32 = np.transpose(normalized_rgb_hwc_np_f32, (2, 0, 1))
normalized_rgb_chw_np_f16 = normalized_rgb_chw_np_f32.astype(np.float16)

x = np.expand_dims(normalized_rgb_chw_np_f32, axis=0)
print(f"{x.shape=}")
print(f"{x.dtype=}")
# x = np.zeros(
#     shape=(1, 3, 1088, 1920),  # maybe (3, 1088, 1920)
#     dtype=np.float16
# )

ort_sess = ort.InferenceSession(str(onnx_file_path))
output_names = [output.name for output in ort_sess.get_outputs()]
print(f"{output_names=}")
outputs = ort_sess.run(
    output_names=output_names,
    input_feed={'input': x}
)

print(f"{len(outputs)=}")
print(f"{outputs[0].shape=}")


logits_f16 = outputs[0]
# features_f16 = outputs[1]
# print(f"{features_f16.shape=}")
print(f"{logits_f16.shape=}")


if is_logistic_sigmoid_baked_in:  # if the model has a sigmoid baked in, don't do it again:
    grayscale_f16 = logits_f16[0, 0, :, :].astype(np.float16)
else:
    grayscale_f16 = logistic_sigmoid(logits_f16[0, 0, :, :].astype(np.float16))



grayscale_u8 = np.round(grayscale_f16 * 255).astype(np.uint8)
rgba_hwc_np_u8 = np.zeros((1080, 1920, 4), dtype=np.uint8)
rgba_hwc_np_u8[:, :, :3] = rgb_hwc_np_u8
rgba_hwc_np_u8[:, :, 3] = grayscale_u8[:1080, :]
print_image_in_iterm2(rgba_np_uint8=rgba_hwc_np_u8)
print_image_in_iterm2(grayscale_np_uint8=grayscale_u8)


print(f"{Fore.YELLOW}")
print("These better be about 0.0 and 1.0 for most interesting images that have foreground and background")
print(f"{Style.RESET_ALL}")
print(f"{np.min(grayscale_f16)=}")
print(f"{np.max(grayscale_f16)=}")

