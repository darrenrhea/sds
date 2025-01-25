import time
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
import shutil
from prii import (
     prii
)
import textwrap
import onnx
from colorama import Fore, Style
from pathlib import Path
import onnxruntime as ort
import numpy as np
import PIL
import argparse


def logistic_sigmoid(x):
  return 1 / (1 + np.exp(-x))

argp = argparse.ArgumentParser(
    description=textwrap.dedent(
        """
        Infer an ONNX model on an image.
        """
    ),
    usage=textwrap.dedent(
        """\
        Do something like:

        python onnx_infer.py \\
        --show_in_iterm2 \\
        --onnx_file_path \\
        /shared/onnx/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000106.onnx \\
        --clip_id \\
        hou-sas-2024-10-17-sdi \\
        --original_suffix \\
        _original.jpg \\
        --first_frame_index 145000 \\
        --last_frame_index 400000 \\
        --step 100 \\
        --is_logistic_sigmoid_baked_in \\
        True

        python onnx_infer.py \\
        --show_in_iterm2 \\
        --onnx_file_path \\
        /shared/onnx/u3fasternets-floor-1865frames-1920x1088-wednesday-nba2024finalgame5_epoch000106.onnx \\
        --clip_id \\
        slday4game1 \\
        --original_suffix \\
        _original.jpg \\
        --first_frame_index 150000 \\
        --last_frame_index 300000 \\
        --step 1000 \\
        --is_logistic_sigmoid_baked_in \\
        True

        Enjoy.        
        """
    ),
)

argp.add_argument("--onnx_file_path", type=str)
argp.add_argument("--clip_id", type=str)
# frames_dir
argp.add_argument("--frames_dir", type=str)
argp.add_argument("--original_suffix", type=str)
argp.add_argument("--first_frame_index", type=int, required=True)
argp.add_argument("--last_frame_index", type=int, required=True)
argp.add_argument("--step", type=int, required=True)
argp.add_argument("--show_in_iterm2", action='store_true')
argp.add_argument('--is_logistic_sigmoid_baked_in', required=True, type=str)  # binary flags are weird

opt = argp.parse_args()
frames_dir = Path(opt.frames_dir).resolve()
assert frames_dir.is_dir(), f"{frames_dir} is not an extant directory"
show_in_iterm2 = opt.show_in_iterm2
original_suffix = opt.original_suffix
valid_original_suffixes = ["_original.jpg", "_original.png", ".png", ".jpg"]
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
first_frame_index = opt.first_frame_index
last_frame_index = opt.last_frame_index
step = opt.step
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

sess_options = ort.SessionOptions()
ort_sess = ort.InferenceSession(
    str(onnx_file_path),
    sess_options=sess_options,
    providers=["CUDAExecutionProvider"]
)

output_names = [output.name for output in ort_sess.get_outputs()]
print(f"{output_names=}")

for frame_index in range(first_frame_index, last_frame_index + 1, step):
    print(f"{frame_index=}")    
    image_file_path = frames_dir / f"{clip_id}_{frame_index:06d}{original_suffix}"
    assert image_file_path.is_file(), f"{image_file_path} is not a file"

    image_pil = PIL.Image.open(image_file_path)

    rgb_hwc_np_u8 = np.array(image_pil)
    if show_in_iterm2:
        prii(rgb_hwc_np_u8)

    padded_rgb_hwc_np_u8 = np.zeros((1088, 1920, 3), dtype=np.uint8)
    padded_rgb_hwc_np_u8[:rgb_hwc_np_u8.shape[0], :rgb_hwc_np_u8.shape[1], :] = rgb_hwc_np_u8



    # Convert to NCHW under alexnet /imagenet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalized_rgb_hwc_np_f32 = (padded_rgb_hwc_np_u8.astype(np.float32) / 255 - mean) / std

    normalized_rgb_chw_np_f32 = np.transpose(normalized_rgb_hwc_np_f32, (2, 0, 1))
    normalized_rgb_chw_np_f16 = normalized_rgb_chw_np_f32.astype(np.float16)

    x = np.expand_dims(normalized_rgb_chw_np_f16, axis=0)
    # print(f"{x.shape=}")
    # print(f"{x.dtype=}")
    # x = np.zeros(
    #     shape=(1, 3, 1088, 1920),  # maybe (3, 1088, 1920)
    #     dtype=np.float16
    # )

    
    outputs = ort_sess.run(
        output_names=output_names,
        input_feed={'input': x}
    )

    # print(f"{len(outputs)=}")
    # print(f"{outputs[0].shape=}")


    logits_f16 = outputs[0]
    # features_f16 = outputs[1]
    # print(f"{features_f16.shape=}")
    # print(f"{logits_f16.shape=}")


    if is_logistic_sigmoid_baked_in:  # if the model has a sigmoid baked in, don't do it again:
        grayscale_f16 = logits_f16[0, 0, :, :].astype(np.float16)
    else:
        grayscale_f16 = logistic_sigmoid(logits_f16[0, 0, :, :].astype(np.float16))



    grayscale_u8 = np.round(grayscale_f16 * 255).astype(np.uint8)
    if show_in_iterm2:
        prii(grayscale_u8)

    rgba_hwc_np_u8 = np.zeros(
        shape=(
            1080,
            1920,
            4,
        ),
        dtype=np.uint8
    )

    if show_in_iterm2:
        prii(
            padded_rgb_hwc_np_u8
        )

    rgba_hwc_np_u8[:, :, :3] = rgb_hwc_np_u8
    rgba_hwc_np_u8[:, :, 3] = grayscale_u8[:1080, :]

    convention = "floor_not_floor"
    mother_dir = Path(f"/shared/preannotations/{convention}")
    mother_dir.mkdir(exist_ok=True, parents=True)
    folder = mother_dir / clip_id
    folder.mkdir(exist_ok=True)
    out_path = folder / f"{clip_id}_{frame_index:06d}_nonfloor.png"
    if show_in_iterm2:
        prii(rgba_hwc_np_u8)


    write_rgba_hwc_np_u8_to_png(
        rgba_hwc_np_u8=rgba_hwc_np_u8,
        out_abs_file_path=out_path
    )
    
    print(f"pri {out_path}")

    shutil.copy(
        src=image_file_path,
        dst=folder
    )

    start  = time.time()
    print(f"{Fore.YELLOW}")
    print("These better be about 0.0 and 1.0 for most interesting images that have foreground and background")
    print(f"{Style.RESET_ALL}")
    print(f"{np.min(grayscale_f16)=}")
    print(f"{np.max(grayscale_f16)=}")
    stop  = time.time()
    print(f"{stop - start=}")


