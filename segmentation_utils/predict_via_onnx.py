# to predict from just the {fn}.onnx file
# you neet to comment out the line 52 in https://github.com/muellerzr/fastinference/blob/master/fastinference/onnx.py
# self.dls = torch.load(fn+'.pkl')
# because we want nothing to do with pickle as it is a Python thing


# https://github.com/muellerzr/fastinference/blob/master/fastinference/onnx.py

# install these two ssince they claim they can drop it to ONNX:
# pip install fastinference
# pip install fastinference[onnx-gpu]
# Then
# the python standard library inspect has be imported inside the library, so edit it:
# pico /home/drhea/miniconda3/envs/floor_not_floor/lib/python3.8/site-packages/fastinference/onnx.py
# add the line: import inspect to the top.

import sys
import time
import numpy as np
import PIL
import PIL.Image
import torch
from pathlib import Path
import pprint as pp
# from fastai.vision.all import *
# from fastai.vision.learner import unet_learner
# from fastai.vision.models import resnet34
# from fastai.vision.models.unet import DynamicUnet
import torch.nn as nn
from fastinference import *
from fastinference.onnx import *
from pathlib import Path


def make_one_chw_0_to_1_float32_numpy(cropping_index):
    image_path = Path(f"~/r/gsw1/full_croppings/{cropping_index}_color.png").expanduser()
    image_pil = PIL.Image.open(image_path)
    image_np_uint8 = np.array(image_pil)
    image_np_float32 = image_np_uint8.astype(np.float32)
    image_np_float32 /= 255.0
    image_np_float32 = image_np_float32.transpose(2, 0, 1)
    image_np_float32 = image_np_float32[ :, :, :]
    assert 0.0 <= np.min(image_np_float32)
    assert np.max(image_np_float32) <= 1.0
    assert image_np_float32.shape == (3, 400, 400)
    return image_np_float32


def make_a_batch(k, bs):
    """
    make_a_batch(0, bs=8) will make a batch which is
    a bs x c x h x w float32 numpy array
    out of the first 8 images.
    make_a_batch(1, bs=8) would make a batch out of the next 8 images.
    etc.
    """
    batch_np_float32 = np.zeros(shape=(bs, 3, 400, 400), dtype=np.float32)
    for batch_index in range(bs):
        batch_np_float32[batch_index, :, :, :] = make_one_chw_0_to_1_float32_numpy(bs * k + batch_index)
    return batch_np_float32


fn = "fastai_32e_21f"
fast_onnx = fastONNX(fn=fn)
print("type of fast_onnx is", type(fast_onnx))

print("Starting to make batches:")

bs = 16
num_batches = 6

batches = [
    make_a_batch(k=k, bs=bs)
    for k in range(num_batches)
]
print("Finished making batches.")

out_batches = []  # we are going to save the outputs

start_time = time.time()
for k in range(num_batches):
    batch = batches[k]
    

    # inps = [torch.randn(bs, 3, 400, 400, device='cuda')]
    inps = [torch.tensor(batch, device='cuda')]  # inps is a singleton list since the network has only one input tensor.
    assert inps[0].size() == torch.Size([bs, 3, 400, 400])
    assert inps[0].dtype == torch.float32

    outs = fast_onnx.predict(inps)
    print(type(outs[0]))  # it is a numpy array
    print(outs[0].shape)
    print(outs[0].dtype)
    ans = outs[0]
    assert ans.shape == (bs, 2, 400, 400)
    out_batches.append(ans)

  
stop_time = time.time()
print(f"Took {stop_time - start_time} seconds to do {num_batches} batches of {bs}")

for k, out_batch in enumerate(out_batches):
    log_probs = out_batch[:, 1, :, :]
    probs = np.exp(log_probs)
    for batch_index in range(bs):
        vis = (probs[batch_index] > 0.5).astype(np.uint8) * 255
        out_pil = PIL.Image.fromarray(vis)
        output_path = f"onnx_output/batch_{k}_{batch_index}.png"
        out_pil.save(output_path)
        print(f"See {output_path}")