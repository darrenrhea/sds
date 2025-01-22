# md5sum fastai_32e_21f.onnx
# 2fc89484b591cea052a54090ab310a11  fastai_32e_21f.onnx
# https://github.com/muellerzr/fastinference/blob/master/fastinference/onnx.py

# install these two ssince they claim they can drop it to ONNX:
# pip install fastinference
# pip install fastinference[onnx-gpu]
# Then
# the python standard library inspect has be imported inside the library, so edit it:
# pico /home/drhea/miniconda3/envs/floor_not_floor/lib/python3.8/site-packages/fastinference/onnx.py
# add the line: import inspect to the top.

# You MUST change the torch.onnx.export invokation to ssay
# opset_version=11 see https://pytorch.org/docs/stable/onnx.html#getter

# https://github.com/muellerzr/fastinference/blob/master/fastinference/onnx.py

import sys
import torch
from pathlib import Path
import pprint as pp
from fastai.vision.all import *
from fastai.vision.learner import unet_learner
from fastai.vision.models import resnet34
from fastai.vision.models.unet import DynamicUnet
import torch.nn as nn
from fastinference import *
from fastinference.onnx import *
from pathlib import Path

torch.cuda.set_device("cuda:1")  # dummy input is 16

path = Path('~/r/gsw1/full_croppings').expanduser()
fnames = list(path.glob('*_color.png')) # list of input frame paths

def label_func(fn):
    return path/f"{fn.stem[:-6]}_nonfloor{fn.suffix}"
# label_func(fnames[0])
# for i in range(len(fnames)):
#     print(label_func(fnames[i]))
# len([label_func(fn) for fn in fnames])
codes = np.array(["players", "nonfloor"])

dls = SegmentationDataLoaders.from_label_func(
    path, 
    bs=16, 
    fnames = fnames, 
    label_func = label_func, 
    codes = codes,
    valid_pct=0.1,
    seed=42, # random seed,
    batch_tfms=aug_transforms(
        mult=1.0,
        do_flip=True, 
        flip_vert=False, 
        max_rotate=10.0, 
        min_zoom=1.0, 
        max_zoom=1.1, 
        max_lighting=0.2, 
        max_warp=0.2, 
        p_affine=0.75, 
        p_lighting=0.75, 
        xtra_tfms=None, 
        size=None, 
        mode='bilinear', 
        pad_mode='reflection', 
        align_corners=True, 
        batch=False, 
        min_scale=1.0)
)

print("type of dls is", type(dls))



m = resnet34(pretrained=True)

model_path = Path("~/r/trained_models/fastai_32e_21f").expanduser()
learner = unet_learner(dls, resnet34)

learner = learner.load(model_path)
learner.model.cuda()

print(f"bs was originally {learner.dls[0].bs}")
dummy_inp = next(iter(learner.dls[0]))


print("type of learn.model is:", type(learner.model))

print("type of learner is ", type(learner))  # <class 'fastai.learner.Learner'>

base_name_to_save_onnx_to = "fastai_32e_21f"

learner.to_onnx(base_name_to_save_onnx_to)
