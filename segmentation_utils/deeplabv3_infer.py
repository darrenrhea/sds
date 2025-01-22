"""
rm temp/*.jpg && python deeplabv3_infer.py
"""

from all_imports_for_image_segmentation import *
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import sys
from collections import OrderedDict
import torchinfo


use_case_name = "brooklyn"
crop_height = 400  # we cut slightly bigger rectangles out so that we can wiggle it to promote translation invariance
crop_width = crop_height  # we crop out squares
desired_mask_names = ["nonfloor", "inbounds"]  # the "relevant" mask tells the loss function whether we care about accuracy for each particular pixel
batch_size = 32
out_dir = Path(f"~/awecom/data/clips/BKN_CITY_2021-11-03_PGM/masking_attempts/deeplabv3_brooklyn_83f").expanduser()

# list_of_annotated_images = get_list_of_annotated_images(
#     use_case_name=use_case_name,
#     must_have_these_masks=desired_mask_names
# )

# testing_data_dir = Path(f"~/r/brooklyn_nets_barclays_center/nonfloor_segmentation_test").expanduser()
# list_of_annotated_images = [f.resolve() for f in testing_data_dir.iterdir() if f.is_file() and str(f).endswith("_color.png")]
testing_data_dir = Path(f"~/awecom/data/clips/BKN_CITY_2021-11-03_PGM/frames").expanduser()
list_of_annotated_images = []
for f in testing_data_dir.iterdir():
    abs_path = f.resolve()
    
    if not f.is_file():
        continue
    if not str(f).endswith(f".jpg"):
        continue
    str_digits = str(f)[-10:-4]
    print(f"{str_digits=}")
    frame_index = int(str_digits)
    print(f"{frame_index=}")
    if frame_index > 2000 or frame_index < 1000:
        continue
    list_of_annotated_images.append(abs_path)


print(f"list of annotated images: {list_of_annotated_images}")

# Pick a GPU.
torch_device = my_pytorch_utils.get_the_correct_gpu("5000", which_copy=1)
twentyone = True
if twentyone:
    model_load_path = Path("~/r/trained_models/deeplabv3_brooklyn_83f.pth").expanduser()
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        # num_classes=2,
        pretrained=False
    )
else:
    # model_load_path = Path("~/r/trained_models/deeplabv3_brooklyn_83f.pth").expanduser()
    model_load_path = Path("~/r/trained_models/deeplabv3_brooklyn_83f_2classes.pth").expanduser()
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        num_classes=2,
        pretrained=False
    )

# torchinfo.summary(model, (1, 3, 400, 400))
# sys.exit(0)

model.load_state_dict(torch.load(model_load_path))
model.eval().to(torch_device)

# traced_model = torch.fx.symbolic_trace(model)
# print(traced_model.graph)

# put it into evaluation mode for speed:
# model.eval()
# num_images_to_infer_total = 80
num_images_to_infer_total = len(list_of_annotated_images)
num_images_to_infer_per_batch = 2
# let us preload a bunch of images for better timing

all_images_preloaded = np.zeros(
    shape=[num_images_to_infer_total, 3, 1080, 1920],
    dtype=np.uint8
)

all_answers = np.zeros(
    shape=[num_images_to_infer_total, 1080, 1920],
    dtype=np.uint8
)

all_answers_torch = torch.zeros(
    [num_images_to_infer_total, 1080, 1920],
    dtype=torch.uint8
)

xb_cpu = torch.zeros([num_images_to_infer_per_batch, 3, 1080, 1920], dtype=torch.float32)

image_pils = []
inferred_image_ids = []
for k in range(num_images_to_infer_total):
    # image_pil = Image.open(list_of_annotated_images[k % 90]["image_path"]).convert("RGB")
    # inferred_image_ids.append(list_of_annotated_images[k % 90]["image_id_str"])
    image_pil = Image.open(list_of_annotated_images[k]).convert("RGB")
    image_id = str(list_of_annotated_images[k]).split("/")[-1].rsplit('_', 1)[0]
    inferred_image_ids.append(image_id)
    image_pils.append(image_pil)
    
    image_hwc_np_uint8 = np.array(image_pil)
    all_images_preloaded[k, :, :, :] = np.transpose(image_hwc_np_uint8, axes=(2, 0, 1))

# print(inferred_image_ids)
alexnet_std = torch.tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).to(torch_device)
alexnet_mean = torch.tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).to(torch_device)
print(f"loading finished")

# def fn(d:OrderedDict[str, torch.Tensor]):
#     return d['out']

# traced_fn = torch.jit.trace(fn, {'out': torch.rand(2, 3)})

# @torch.jit.script
def get_result(xb, alexnet_mean, alexnet_std):
    normalized = (xb.to(torch.float32) / 255.0 - alexnet_mean[..., None, None]) / alexnet_std[..., None, None]
    outputs = model(normalized)
    # print(f"outputs size {outputs['out'].size()}")
    # print(f"outputs type {outputs['out'].dtype}")
    # out = traced_fn(outputs)
    # return torch.zeros([1, 2, 1080, 1920], device=torch_device)
    return outputs['out']


# with torch.jit.fuser("fuser2"):
#     for i in range(3):
#         get_result(xb_cpu.to(torch_device), alexnet_mean, alexnet_std)

torch.cuda.synchronize()
start_time = time.time()
for batch_index in range(num_images_to_infer_total // num_images_to_infer_per_batch):
    xb_cpu[:, :, :, :] = torch.tensor(
        all_images_preloaded[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :, :]
    )
    
    xb = xb_cpu.to(torch_device)
    # normalized = (xb.to(torch.float32) / 255.0 - alexnet_mean[..., None, None]) / alexnet_std[..., None, None]
    # outputs = model(normalized)
    # out = outputs['out']
    out = get_result(xb, alexnet_mean, alexnet_std)
    # outputs = model(normalized)
    # out = outputs['out']
    all_answers_torch[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :]  = torch.argmax(out, dim=1).to(torch.uint8)
    # all_answers[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :]  = torch.argmax(out, dim=1).to(torch.uint8).detach().cpu().numpy()

torch.cuda.synchronize()  
stop_time = time.time()
print(f"{stop_time  - start_time} seconds to do {num_images_to_infer_total}, i.e. {num_images_to_infer_total / (stop_time  - start_time)} fps")

all_answers = all_answers_torch.detach().cpu().numpy()
for k, image_pil in enumerate(image_pils):
    shifted_all_answers = np.roll(all_answers[k, :, :], shift=7, axis=0)
    PIL.Image.fromarray(255*shifted_all_answers).save(
        Path(f"{out_dir}/{inferred_image_ids[k]}.jpg").expanduser()
    )

print(f"for x in /*.jpg ; do pri $x ; done")
# print(f"{list_of_annotated_images}")
# for my_image in list_of_annotated_images:
#     print(f"{my_image['image_id_str']}")
# for image_id in inferred_image_ids:
#     print(f"\"{image_id}\",")
# print(f"{inferred_image_ids}")
assert len(inferred_image_ids) == num_images_to_infer_total
