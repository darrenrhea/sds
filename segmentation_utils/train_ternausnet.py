# https://raw.githubusercontent.com/ternaus/TernausNet/master/ternausnet/models.py
#
# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
#
# https://www.kaggle.com/cordmaur/38-cloud-simple-unet/notebook
#
# typical `reload_ext` boilerplate so it takes up changes made to modules as best it can.
# We want jupyter notebook cells to be as wide as possible.  
import time
from all_imports_for_image_segmentation import *
from pathlib import Path
import PIL
from PIL import Image
from fastai.vision.all import (
    resnet34, unet_learner, SegmentationDataLoaders, aug_transforms
)
from fastai.distributed import *
import numpy as np
from annotated_data import get_list_of_annotated_images_from_several_directories
from randomization_utils import choose_name_from_name_to_prob
import pprint as pp
from SegmentationDatasetNoRelevance import (
     SegmentationDatasetNoRelevance
)
from colorama import Fore, Style

# potentially the initial croppings are bigger than the final nn_input
crop_height = 416
crop_width = 416
regenerate_crops = True
desired_mask_names = ["nonfloor"]
target_mask = "nonfloor"

# if downsampling, make sure to downsample the original training masks and point the function
# get_list_of_annotated_images() to the directory containing the downsampled masks.
list_of_annotated_images = get_list_of_annotated_images_from_several_directories(
    must_have_these_masks=desired_mask_names,
    directories_to_gather_from_with_limits = [
        (
            Path(f"~/r/gsw_floor/anna").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/baynzo").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/chillar").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/darren").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/grace").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/priebe").expanduser(),
            1000
        ),
        (
            Path(f"~/r/gsw_floor/thomas").expanduser(),
            1000
        ),
    ]
)

num_frames = len(list_of_annotated_images)

nn_input_width = 416
nn_input_height = nn_input_width  # the input size the neural network actually takes in


pp.pprint(list_of_annotated_images)

print(f"num_frames = {num_frames}")




# !find ~/gsw/416x416/ -type f -name '*.png' -delete


croppings_dir = "~/gsw/416x416"
Path(croppings_dir).expanduser().mkdir(parents=True, exist_ok=True)
start = time.time()
if regenerate_crops:
    cropped_hand_annotated_training_data = get_numpy_arrays_of_croppings_and_their_masks(
        list_of_annotated_images=list_of_annotated_images,
        crop_height=crop_height,
        crop_width=crop_width,
        desired_mask_names=desired_mask_names,
        mask_name_to_min_amount={ 
            target_mask: 0
        }, # we are forcing croppings to have at least one pixel of mainrectangle mask on
         mask_name_to_max_amount={
            target_mask: (crop_height*crop_width),
        }, # we are forcing croppings to have at least one pixel of mainrectangle mask off
        how_many_originals=len(list_of_annotated_images),
        how_many_crops_per_original=500,
        mask_encoding_type="alpha_channel_of_rgba"
    )

    dct = cropped_hand_annotated_training_data
    print([k for k in dct.keys()])
    assert dct["num_croppings_cut"] > 0
    assert isinstance(dct["mask_name_to_cropped_masks"], dict)
    assert isinstance(dct["cropped_originals"], np.ndarray)
    for mask_name in desired_mask_names:
        assert dct["mask_name_to_cropped_masks"][mask_name].shape[0] == dct["cropped_originals"].shape[0]
    for k in range(dct["num_croppings_cut"]):
        save_hwc_np_uint8_to_image_path(dct["cropped_originals"][k], Path(f"{croppings_dir}/{k}_color.png").expanduser())
   
        save_hwc_np_uint8_to_image_path(
            dct["mask_name_to_cropped_masks"][target_mask][k],
            Path(f"{croppings_dir}/{k}_{target_mask}.png").expanduser())

        if (k%1000 == 0):
            print(f"saving {k}")
            update = time.time()
            print(f"{update - start} seconds")
    end = time.time()
    print(f"total {end - start} seconds")
    

# * a np.array of orginal color croppings
# * another np.array of corresponding player/foreground-object masks.
# * another np.array of relevance masks.  Off of it, we do not impose any loss for getting the answer wrong.

new_cropped_originals = cropped_hand_annotated_training_data["cropped_originals"]
new_cropped_masks = cropped_hand_annotated_training_data["mask_name_to_cropped_masks"]["nonfloor"]

print(new_cropped_originals.shape)
print(new_cropped_masks.shape)


def my_augmenter(x):
    """
    WARNING: notice that this has been short-circuited to be the identity function.
    Takes in a hwc rgb uint8 numpy.array image
    randomly mutates it via like color shift, blurring, jpegification
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3  # enforce that it is height width channel
    assert x.shape[2] == 3  # enforce that it is rgb
    assert x.dtype == np.uint8  # and uint8 dtype
    return x  # dont augment at all
    
    augmented_pil = PIL.Image.fromarray(x)  # the transformations work best on PIL Images, and they mutate in-place.
    
    if np.random.randint(0, 1 + 1):
        
        blur_filter = PIL.ImageFilter.GaussianBlur(
            radius=random.uniform(0.1, 0.5)
        )
        
        augmented_pil = augmented_pil.filter(filter=blur_filter)
    

 
    augmented_pil = torchvision.transforms.functional.adjust_brightness(
        augmented_pil,
        random.uniform(0.95, 1.05)
    )
    augmented_pil = torchvision.transforms.functional.adjust_contrast(
        augmented_pil,
        random.uniform(0.95, 1.05)
    )
    augmented_pil = torchvision.transforms.functional.adjust_saturation(
        augmented_pil,
        random.uniform(0.95, 1.05)
    )
    #torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
    #torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
    #torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
    
    augmented_np = np.array(augmented_pil)
    return augmented_np


urns_with_probs_and_data_augmenters = dict(
    crop_height=crop_height,
    crop_width=crop_width,
    urn_name_to_urn=dict(
        cropped_hand_annotated_training_data=cropped_hand_annotated_training_data
    ),
    urn_name_to_probability_of_sampling_from_that_urn=dict(
        cropped_hand_annotated_training_data=1.0
    ),
    urn_name_to_augmenter = dict(
        cropped_hand_annotated_training_data=my_augmenter
    )
)

train_dataset = SegmentationDatasetNoRelevance(
    nn_input_height=nn_input_height,
    nn_input_width=nn_input_width,
    urns_with_probs_and_data_augmenters=urns_with_probs_and_data_augmenters,
    mask_name_to_predict="nonfloor"
)

torch_device = my_pytorch_utils.get_the_correct_gpu(substring_of_the_name="8000", which_copy=1)

batch_size = 32

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=4,
                          pin_memory=True,
                          drop_last=True)

val_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True)

# This is just an assertion that we understand what is coming out of train_loader:

for cntr, (xs, ys) in enumerate(train_loader):
    # xs should be a minibatch of 16 color 3 x 224 x 224 training images
    assert xs.size() == torch.Size(
        [batch_size, 3, nn_input_width, nn_input_width]), f"{xs.size()}"
    assert xs.dtype == torch.float32
    # ys are the "labels" or "truth" for the 16 images, expressed as num_joints heatmaps/beliefmaps, each 64x64
    print(xs.device)
    assert ys.size() == torch.Size(
        [batch_size, nn_input_height, nn_input_width])
    assert ys.dtype == torch.int64, f"{ys.dtype}"
    print(f"xs has size {xs.size()}")
    print(f"xs has dtype {xs.dtype}")
    print(f"ys has size {ys.size()}")
    print(f"ys has dtype {ys.dtype}")
    if cntr > 1:
        break

from TernausNet import UNet16

model = UNet16(
    num_classes=2,
    num_filters=32,
    pretrained=False,
    is_deconv=False
)

load_already_trained_model = False
if load_already_trained_model:
    path_to_resume_training_from = Path(f"~/r/trained_models/TernausNet_416x416_2023-01-25_gsw.tar").expanduser()
    dct = torch.load(path_to_resume_training_from)
    model.load_state_dict(dct['model_state_dict'])
    print(f"{Fore.YELLOW}Loaded weights from {path_to_resume_training_from}{Style.RESET_ALL}")


criterion = torch.nn.CrossEntropyLoss()
print(type(criterion))

# +
optimizer = torch.optim.Adam(model.parameters()) #torch.optim.get_optimizer(optimizer_name="adam", model=model)
print(type(optimizer))

LR_STEP = [30, 50]
LR_FACTOR = 0.1
if isinstance(LR_STEP, list):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        LR_STEP,
                                                        LR_FACTOR,
                                                        last_epoch=-1)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   LR_STEP,
                                                   LR_FACTOR,
                                                   last_epoch=-1)
# -

# possibly after loading it full of weights, you got to move the model onto the gpu or else bad things
# about Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
# will happen:
model = model.to(torch_device)  # stuff it onto the gpu

train_some_more = True
if train_some_more:
    num_epochs = 40  # should take 8730 seconds to do 100
    num_datapoints_per_epoch = 1024
    num_batches_per_epoch = num_datapoints_per_epoch // batch_size
    print("Beginning to train")
    start_time = time.time()
    model.train(
    )  # make sure to put the model into train modality for BatchNorms, Dropout, whatever
    for epoch_index in range(num_epochs):
        avg_loss = 0.0
        for batch_index, (xb_cpu, yb_cpu) in enumerate(train_loader):
            if batch_index >= num_batches_per_epoch:
                break
            xb = xb_cpu.to(torch_device)
            # print(f"xb is on device {xb.device}")
            yb = yb_cpu.to(torch_device)
            # print(f"yb is on device {yb.device}")
            # print(type(xb))
            # print(f"yb has dtype {yb.dtype}")
            out = model(
                xb
            )  # throw the batch xb through the neural network to get answers/outputs
            # print(yb.shape)
            # print(yb.dtype)
            loss = criterion(out, yb)
            # print(loss)
            loss.backward()  # backpropagate the gradient of the loss
            optimizer.step()
            optimizer.zero_grad()
            l = loss.detach().cpu()
            avg_loss += l.item()

        if epoch_index % 1 == 0:
            print(f"The loss currently is {avg_loss / num_batches_per_epoch}")
            print(
                f"did {batch_index} batches of {batch_size} to finish epoch {epoch_index}"
            )

    stop_time = time.time()
    print(
        f"It took {stop_time - start_time} seconds to do {num_epochs} epochs")

# Why does saving not preserve the loss progress?!!  Smells bad. Need to save optimizer, lr_scheduler?


save_model = True
if save_model:
    # model_path = Path(f"~/model.pt").expanduser()
    # optimizer_path = Path(f"~/optimizer.pt").expanduser()
    # torch.save(model.state_dict(), model_path)
    # torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(
        {
            'class_name': "UNet1",
            'epoch': 300,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': 0.07128262985497713,
            'avg_false_positives': 33.89,
            'avg_false_negatives': 36.97,
        },
        Path(f"~/r/trained_models/TernausNet_416x416_2023-01-26_gsw_3.tar").expanduser())





