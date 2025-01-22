# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Environment (conda_floor_not_floor)
#     language: python
#     name: conda_floor_not_floor
# ---

# https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705
#
# https://www.kaggle.com/cordmaur/38-cloud-simple-unet/notebook
#
# typical `reload_ext` boilerplate so it takes up changes made to modules as best it can.
# We want jupyter notebook cells to be as wide as possible.  

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))

# We got tired of looking at all the imports:

from all_imports_for_image_segmentation import *

use_case_name = "nba"  # ultimately we would like this to generalize to many sports.  We have a long way to go on this.
crop_height = 256  # we cut slightly bigger rectangles out so that we can wiggle it to promote translation invariance
crop_width = crop_height  # we crop out squares
nn_input_width = 224  # after possible data-augementation, the 256x256 is cropped to 224x224
nn_input_height = nn_input_width  # force the neural network input be a 224x224 square for now
desired_mask_names = ["nonfloor", "relevant"]  # the "relevant" mask tells the loss function whether we care about accuracy for each particular pixel

list_of_annotated_images = get_list_of_annotated_images(
    use_case_name=use_case_name,
    must_have_these_masks=desired_mask_names
)

pp.pprint(list_of_annotated_images)

# This next cell is *just a demo* of what `cut_this_many_interesting_subrectangles_from_annotated_image` does. It is not necessary for the remainder of the notebook so you can skip it once you know what cut_this_many_interesting_subrectangles_from_annotated_image does.

# +
dct = cut_this_many_interesting_subrectangles_from_annotated_image(
    annotated_image=list_of_annotated_images[30],  # which annotated_image to cut croppings from
    how_many_croppings_to_cut_out=50,
    crop_height=crop_height,
    crop_width=crop_width,
    desired_mask_names=desired_mask_names,
    mask_name_to_min_amount=dict(relevant=1),  # we are forcing croppings to have at least one pixel of mainrectangle mask on
    mask_name_to_max_amount=dict(),
    mask_encoding_type="alpha_channel_of_rgba",
)
assert dct["success"]

for k in range(dct["num_croppings"]):
    print("original:")
    display(
        PIL.Image.fromarray(
            dct["cropped_originals"][k]
        )
    )
    for mask_name in desired_mask_names:
        print(f"{mask_name}:")
        display(
        PIL.Image.fromarray(
            255 * dct["mask_name_to_cropped_masks"][mask_name][k]
        )
    )
# -

# We want to make so-called **interesting** croppings that meet certain requirements.
# For instance, if a cropping contains absolutely none of the main floor,
# which is the region where we care about correct classification,
# e.g. if the cropping is way out in the crowd somewhere, it is not interesting to us.
# We make croppings that are "slightly too big" at 256x256 so that we can
# gaussian blur or change the color somewhat at size 256x256 then crop out a 224x224
# at various translations as a form of data augmentation.

# +
cropped_hand_annotated_training_data = get_numpy_arrays_of_croppings_and_their_masks(
    list_of_annotated_images=list_of_annotated_images,
    crop_height=crop_height,
    crop_width=crop_width,
    desired_mask_names=desired_mask_names,
    mask_name_to_min_amount=dict(relevant=1),
    mask_name_to_max_amount=dict(),
    how_many_originals=len(list_of_annotated_images),
    how_many_crops_per_original=10000,
    mask_encoding_type="alpha_channel_of_rgba"
)

dct = cropped_hand_annotated_training_data
print([k for k in dct.keys()])
assert dct["num_croppings_cut"] > 0
assert isinstance(dct["mask_name_to_cropped_masks"], dict)
assert isinstance(dct["cropped_originals"], np.ndarray)
for mask_name in desired_mask_names:
    print(f"We have the masks known as {mask_name}")
    assert dct["mask_name_to_cropped_masks"][mask_name].shape[0] == dct["cropped_originals"].shape[0]
    
# -

# * a np.array of orginal color croppings
# * another np.array of corresponding player/foreground-object masks.
# * another np.array of relevance masks.  Off of it, we do not impose any loss for getting the answer wrong.

# +

new_cropped_originals = cropped_hand_annotated_training_data["cropped_originals"]
new_cropped_masks = cropped_hand_annotated_training_data["mask_name_to_cropped_masks"]["nonfloor"]
new_relevance_masks = cropped_hand_annotated_training_data["mask_name_to_cropped_masks"]["relevant"]
# -

print(new_cropped_originals.shape)
print(new_cropped_masks.shape)
print(new_relevance_masks.shape)

# If you want to you can look at them, or at augmentations of it:

#todo: make this red blue black white green
from PIL import ImageFilter
look_at_training_data = True
if look_at_training_data:
    for k in range(new_cropped_masks.shape[0]):
        if np.random.rand() > 4 / new_cropped_masks.shape[0]:
            continue
        print(f"We randomly selected to show you the {k}-ith image:")
        x_pil = PIL.Image.fromarray(new_cropped_originals[k])
        blur_filter = PIL.ImageFilter.GaussianBlur(radius=0.7)
        augmented_pil = x_pil.filter(filter=blur_filter)
        display(x_pil)
        #display(augmented_pil)
        display(PIL.Image.fromarray(new_cropped_masks[k] * 255))
        display(PIL.Image.fromarray(new_relevance_masks[k] * 255))

len(new_cropped_originals), len(new_cropped_masks), len(new_relevance_masks)


def my_augmenter(x):
    """
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

#todo: make this red blue black white green
look_at_augmentation = True
if look_at_augmentation:
    for k in range(new_cropped_masks.shape[0]):
        if np.random.rand() > 40 / new_cropped_masks.shape[0]:
            continue
        print(f"We randomly selected to show you the {k}-ith image:")
        x_pil = PIL.Image.fromarray(new_cropped_originals[k])
        
        augmented_np = my_augmenter(new_cropped_originals[k])
        augmented_pil = PIL.Image.fromarray(augmented_np)
        display(x_pil)
        display(augmented_pil)


# +
from SegmentationDataset import SegmentationDataset
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

train_dataset = SegmentationDataset(
    nn_input_height=nn_input_height,
    nn_input_width=nn_input_width,
    urns_with_probs_and_data_augmenters=urns_with_probs_and_data_augmenters,
    mask_name_to_predict="nonfloor",
    relevance_mask_name="relevant"
)
# -

# We now include a third item in the triplet, z, which states how relevant it is to get the answer correct.
# Off of the relevance mask, the loss function does not care about accuracy at all -- for instance in the crowd, or on color floor logos

visualize_train_dataset = True
if visualize_train_dataset:
    for cntr, (x, y, z) in enumerate(train_dataset):
        if np.random.rand() > 0.01:
            continue
        print(x.shape)  # color image
        print(y.shape)  # label, i.e. i.e. for each pixel is it foreground
        print(z.shape)  # relevance of each pixel
        print(cntr)
        if cntr >= 2000:
            break

        display_numpy_chw_rgb_image(chw=np.array(x))
        print(train_dataset.mask_name_to_predict)
        display(PIL.Image.fromarray(255*np.array(y).astype(np.uint8)))
        print(train_dataset.relevance_mask_name)
        display(PIL.Image.fromarray(255*np.array(z).astype(np.uint8)))

torch_device = my_pytorch_utils.get_the_correct_gpu("Quadro", which_copy=1)

batch_size = 64

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

for cntr, (xs, ys, zs) in enumerate(train_loader):
    # xs should be a minibatch of 16 color 3 x 224 x 224 training images
    assert xs.size() == torch.Size(
        [batch_size, 3, nn_input_width, nn_input_width]), f"{xs.size()}"
    assert xs.dtype == torch.float32
    # ys are the "labels" or "truth" for the 16 images, expressed as num_joints heatmaps/beliefmaps, each 64x64
    print(xs.device)
    assert ys.size() == torch.Size(
        [batch_size, nn_input_height, nn_input_width])
    assert zs.size() == torch.Size(
        [batch_size, nn_input_height, nn_input_width])
    assert ys.dtype == torch.int64, f"{ys.dtype}"
    assert zs.dtype == torch.int64, f"{zs.dtype}"
    print(f"xs has size {xs.size()}")
    print(f"ys has size {ys.size()}")
    print(f"ys has size {zs.size()}")
    if cntr > 1:
        break

# We are going to wire the neural network together and then rapidly check
# (via assertions) that the shapes of tensors within it are what we think we are doing:

from UNet1 import UNet1
# make an instance of the neural network
model = UNet1(
    nn_input_height=nn_input_height,
    nn_input_width=nn_input_width,
    in_channels=3,
    out_channels=2  # for binary you want 2 out_channels
)

# Throw a dummy input batch through the model while it is on the cpu side.
# Notice you get out batch_size=64 answers since you threw a batch of 64 224x224 images in.
# You also get 2 channels out, namely floorness and foregroundness.

dummy_input = torch.randn(batch_size, 3, nn_input_height, nn_input_width, device="cpu")  # 3 for rgb
assert dummy_input.dtype == torch.float32
dummy_output = model(dummy_input)
dummy_output.shape

load_already_trained_model = True
if load_already_trained_model:
    path_to_resume_training_from = Path(f"~/basketball_nonfloor_UNet1_2020_09_15_trained_on_17.tar").expanduser()
    dct = torch.load(path_to_resume_training_from)
    model.load_state_dict(dct['model_state_dict'])
    # optimizer.load_state_dict(dct['optimizer_state_dict'])
    #lr_scheduler.load_state_dict(dct['scheduler_state_dict'])
    dct['epoch'], dct["loss"]
    print("Loaded")

# The output out of the neural network is a torch.float32 [bs, 2, height, width]., i.e. [64, 2, 224, 224]
#
# The target mask yb is type `torch.int64` size [bs, height, width] `[64, 224, 224]`
#
# cross_entropy_loss = torch.nn.CrossEntropyLoss()
#
# we need cross_entropy_loss(out, yb)
#
# to be the same as relevant_cross_entropy_loss(out, yb, zb) when the relevance_mask zb is all on.
#
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#crossentropyloss
#
# https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cross_entropy
#
# They have this ignore index meaning don't add the the gradient calculation
# ignore_index: int = -100

# instead of the word "loss", they say criterion for some reason. So do the PyTorch docs.
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
print(type(criterion))

# +
optimizer = utils.get_optimizer(optimizer_name="adam", model=model)
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

train_some_more = False
if train_some_more:
    num_epochs = 50 * 6
    num_datapoints_per_epoch = 1024
    num_batches_per_epoch = num_datapoints_per_epoch // batch_size
    print("Beginning to train")
    start_time = time.time()
    model.train(
    )  # make sure to put the model into train modality for BatchNorms, Dropout, whatever
    for epoch_index in range(num_epochs):
        avg_loss = 0.0
        for batch_index, (xb_cpu, yb_cpu, zb_cpu) in enumerate(train_loader):
            if batch_index >= num_batches_per_epoch:
                break
            xb = xb_cpu.to(torch_device)
            # print(f"xb is on device {xb.device}")
            yb = yb_cpu.to(torch_device)
            zb = zb_cpu.to(torch_device)
            # print(f"yb is on device {yb.device}")
            # print(type(xb))
            # print(f"yb has dtype {yb.dtype}")
            out = model(
                xb
            )  # throw the batch xb through the neural network to get answers/outputs
            # print(yb.shape)
            # print(yb.dtype)
            loss = criterion(out, zb*yb - 100*(1-zb)*yb) # when zb = 1, yb.  When zb=0, -100, the ignore_index
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

50 / 323.4 * 3600 * 1.0

visualize_train_loader = True
display_in_jupyter = True
save_predictions_to_disk = False
threshold = 0.5  # where to cut the "probability" / decision function per pixel
if visualize_train_loader:
    print("cyan is FALSE NEGATIVE")
    print("red is FALSE POSITIVE")
    print("black is TRUE NEGATIVE")
    print("white is TRUE POSITIVE")
    print("dark green is IRRELEVANT")
    
    

    start_time = time.time()
    model.eval(
    )  # make sure to put the model into train modality for BatchNorms, Dropout, whatever
    cntr = 0
    fn_lst = []
    fp_lst = []
    for batch_index, (xb_cpu, yb_cpu, zb_cpu) in enumerate(train_loader):
        xb = xb_cpu.to(torch_device)
        # print(f"xb is on device {xb.device}")
        yb = yb_cpu.to(torch_device)
        # print(f"yb is on device {yb.device}")
        # print(type(xb))
        # print(f"yb has dtype {yb.dtype}")
        out = model(
            xb
        )  # throw the batch xb through the neural network to get answers/outputs
        out_cpu_torch = out.detach().cpu()
        log_probs_torch = F.log_softmax(
            out_cpu_torch, dim=1
        )  # let's get to class probabilities. The 1-ith index is which class.
        log_probs_np = log_probs_torch.numpy()
        probs_np = np.exp(log_probs_np)
        # print(f"prob_np has shape {probs_np.shape} and max {np.max(probs_np)}")
        
        total_false_positives = np.zeros((nn_input_width, nn_input_height),
                                         dtype=np.int32)
        total_false_negatives = np.zeros((nn_input_width, nn_input_height),
                                         dtype=np.int32)
        for k in range(batch_size):
            chw = np.array(xb_cpu[k, :, :, :])

            mask_should_be = (np.array(yb_cpu[k, :, :]) == 1).astype(np.uint8)
            relevant = np.array(zb_cpu[k, :, :]) == 1
            binary_prediction = (probs_np[k, 1, :, :] > threshold).astype(
                np.uint8)
            error_type_image = np.zeros((nn_input_height, nn_input_width, 3),
                                        dtype=np.uint8)
            # red is false positive
            # cyan is false negative
            # green is irrelevant
            false_positives = np.logical_and(
                np.logical_and(
                    mask_should_be == 0,
                    binary_prediction == 1
                ),
                relevant == 1
                
            )
            total_false_positives += false_positives
            num_false_positives = np.sum(false_positives)
            false_negatives = np.logical_and(
                np.logical_and(
                    mask_should_be == 1,
                    binary_prediction == 0
                ),
                relevant == 1
            )
            num_false_negatives = np.sum(false_negatives)
            total_false_negatives += false_negatives
            fp_lst.append(num_false_positives)
            fn_lst.append(num_false_negatives)

            error_type_image[:, :, 0] = binary_prediction * 255
            for c in range(1, 3):
                error_type_image[:, :, c] = mask_should_be * 255
            error_type_image[np.logical_not(relevant), :] = [0, 50, 0]
            color = numpy_chw_rgb_to_uint8_np(chw)
            
            color_pil_image = PIL.Image.fromarray(color)
            error_type_pil_image = PIL.Image.fromarray(error_type_image)
            if save_predictions_to_disk:
                color_pil_image.save(
                    f"/home/drhea/example_predictions/{cntr}_color.png")
                error_type_pil_image.save(
                    f"/home/drhea/example_predictions/{cntr}_mask.png")
            if display_in_jupyter:
                # display_numpy_hwc_rgb_image(color, dilate_by=2)
                display(color_pil_image)
                display(error_type_pil_image)
                #display_numpy_hwc_rgb_image(error_type_image, dilate_by=2)
                print(f"num_false_positives = {num_false_positives}")
                print(f"num_false_negatives = {num_false_negatives}\n")
            cntr += 1
        if batch_index >= 10:
            break

    print(f"avg_false_positives = {np.mean(fn_lst)}")
    print(f"avg_false_negatives = {np.mean(fp_lst)}\n")
    stop_time = time.time()
    print(f"Took {stop_time - start_time} seconds to do {cntr} images")
    plt.figure(figsize=(8, 8))
    plt.imshow(total_false_positives / cntr, alpha=0.7, cmap="jet")
    plt.figure(figsize=(8, 8))
    plt.imshow(total_false_negatives / cntr, alpha=0.7, cmap="jet")
    plt.colorbar()

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
        Path(f"~/r/trained_models/basketball_nonfloor_UNet1_2020_12_10_trained_on_31_relevant.tar").expanduser())

# +
real = True
if real:
    frame_index = np.random.randint(0, 3595)
    clip_index = np.random.randint(1, 2+1)
    color_original_pngs_dir = Path(
        f"~/awecom/data/clips/lahouyadif{clip_index}/frames").expanduser()
    img_file_name = color_original_pngs_dir / f"lahouyadif{clip_index}_{frame_index:06d}.jpg"
    print(img_file_name)
else:  # try it on a synthetic image
    color_original_pngs_dir = Path(f"~/synthetic_soccer/").expanduser()
    img_file_name = color_original_pngs_dir / f"9182642798283701210_everything.png"

assert os.path.isfile(img_file_name), f"{img_file_name} does not exist!"
img_pil = PIL.Image.open(img_file_name).convert("RGB")
img_hwc_np_uint8 = np.array(img_pil)
pred_for_chunk(
    torch_device=torch_device,
    model=model,
    threshold=0.5,
    img_hwc_np_uint8=img_hwc_np_uint8,
    left=1050,
    upper=700,
    chunk_w=224,
    chunk_h=224,
    show_plots=True
)

None
# -

from stride_score_image import stride_score_image

training_indices = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1110, 1200, 1300, 1400, 1500, 1700, 1600, 1798]
validation_indices = []
total_error = 0.0
num_frames = 0
for frame_index in training_indices:
    clip_name = "stapleslahou1"
    pngs_dir = Path(f"~/r/{clip_name}/").expanduser()
    color_image_path = pngs_dir / f"{clip_name}_{frame_index:06d}_color.png"
    true_image_path = pngs_dir / f"{clip_name}_{frame_index:06d}_nonfloor.png"
    relevance_image_path = pngs_dir / f"{clip_name}_{frame_index:06d}_relevant.png"
    for image_path in [color_image_path, true_image_path, relevance_image_path]:
        assert image_path.is_file(), f"{image_path} does not exist!"

    angle_in_degrees = 0
    original_image_pil = PIL.Image.open(str(color_image_path)).convert("RGB")
    image_pil = original_image_pil.rotate(angle=angle_in_degrees, expand=1, resample=PIL.Image.BICUBIC)
    
    original_relevance_pil = PIL.Image.open(relevance_image_path)
    relevance_pil = original_relevance_pil.rotate(angle=angle_in_degrees, expand=1, resample=PIL.Image.BICUBIC)
    
    original_truth_pil = PIL.Image.open(true_image_path)
    truth_pil = original_truth_pil.rotate(angle=angle_in_degrees, expand=1, resample=PIL.Image.BICUBIC)
    
    relevance_mask = (np.array(relevance_pil)[:,:,3] >= 128).astype(np.uint8)
    truth_mask = (np.array(truth_pil)[:,:,3] >= 128).astype(np.uint8)
    hwc_np_uint8 = np.array(image_pil)
    
    print(f"{color_image_path}")
    prediction_mask = stride_score_image(
        hwc_np_uint8=hwc_np_uint8,
        torch_device=torch_device,
        model=model,
        threshold=0.5,  # 0.75 causes way to many false negatives
        stride=223,
        batch_size=64
    )

    #display_numpy_hw_grayscale_image(prediction_mask)

    from image_displayers_for_jupyter import *
    dct = display_segmentation_against_truth_and_relevance(
        prediction_mask=prediction_mask,
        relevance_mask=relevance_mask,
        truth_mask=truth_mask,
        display_in_jupyter=True,
        save_path=f"~/validation_set2/{clip_name}_{frame_index:06d}_prediction.png"
    )
    total_error += dct["error"]
    num_frames += 1
print(total_error / num_frames)

# +
from stride_score_image import stride_score_image
threshold = 0.50
input_extension = "jpg" # bmp is faster to read and write, but huge
video_name = "lahouyadif1"
masking_attempt_id = "anna_12_10"
save_color_information_into_masks = True
height = 1080
width = 1920
# video_name = "barcelona_vs_celtic_at_aviva"
# video_name = "barcelona_vs_atleticomadrid_c000"  # goes to 000521

masking_attempts_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts").expanduser()
masking_attempts_dir.mkdir(exist_ok=True)
out_dir = Path(f"~/awecom/data/clips/{video_name}/masking_attempts/{masking_attempt_id}").expanduser()
out_dir.mkdir(exist_ok=True)
start_time = time.time()
for frame_index in range(0, 3597 + 1):
    color_original_pngs_dir = Path(
        f"~/awecom/data/clips/{video_name}/frames").expanduser()
    image_path = color_original_pngs_dir / f"{video_name}_{frame_index:06d}.{input_extension}"
    assert image_path.is_file(), f"{image_path} does not exist!"
    img_pil = PIL.Image.open(str(image_path)).convert("RGB")
    hwc_np_uint8 = np.array(img_pil)
    
    binary_prediction = stride_score_image(
        hwc_np_uint8=hwc_np_uint8,
        torch_device=torch_device,
        model=model,
        threshold=threshold,
        stride=30,
        batch_size=64
    )
    stop_time = time.time()
    print(f"Took {stop_time - start_time} to score {frame_index + 1} images:\n{image_path}")
    print(f"Took {(stop_time - start_time) / (frame_index +1)} seconds per image, or {(frame_index +1) / (stop_time - start_time)} images per second")

    score_image = PIL.Image.fromarray(
        np.clip(binary_prediction * 255.0, 0, 255).astype(np.uint8))

    # display(img_pil)
    # display(score_image)
    if (save_color_information_into_masks):
        out_hwc_rgba_uint8 = np.zeros(shape=(height, width, 4), dtype=np.uint8)
        out_hwc_rgba_uint8[:, :, :3] = hwc_np_uint8
        out_hwc_rgba_uint8[:, :, 3] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hwc_rgba_uint8)
        # save it to file:

        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
        out_pil.save(out_file_name, "PNG")
    else:
        out_hw_grayscale_uint8 = np.zeros(shape=(height, width), dtype=np.uint8)
        out_hw_grayscale_uint8[:, :] = binary_prediction * 255
        out_pil = PIL.Image.fromarray(out_hw_grayscale_uint8)
        # save it to file:
        out_file_name = out_dir / f"{video_name}_{frame_index:06d}_nonfloor.png"
        out_pil.save(out_file_name, "PNG")
        
    print(f"See {out_file_name}")

if False:
    plt.figure(figsize=(16, 9))
    plt.imshow(img_pil.convert('L'))
    plt.imshow(score_image, alpha=0.7, cmap="jet")
    plt.figure(figsize=(16, 9))
    plt.imshow(img_pil)
    display(PIL.Image.fromarray(binary_prediction * 255))
    display(
        PIL.Image.fromarray(binary_prediction[:, :, np.newaxis] *
                            img_hwc_np_uint8))
    display(
        PIL.Image.fromarray(
            np.logical_not(binary_prediction)[:, :, np.newaxis] *
            img_hwc_np_uint8))
# -




