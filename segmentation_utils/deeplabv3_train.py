from all_imports_for_image_segmentation import *
from pathlib import Path
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy
from image_openers import open_a_grayscale_png_barfing_if_it_is_not_grayscale
from tqdm import tqdm
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import csv

use_case_name = "brooklyn"  # ultimately we would like this to generalize to many sports.  We have a long way to go on this.
crop_height = 400  # we cut slightly bigger rectangles out so that we can wiggle it to promote translation invariance
crop_width = crop_height  # we crop out squares
desired_mask_names = ["nonfloor", "inbounds"]  # the "relevant" mask tells the loss function whether we care about accuracy for each particular pixel
batch_size = 32

list_of_annotated_images = get_list_of_annotated_images(
    use_case_name=use_case_name,
    must_have_these_masks=desired_mask_names
)


class SegmentationDataset(torch.utils.data.IterableDataset):
    """
    A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(
        self,
        list_of_annotated_images,
        transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        seed: int = None,
        fraction: float = None,
        subset: str = None
    ):
        super().__init__()
        self.list_of_annotated_images = list_of_annotated_images
        self.transforms = transforms
        
        self.mask_transforms = mask_transforms
        

        if not fraction:
            self.image_names = [
                x["image_path"]
                for x in self.list_of_annotated_images
            ]
            self.mask_names = [
                x["mask_name_to_mask_path"]["nonfloor"]
                for x in self.list_of_annotated_images
            ]
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(
                [
                    x["image_path"]
                    for x in self.list_of_annotated_images
                ]
            )
            self.mask_list = np.array(
                [
                    x["mask_name_to_mask_path"]["nonfloor"]
                    for x in self.list_of_annotated_images
                ]
            )
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    #def __len__(self) -> int:
    #    return len(self.image_names)

    def __iter__(self) -> Any:
        return self.generator()
    
    def generator(self):
        nn_input_height = 400
        nn_input_width = 400
        while True:
            index = np.random.randint(0, len(self.image_names))
            image_path = self.image_names[index]
            mask_path = self.mask_names[index]
            image = Image.open(image_path)
            image = image.convert("RGB")
            i0 = np.random.randint(0, 1080 - nn_input_height + 1)
            j0 = np.random.randint(0, 1920 - nn_input_width + 1)
            area = (j0, i0, j0 + nn_input_width, i0 + nn_input_height)  # top left lower right ????
            image = image.crop(area)
            mask_pil = Image.open(mask_path)
            mask_pil = mask_pil.crop(area)
            mask_rgba = np.array(mask_pil)
            mask_a = mask_rgba[:, :, 3]
            final_mask_pil = PIL.Image.fromarray(mask_a)
            sample = {"image": image, "mask": final_mask_pil}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
            if self.mask_transforms:
                sample["mask"] = torch.tensor(np.array(sample["mask"]) > 128, dtype=torch.int64)
            yield sample

my_dataset = SegmentationDataset(
     list_of_annotated_images=list_of_annotated_images,
     transforms=None,
     mask_transforms=None,
     seed=None,
     fraction=None,
     subset=None
)

for index, thing in enumerate(my_dataset):
    print(type(thing))
    print(thing.keys())
    print(type(thing["image"]))
    display(thing["image"])
    display(thing["mask"])
    if index > 2:
        break

# Pick a GPU.  Speed not important right now, so we will not be doing parallel, but we may be training so pick big memory i.e. RTX 8000

torch_device = my_pytorch_utils.get_the_correct_gpu("8000", which_copy=0)

model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
    num_classes=2,
    pretrained=False
)
# model_load_path = Path("~/r/trained_models/deeplabv3.pth").expanduser()
# model.load_state_dict(torch.load(model_load_path))
model.to(torch_device)


# define the torchvision image transforms
transform_to_alexnet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def train_model(
    model,
    criterion,
    dataloaders,
    optimizer,
    metrics,
    bpath,
    num_datapoints_per_epoch,
    num_epochs
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    model.to(torch_device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            cntr = 0
            for sample in tqdm(iter(dataloaders[phase])):
                cntr += 1
                if cntr > num_datapoints_per_epoch:
                    break
                inputs = sample['image'].to(torch_device)
                masks = sample['mask'].to(torch_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# +
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_datasets = {
    phase: SegmentationDataset(
        list_of_annotated_images=list_of_annotated_images,
        transforms=image_transforms,
        mask_transforms="dogge",
        seed=None,
        fraction=None,
        subset=None
    )
    for phase in ['Train', 'Test']
}
# -

dataloaders = {
    phase: DataLoader(
        image_datasets[phase],
        batch_size=batch_size,
        num_workers=8
    )
    for phase in ['Train', 'Test']
}

for pair in dataloaders["Train"]:
    print(type(pair["mask"]))
    print(pair["mask"].dtype)
    print(pair["mask"].size())
    print(type(pair["image"]))
    print(pair["image"].dtype)
    print(pair["image"].size())
    break

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
metrics = {}
bpath = "/home/drhea/logs"
num_epochs = 1
start_time = time.time()
train_model(
    model=model,
    criterion=criterion,
    dataloaders=dataloaders,
    optimizer=optimizer,
    metrics=metrics,
    bpath=bpath,
    num_datapoints_per_epoch=93,
    num_epochs=num_epochs
)
stop_time = time.time()
print(f"Took {stop_time - start_time} to do {num_epochs} epochs")


torch.save(
    model.state_dict(),
    Path("~/r/trained_models/deeplabv3_brooklyn_83f_2classes.pth").expanduser()
)

# put it into evaluation mode for speed:
model.eval()
num_images_to_infer_total = 40
num_images_to_infer_per_batch = 1
# let us bundle a bunch of images for better timing

all_images_preloaded = np.zeros(
    shape=[num_images_to_infer_total, 3, 1080, 1920],
    dtype=np.uint8
)

all_answers = np.zeros(
    shape=[num_images_to_infer_total, 1080, 1920],
    dtype=np.uint8
)

xb_cpu = torch.zeros([num_images_to_infer_per_batch, 3, 1080, 1920], dtype=torch.float32)

image_pils = []
for k in range(num_images_to_infer_total):
    image_pil = Image.open(list_of_annotated_images[k % 90]["image_path"]).convert("RGB")
    image_pils.append(image_pil)
    
    image_hwc_np_uint8 = np.array(image_pil)
    all_images_preloaded[k, :, :, :] = np.transpose(image_hwc_np_uint8, axes=(2, 0, 1))

alexnet_std = torch.tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).to(torch_device)
alexnet_mean = torch.tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).to(torch_device)
print(f"loading finished")

start_time = time.time()
for batch_index in range(num_images_to_infer_total // num_images_to_infer_per_batch):
    xb_cpu[:, :, :, :] = torch.tensor(
        all_images_preloaded[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :, :]
    )
    
    xb = xb_cpu.to(torch_device)
    normalized = (xb.to(torch.float32) / 255.0 - alexnet_mean[..., None, None]) / alexnet_std[..., None, None]
    outputs = model(normalized)
    
    out = outputs['out']
    # all_answers[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :, :] = out.detach().cpu().numpy()[:, :2, :, :]
    all_answers[batch_index*num_images_to_infer_per_batch:(batch_index+1)*num_images_to_infer_per_batch, :, :]  = torch.argmax(out, dim=1).to(torch.uint8).detach().cpu().numpy()
    
stop_time = time.time()
print(f"{stop_time  - start_time} seconds to do {num_images_to_infer_total}")

for k, image_pil in enumerate(image_pils):
    PIL.Image.fromarray(255*all_answers[k, :, :]).save(
        Path(f"~/r/segmentation_utils/temp/{k}.jpg").expanduser()
    )
