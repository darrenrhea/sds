"""
Returns pairs (x, y)
where x is a torch.FloatTensor of shape 3 x patch_height x patch_width
and y is a torch.FloatTensor of shape 1 x patch_height x patch_width

batch_index x channel_index x h x  w

For greatest generality, the target masks are assumed to range from 0 to 255 in value.
If you want to do hard binary, code the masks as taking on values as 0 and 255.

This should take in a Python list of numpy frames and masks,
say that were loaded in parallel by load_images_and_masks_in_parallel.
This should only be used for training, not for inference.
It is going to cut patches at random, albeit in a guided way.
It does not resize the frames or masks, as we need it to work
when the training frames and masks have various sizes.
It should not be overly self-aware of what it is being used for.
"""
import numpy as np
from typing import List

import torch
from colorama import Fore, Style
from torch.utils.data import Dataset
torch.backends.cudnn.benchmark = True


class FloorDataset(Dataset):
    def __init__(
        self,
        input_frames: List[np.ndarray],  # full sized numpy uint8 H x W x C frames in rgb, possibly of various sizes
        target_masks: List[np.ndarray],  # full sized numpy uint8 H x W masks in grayscale range from 0 to 255, 0 means background, 255 means foreground
        train_patches_per_image: int,  # how many patches to cut from each image.
        patch_width: int,  # width of each patch we cut out
        patch_height: int,  # height of each patch we cut out
        transform,  
        output_binarized_masks: bool,
        augment = None,
        deterministic = False,
        num_mask_channels = 1,  # for regression problems, this should be 1. for classification problems, this should be the number of classes
    ):
        self.num_mask_channels = num_mask_channels
        for input_frame, target_mask in zip(input_frames, target_masks):
            assert input_frame.dtype == np.uint8, "input_frames must be a list of numpy arrays of dtype uint8"
            assert input_frame.ndim == 3, "input_frames must be a list of numpy arrays of shape [H, W, C] where C is usually 3"
            assert input_frame.shape[2] == 3, "input_frames must be a list of numpy arrays of shape [H, W, C] where C is usually 3"
            assert target_mask.ndim == 2, "target_masks must be a list of numpy arrays of shape [H, W]"
            assert target_mask.dtype == np.uint8, "target_masks must be a list of numpy arrays of dtype uint8"
            assert input_frame.shape[0] == target_mask.shape[0], "each input_frame must have the same height as it s corresponding target_mask"
            assert input_frame.shape[1] == target_mask.shape[1], "each input_frame must have the same width as it s corresponding target_mask"
        
        self.output_binarized_masks = output_binarized_masks
        if output_binarized_masks:
            print(f"{Fore.YELLOW}WARNING: Neural network will be trained on binarized_masks!{Style.RESET_ALL}")
        self.input_frames = input_frames
        self.target_masks = target_masks
        self.MAX_PATCH_SEARCH_ITERATIONS = 50

        self.train_patches_per_image = train_patches_per_image
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.transform = transform
        self.thres_active = False
        self.augment = augment
        self.deterministic = deterministic
        

        self.UNSURE_PATCHES = False
        self.UNSURE_MIN = 0.1
        self.UNSURE_MAX = 0.9
        self.UNSURE_MIN_FRACTION = 0.01

        
        for frame, mask in zip(self.input_frames, self.target_masks):
            assert frame.shape[1] >= patch_width, "Frame too small"
            assert frame.shape[0] >= patch_height, "Frame too small"
            assert frame.shape[0] == mask.shape[0], "Frame and mask must have same height"
            assert frame.shape[1] == mask.shape[1], "Frame and mask must have same width"
        

        if deterministic:
            # make patch locs
            patch_locs = []
            ref = self.target_masks[0]
            for _ in range(len(self.input_frames)):
                patch_locs_frame = []
                for _ in range(train_patches_per_image):
                    if ref.shape[0] > patch_height:
                        y0 = np.random.randint(ref.shape[0] - patch_height)
                    else:
                        y0 = 0

                    if ref.shape[1] > patch_width:
                        x0 = np.random.randint(ref.shape[1] - patch_width)
                    else:
                        x0 = 0
                    patch_locs_frame.append([x0, y0])
                patch_locs.append(patch_locs_frame)
            self.patch_locs = patch_locs


    def __len__(self):
        train_patches_per_image = self.train_patches_per_image
        return len(self.input_frames) * train_patches_per_image

    def __getitem__(self, idx):
        # MAX_LABEL_FRACTION = self.MAX_LABEL_FRACTION
        # UNSURE_MIN_FRACTION = self.UNSURE_MIN_FRACTION
        MAX_PATCH_SEARCH_ITERATIONS = self.MAX_PATCH_SEARCH_ITERATIONS
        train_patches_per_image = self.train_patches_per_image
        patch_width = self.patch_width
        patch_height = self.patch_height
        
        frame = self.input_frames[idx // train_patches_per_image]
        mask = self.target_masks[idx // train_patches_per_image]

        if self.augment:
            transformed = self.augment(image=frame, mask=mask)
            frame = transformed['image']
            mask = transformed['mask']

        # this is the alexnet normalization and chw ification:
        if self.transform:
            frame = self.transform(frame)
            #transformed = self.transform(image=frame, mask=mask)
            #frame = transformed['image']
            #mask = transformed['mask']

        # to one hot
        # TODO: add support for more classes
        if self.output_binarized_masks:
            binarized_mask = mask > 127
            assert binarized_mask.max() <= 1, "mask must be binary"
            assert binarized_mask.min() >= 0, "mask must be binary"

            if self.num_mask_channels == 2:  # means mask has shape [2, H, W]
                mask = torch.FloatTensor(np.array([binarized_mask == 0, binarized_mask == 1]))
            elif self.num_mask_channels == 1:  # means mask has shape [1, H, W]
                mask = torch.FloatTensor(binarized_mask).unsqueeze(0)
            else:
                raise Exception("num_mask_channels must be 1 or 2")
        else:
            mask = torch.FloatTensor(mask).unsqueeze(0) / 255.0

        # get a patch

        if self.deterministic:
            x0, y0 = self.patch_locs[idx // train_patches_per_image][idx % train_patches_per_image]
            patch_img = frame[:, y0:y0 + patch_height, x0:x0 + patch_width]
            patch_mask = mask[:, y0:y0 + patch_height, x0:x0 + patch_width]
            return patch_img, patch_mask

        # random patch.
        # Used to be rejection of patches that weren't good enough.  We could put that back.
        if frame.shape[1] > patch_height:
            y0 = np.random.randint(frame.shape[1] - patch_height)
        else:
            y0 = 0

        if frame.shape[2] > patch_width:
            x0 = np.random.randint(frame.shape[2] - patch_width)
        else:
            x0 = 0

        patch_img = frame[:, y0:y0 + patch_height, x0:x0 + patch_width]
        patch_mask = mask[:, y0:y0 + patch_height, x0:x0 + patch_width]

        assert patch_img.dtype == torch.float32
        assert patch_img.device == torch.device('cpu')
        assert patch_img.shape[0] == 3
        assert patch_img.shape[1] == patch_height
        assert patch_img.shape[2] == patch_width
        
        assert patch_mask.dtype == torch.float32
        assert patch_mask.device == torch.device('cpu')
        assert patch_mask.ndim == 3, "For uniformity, we want to have a channel index in all cases, even if there is only one channel"
        assert patch_mask.shape[0] == self.num_mask_channels
        assert patch_mask.shape[1] == patch_height
        assert patch_mask.shape[2] == patch_width


        return patch_img, patch_mask
