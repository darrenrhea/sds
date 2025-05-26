from cutout_a_nonempty_patch_from_fullsize import (
     cutout_a_nonempty_patch_from_fullsize
)
import numpy as np
from typing import List
import torch
from colorama import Fore, Style
from torch.utils.data import Dataset
from cutout_warped_patches_from_fullsize import (
    cutout_a_warped_patch_from_a_fullsize_image,
    make_is_over_js
)
from print_image_in_iterm2 import print_image_in_iterm2
torch.backends.cudnn.benchmark = True



warp_dataset = WarpDataset(
        channel_stacks: List[np.ndarray], # each datapoint in the list is a stack of rgb and a target_mask and a weight_mask. Later we can stack more target_masks and weight_masks
        train_patches_per_image: int,  # how many patches to cut from each image.
        patch_width: int,  # width of each patch we cut out
        patch_height: int,  # height of each patch we cut out
        normalization_and_chw_transform, # only does, like, Alexnet normalization and channel x height x width-ification i.e. hwc to chw
        output_binarized_masks: bool,
        augment,
        num_mask_channels = 1,  # for regression problems, this should be 1. for classification problems, this should be the number of classes
    )
        assert augment is not None, f"augment function has type {type(augment)}"
        assert num_mask_channels in [1, 2], "Unless you are doing something exotic like 3 class classification or rgba, num_mask_channels should be 1 or 2"

        self.num_mask_channels = num_mask_channels

        self.patch_height = patch_height
        self.patch_width = patch_width

        assert isinstance(channel_stacks, list)

        for channel_stack in channel_stacks:
            assert channel_stack.dtype == np.uint8, "channel_stacks must be a list of numpy arrays of dtype uint8"
            assert channel_stack.ndim == 3, "channel_stacks must be a list of numpy arrays of shape [H, W, C]"
            assert channel_stack.shape[2] == 5, "channel_stacks must be a list of numpy arrays of shape [H, W, C] where C is usually 5 since there is at least one target_mask and at least one weight_mask"

        # the full-size training points:
        self.channel_stacks = channel_stacks

        assert isinstance(output_binarized_masks, bool)
        self.output_binarized_masks = output_binarized_masks

        if output_binarized_masks:
            print(f"{Fore.YELLOW}WARNING: Neural network will be trained on binarized_masks!{Style.RESET_ALL}")
        
        self.train_patches_per_image = train_patches_per_image
       
        self.normalization_and_chw_transform = normalization_and_chw_transform
        self.augment = augment

        self.is_over_js = make_is_over_js(
            patch_height=patch_height,
            patch_width=patch_width,
        )
        self.visualization_counter = 0
        
    def __len__(self):
        train_patches_per_image = self.train_patches_per_image
        return len(self.channel_stacks) * train_patches_per_image

    def __getitem__(self, idx):
        """
        This is called every time a batch is needed,
        so the random augmentations are different each time.
        """
        train_patches_per_image = self.train_patches_per_image
        patch_width = self.patch_width
        patch_height = self.patch_height
        
        fullsize_image_np_u8 = self.channel_stacks[idx // train_patches_per_image]

        # channel_stack_patch = cutout_a_warped_patch_from_a_fullsize_image(
        #     patch_width=self.patch_width,
        #     patch_height=self.patch_height,
        #     fullsize_image_np_u8=fullsize_image_np_u8,
        #     is_over_js=self.is_over_js
        # )
        channel_stack_patch = cutout_a_nonempty_patch_from_fullsize(
            patch_width=self.patch_width,
            patch_height=self.patch_height,
            fullsize_image_np_u8=fullsize_image_np_u8,
            onscreen_channel_index=4,
        )
        assert channel_stack_patch.shape[0] == patch_height
        assert channel_stack_patch.shape[1] == patch_width
        assert channel_stack_patch.shape[2] == 5
        frame_patch = channel_stack_patch[:, :, 0:3]
        mask_patch = channel_stack_patch[:, :, 3]
        weight_patch = channel_stack_patch[:, :, 4]

        # augment should provide further augmentations other than homography warping and scaling and rotating.
        # You may want to visualize the result to make sure you
        # aren't destroying the normal appearance of the image.
       
        transformed = self.augment(image=frame_patch, mask=mask_patch, importance_mask=weight_patch)
        
        aug_frame_patch = transformed['image']
        aug_target_mask_patch = transformed['mask']
        aug_weight_mask_patch = transformed['importance_mask']  # for now has to be called importance_mask
       

        visualize_augmentations = False
        if visualize_augmentations and self.visualization_counter < 100:
            # num_gpus and workers_per_gpu must both be 1 for printing into the iterm2 terminal to work, otherwise too much crosstalk
            print_image_in_iterm2(rgb_np_uint8=aug_frame_patch)
            print_image_in_iterm2(grayscale_np_uint8=aug_target_mask_patch)
            print_image_in_iterm2(grayscale_np_uint8=aug_weight_mask_patch)
            self.visualization_counter += 1
        
        # This is what makes data augmentations hard to visualize: alexnet normalization and channel x height x width-ification:
        frame_patch_torch_f32_cpu = self.normalization_and_chw_transform(aug_frame_patch)

        # to one hot
        # TODO: add support for more classes
        if self.output_binarized_masks:
            assert False, "surprised you are on this code path, we are going hard after regression"
            binarized_mask = aug_target_mask_patch > 127
            assert binarized_mask.max() <= 1, "mask must be binary"
            assert binarized_mask.min() >= 0, "mask must be binary"

            if self.num_mask_channels == 2:  # means mask has shape [2, pH, pW]
                mask = torch.FloatTensor(np.array([binarized_mask == 0, binarized_mask == 1]))
            elif self.num_mask_channels == 1:  # means mask has shape [1, pH, pW]
                mask = torch.FloatTensor(binarized_mask).unsqueeze(0)
            else:
                raise Exception("num_mask_channels must be 1 or 2")
        
        target_mask_patch_torch_f32_cpu = torch.FloatTensor(aug_target_mask_patch).unsqueeze(0) / 255.0
        weight_mask_patch_torch_f32_cpu = torch.FloatTensor(aug_weight_mask_patch).unsqueeze(0) / 255.0

        assert target_mask_patch_torch_f32_cpu.dtype == torch.float32
        assert target_mask_patch_torch_f32_cpu.device == torch.device('cpu')
        assert target_mask_patch_torch_f32_cpu.ndim == 3, "For uniformity, we want to have a channel index in all cases, even if there is only one channel"
        assert target_mask_patch_torch_f32_cpu.shape[0] == self.num_mask_channels
        assert target_mask_patch_torch_f32_cpu.shape[1] == patch_height, f"target_mask_patch_torch_f32_cpu.shape == {target_mask_patch_torch_f32_cpu.shape} but {patch_height=}"
        assert target_mask_patch_torch_f32_cpu.shape[2] == patch_width, f"target_mask_patch_torch_f32_cpu.shape == {target_mask_patch_torch_f32_cpu.shape} but {patch_width=}"

        assert weight_mask_patch_torch_f32_cpu.dtype == torch.float32
        assert weight_mask_patch_torch_f32_cpu.device == torch.device('cpu')
        assert weight_mask_patch_torch_f32_cpu.ndim == 3, "For uniformity, we want to have a channel index in all cases, even if there is only one channel"

        aug_channel_stack_patch_f32_cpu = torch.concatenate(
            [
                frame_patch_torch_f32_cpu,
                target_mask_patch_torch_f32_cpu,
                weight_mask_patch_torch_f32_cpu,
            ],
            axis=0
        )

        assert aug_channel_stack_patch_f32_cpu.dtype == torch.float32
        assert aug_channel_stack_patch_f32_cpu.device == torch.device('cpu')
        assert aug_channel_stack_patch_f32_cpu.shape[0] == 5
        assert aug_channel_stack_patch_f32_cpu.shape[1] == patch_height
        assert aug_channel_stack_patch_f32_cpu.shape[2] == patch_width
        
        return aug_channel_stack_patch_f32_cpu
