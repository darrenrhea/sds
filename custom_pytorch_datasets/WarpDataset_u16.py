from cutout_a_patch_from_fullsize_u16 import (
     cutout_a_patch_from_fullsize_u16
)
import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset
from cutout_warped_patches_from_fullsize import (
    make_is_over_js
)


class WarpDataset_u16(Dataset):
    """
    Returns a cpu torch Tensor of shape (3 + num_other_channels) x patch_height x patch_width and dtype float32.
   
    For greatest generality, the target masks are assumed to range from 0 to 65535 in value.

    This should take in a Python list of np.uint16 channel_stacks

    This should only be used for training, not for inference, as there is random selection of patches.

    It is going to cut patches at random, albeit in a guided way.
    
    It does not resize the frames or masks, as we need it to work
    
    when the training frames and masks have various sizes, so long as they are all at least as big as the patch size.
    
    It should not be overly self-aware of what it is being used for.
    """
    def __init__(
        self,
        num_channels: int,
        channel_stacks: List[np.ndarray],  # each datapoint in the list is a hwc channel_stack of num_channels
        train_patches_per_image: int,  # how many patches to cut from each image.
        patch_width: int,  # width of each patch we cut out
        patch_height: int,  # height of each patch we cut out
    ):
        self.num_channels = num_channels
        self.patch_height = patch_height
        self.patch_width = patch_width

        assert isinstance(channel_stacks, list)

        for channel_stack in channel_stacks:
            assert channel_stack.dtype == np.uint16, "channel_stacks must be a list of numpy arrays of dtype uint16"
            assert channel_stack.ndim == 3, "channel_stacks must be a list of numpy arrays of shape [H, W, C]"
            assert channel_stack.shape[2] == 5, "channel_stacks must be a list of numpy arrays of shape [H, W, C] where C is usually 5 since there is at least one target_mask and at least one weight_mask"

        self.channel_stacks = channel_stacks

        self.train_patches_per_image = train_patches_per_image
       
        self.is_over_js = make_is_over_js(
            patch_height=patch_height,
            patch_width=patch_width,
        )
        
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
        
        fullsize_image_np_u16 = self.channel_stacks[idx // train_patches_per_image]

        # cut a patch of all the channels at the same location:
        channel_stack_patch_hwc_np_u16 = cutout_a_patch_from_fullsize_u16(
            patch_width=self.patch_width,
            patch_height=self.patch_height,
            fullsize_image_np_u16=fullsize_image_np_u16
        )

        # If you want warped cutouts, use this:
        # channel_stack_patch = cutout_a_warped_patch_from_a_fullsize_u16(
        #     patch_width=self.patch_width,
        #     patch_height=self.patch_height,
        #     fullsize_image_np_u16=fullsize_image_np_u16,
        #     is_over_js=self.is_over_js
        # )

       

        # if you are worried it is "nonempty":
        # channel_stack_patch = cutout_a_nonempty_patch_from_fullsize(
        #     patch_width=self.patch_width,
        #     patch_height=self.patch_height,
        #     fullsize_image_np_u8=fullsize_image_np_u8,
        #     onscreen_channel_index=4,
        # )



        # Currently torch cannot handle np.uint16, so we need to convert to float32
        channel_stack_patch_chw_torch_f32_cpu = torch.from_numpy(
            channel_stack_patch_hwc_np_u16.astype(np.float32)
        ).permute(2, 0, 1) / 65535.0

       
        # Get it to float32 and normalize all channels to [0, 1]
       
        assert channel_stack_patch_chw_torch_f32_cpu.dtype == torch.float32
        assert channel_stack_patch_chw_torch_f32_cpu.device == torch.device('cpu')
        assert channel_stack_patch_chw_torch_f32_cpu.shape[0] == self.num_channels
        assert channel_stack_patch_chw_torch_f32_cpu.shape[1] == patch_height
        assert channel_stack_patch_chw_torch_f32_cpu.shape[2] == patch_width
        
        return channel_stack_patch_chw_torch_f32_cpu
