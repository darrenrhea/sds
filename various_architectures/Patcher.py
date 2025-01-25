
import torch
import numpy as np
from typing import Union


class Patcher(object):
    """
    This should only be used for inference, not training.
    Make sure Patcher is not used nor mentioned in any training code.
    """
    def __init__(
        self,
        frame_width,
        frame_height,
        patch_width,
        patch_height,
        stride_width = 0,
        stride_height = 0,
        pad_width = 0,
        pad_height = 0,
        boost_center = 0
    ):
        """
        This is a class for (over-)covering a frame
        of size frame_width x frame_height
        with many patches of size patch_width x patch_height,
        and later stitching/voting answers concerning them
        them back together.
        """
        self.frame_width = frame_width + pad_width
        self.frame_height = frame_height + pad_height
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.stride_width = stride_width or patch_width
        self.stride_height = stride_height or patch_height
        self.pad_width = pad_width
        self.pad_height = pad_height
        self.is_strided = self.stride_width != self.patch_width or self.stride_height != self.patch_height
        self.boost_center = boost_center
        self.make_coords()

    def make_coords(self):
        # x steps
        xs = []
        # print(f"{self.frame_width=}, {self.stride_width=}")
        for x in range(0, self.frame_width, self.stride_width):
            if x + self.patch_width < self.frame_width:
                xs.append(x)
            else:
                # shift last column left for full frame patch
                xs.append(self.frame_width - self.patch_width)
                break
        self.coords_x = xs
        # print(f"{xs=}")

        # y steps
        ys = []
        for y in range(0, self.frame_height, self.stride_height):
            if y + self.patch_height < self.frame_height:
                ys.append(y)
            else:
                # shift last row up for full frame patch
                ys.append(self.frame_height - self.patch_height)
                break
        self.coords_y = ys

        self.num_x = len(xs)
        self.num_y = len(ys)
        self.num = self.num_x * self.num_y

        # indexes
        ixs = range(len(xs))
        iys = range(len(ys))
        # self.coords_xy are the top-left corners of each patch in stack-of-patches order:
        self.coords_xy = [(x, y) for y in ys for x in xs]
        # print(f"{self.coords_xy=}")
        self.idxs_xy = [(ix, iy) for iy in iys for ix in ixs]

        # boundaries
        boundaries = []
        bd_left = False
        bd_right = False
        bd_top = False
        bd_bot = False
        for y in iys:
            if y == 0:
                bd_top = True
                bd_bot = False
            elif y == self.num_y - 1:
                bd_top = False
                bd_bot = True
            else:
                bd_top = False
                bd_bot = False

            for x in ixs:
                if x == 0:
                    bd_left = True
                    bd_right = False
                elif x == self.num_x - 1:
                    bd_left = False
                    bd_right = True
                else:
                    bd_left = False
                    bd_right = False
                boundaries.append((bd_left, bd_right, bd_top, bd_bot))

        self.boundaries = boundaries


    def patch(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        mask = None,
        frame_out = None,
        mask_out = None,
        channels = 3,
        device = None
    ):
        assert (
            isinstance(frame, np.ndarray)
            or
            isinstance(frame, torch.Tensor)
        ), "ERROR: frame must be either a numpy array or torch.Tensor"

        assert frame.ndim == 3, "ERROR: frame must be a 3D array"
        if frame_out is None:
            frame_out = torch.zeros(self.num, channels, self.patch_height, self.patch_width, dtype = torch.float, device = device)

        # add padding if needed
        if frame.shape[1] != self.frame_height or frame.shape[2] != self.frame_width:
            frame_tmp = torch.zeros(channels, self.frame_height, self.frame_width, dtype = frame.dtype, device = frame.device)
            frame_tmp[:, :frame.shape[1], :frame.shape[2]] = frame
            frame = frame_tmp

            if not mask is None:
                mask_tmp = torch.zeros(self.frame_height, self.frame_width, dtype = mask.dtype, device = mask.device)
                mask_tmp[:mask.shape[0], :mask.shape[1]] = mask
                mask = mask_tmp

        for idx, (x, y) in enumerate(self.coords_xy):
            patch_img = frame[:, y:y + self.patch_height, x:x + self.patch_width]
            frame_out[idx, :, :, :] = patch_img

        if mask is None:
            return frame_out
        else:
            if mask_out is None:
                mask_out = torch.zeros(self.num, self.patch_height, self.patch_width, dtype = mask.dtype, device = device)
            
            for idx, (x, y) in enumerate(self.coords_xy):
                patch_mask = mask[y:y + self.patch_height, x:x + self.patch_width]
                mask_out[idx, :, :] = patch_mask

            return frame_out, mask_out

    def stitch(self, patches, plaid = None):

        if plaid is None:
            # print("Allocates repeatedly")
            plaid = torch.zeros(self.frame_height, self.frame_width, dtype = patches.dtype, device = patches.device)
            numerator = torch.zeros(self.frame_height, self.frame_width, dtype = torch.float, device = patches.device)
            denominator = torch.zeros(self.frame_height, self.frame_width, dtype = torch.float, device = patches.device)

        else:
            # TODO: remove when proper boundary handling in place
            if self.is_strided:
                plaid[:, :] = 0


        if len(patches.shape) == 2:
            plaid[:, :] = patches[:, :]

            if self.pad_width > 0 or self.pad_height > 0:
                plaid = plaid[:self.frame_height - self.pad_height, :self.frame_width - self.pad_width]

            return plaid

        if not self.is_strided:
            #print('plaid', plaid.shape, 'patches', patches.shape)
            for idx, (x, y) in enumerate(self.coords_xy):
                # print('idx', idx, 'xy', x, y, 'patch', self.patch_width, self.patch_height)
                # print(f"{type(plaid)=}")
                # print(f"{plaid.shape=}")
                # print(f"{plaid.dtype=}")

                # print(f"{type(patches)=}")
                # print(f"{patches.shape=}")
                # print(f"{patches.dtype=}")

                # print(
                #     plaid[y:y+self.patch_height, x:x + self.patch_width]
                # )
                # print(
                #     patches[idx, :, :]
                # )
                plaid[y:y+self.patch_height, x:x + self.patch_width] = patches[idx, :, :]

            if self.pad_width > 0 or self.pad_height > 0:
                plaid = plaid[:self.frame_height - self.pad_height, :self.frame_width - self.pad_width]

            return plaid

        # strided
        for idx, (x, y) in enumerate(self.coords_xy):
            # TODO: use self.pad_width, self.pad_height and self.boundaries and set value instead of adding
            numerator[y:y+self.patch_height, x:x + self.patch_width] += patches[idx, :, :]
            denominator[y:y+self.patch_height, x:x + self.patch_width] += 1
            if self.boost_center > 0:
                numerator[y + self.patch_height // 4: y + 3 * (self.patch_height // 4), x + self.patch_width // 4: x + 3 * (self.patch_width // 4)] += patches[idx, self.patch_height // 4:3 * (self.patch_height // 4), self.patch_width // 4:3 * (self.patch_width // 4)] * self.boost_center
                denominator[y + self.patch_height // 4: y + 3 * (self.patch_height // 4), x + self.patch_width // 4: x + 3 * (self.patch_width // 4)] += self.boost_center

        plaid[:, :] = numerator / denominator

        if self.pad_width > 0 or self.pad_height > 0:
            plaid = plaid[:self.frame_height - self.pad_height, :self.frame_width - self.pad_width]

        return plaid
