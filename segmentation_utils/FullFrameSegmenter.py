import numpy as np
import torch
import torch.nn.functional as F
import time
from fastai.vision.all import *
import PIL
import PIL.Image
import my_pytorch_utils
from pathlib import Path
import numpy as np
import math
from colorama import Fore, Style
import sys


def remove_pth_suffix(model_name: str) -> str:
    """
    Fixes a weird issue that fastai does not want the full model name
    """
    if model_name[-4:] == ".pth":
        model_name_without_the_pth_suffix = model_name[:-4]
    elif model_name[-3:] == ".pt":
        model_name_without_the_pth_suffix = model_name[:-3]
    else:
        print(f"ERROR: fastai model weights files should end in .pth or .pt, but model_name = {model_name}")
        sys.exit(1)
    return model_name_without_the_pth_suffix

class FullFrameSegmenter(object):
    """
    We need a segmenter object that we can set up once
    and then call with a few frames.
    The setup process involves telling it:

    original_height and original_width of the full sized image, usually 1920 1080
    which gpu to use
    which model to use
    model info, like 400x400 tiles and downsample_factor, which is usually either 1 or 2
    how many frames at a time.
    
    """
    def __init__(
        self,
        gpu_substring,
        which_gpu,
        original_height,
        original_width,
        downsample_factor,
        nn_input_width,
        nn_input_height,
        num_frames_per_gpu_batch,
        model_name,
        architecture
    ):
        assert gpu_substring in ["A5000", "8000"], f"Questionable gpu_string {gpu_substring}, halting."
        self.gpu_substring = gpu_substring

        assert which_gpu in [
            0,
            1,
        ], "ERROR: which_gpu must be 0 or 1 to identify which of the two gpus selected by gpu_substring you want."
        self.which_gpu = which_gpu

        assert original_width in [1920], f"Suspicious original_width {original_width}, halting."
        self.original_width = original_width

        assert original_height in [1080], f"Suspicious original_height {original_height}, halting."
        self.original_height = original_height

        assert downsample_factor in [1, 2], f"Very suspicious downsample_factor {downsample_factor}, halting"
        self.downsample_factor = downsample_factor

        assert num_frames_per_gpu_batch in [1, 2, 3, 4], f"Very suspicious num_frames_per_gpu_batch {num_frames_per_gpu_batch}, halting"
        self.num_frames_per_gpu_batch = num_frames_per_gpu_batch  

        assert architecture in ["resnet18", "resnet34"]
        self.architecture = architecture

        models_dir = Path("~/r/trained_models").expanduser()
        abs_path = models_dir / f"{model_name}"
        # The weird thing is that fastai does not want the model_path to end in .pth
        model_name_without_the_pth_suffix = remove_pth_suffix(model_name)
        self.model_path = models_dir / f"{model_name_without_the_pth_suffix}"
        assert abs_path.is_file(), f"The model {abs_path} does not exist"

        # the neural network takes in a smaller size like 224x224, 400x400, or 320x280:
        self.nn_input_width = nn_input_width
        self.nn_input_height = nn_input_height
        
        
      
        # WARNING: this next line seems crazy specific to gsw1, but it is actually necessary
        # Because fastai is essentially forcing us to make a dataloader as-if we were training
        # despite that we are only doing inference.
        # What has to be in this directory for it to work?
        path = Path("~/r/gsw1/224_224_one_third_downsample_croppings").expanduser()
        # path = Path('~/r/gsw1/280_320_croppings').expanduser()
        # path = Path(f'~/r/brooklyn_nets_barclays_center/320_280_one_half_downsampled_croppings').expanduser()
        

        self.torch_device = my_pytorch_utils.get_the_correct_gpu(self.gpu_substring, which_copy=self.which_gpu)
        fnames = list(path.glob("*_color.png"))  # list of input frame paths
        codes = np.array(["nonfloor", "floor"])


        def label_func(fn):
            return path / f"{fn.stem[:-6]}_nonfloor{fn.suffix}"


        dls = SegmentationDataLoaders.from_label_func(
            path=path,
            bs=32,
            fnames=fnames,
            label_func=label_func,
            codes=codes,
            valid_pct=0.1,
            seed=42,  # random seed
        )

        if self.architecture == "resnet34":
            arch = resnet34
        elif self.architecture == "resnet18":
            arch = resnet18
        learner = unet_learner(dls=dls, arch=arch)
        self.model = learner.load(self.model_path)
        self.model = self.model.to(self.torch_device)
        self.model.eval()  # we aren't training, only infering
        torch.cuda.synchronize()


    def infer_full_size_rgb_pil_frames(self, pil_frames):
        """
        We are trying to make something that is reusable here.
        Previous codes were too bound up with file access, directories, etc.
        There should be no file access in here, neither file reading nor writing.
        
        Takes in a List of PIL frames, usually 1920x1080.
        Why PIL?  We don't know, other than to say that it has a resize function.
        Should this even do resizing?

        Returns a num_images by height by width numpy uint8 filled with the images' segmentations.

        Takes in num_frames_per_gpu_batch image files at a time,
        each of which is originally 1920x1080

        but they might be downsampled by 2 in both dimensions immediately to 960x540.

        Then each image gets covered by "neural network input tiles."
        That is a total of 15 different 400x400 NN inputs, which together form a batch tensor
        of size 15 x 3 x 400 x 400 (Batch x RGBChan x Height x Width).

        This batch gets put through the NN, and we get segmentations for the tiles.
        Then we have to reassemble the tiles into a segmentation for the whole image.

        Iterate over all frames and collect batches. The size of each batch
        is based on the desired neural network input width and height.
        Once a collection of batches is obtained, each batch in the
        collection is processed by the neural network.
        Note that we need to maintain the association between
        any patch inside a batch with the original frame that it came from
        so that we can reconstruct an answer for each frame from the
        answers of all patches that constitute it, even if they
        span multiple batches.
        """
        
        assert len(pil_frames) <= self.num_frames_per_gpu_batch, f"Why is len(pil_frames) == {len(pil_frames)}"

        # Before covering it by tiles, we downsample by the downsampling_factor:
        small_height = self.original_height // self.downsample_factor
        small_width = self.original_width // self.downsample_factor

        width = small_width
        height = small_height
        num_patches_per_frame = math.ceil(small_width / self.nn_input_width) * math.ceil(small_height / self.nn_input_width)
        # print(f"num patches per frame {num_patches_per_frame}")
        j_stride = self.nn_input_width  # no overlap: the tiles are 0:320, 320:640, 640:960 as far as j.
        i_stride = self.nn_input_height  # Because nn_input_height - 20 = 260.  TODO: generalize this.
        # vertically the top row tiles have i range over 0:280,
        # then the next row of tiles has i range over in 260:540.

        lefts = [x for x in range(0, width - self.nn_input_width, j_stride)] + [width - self.nn_input_width]
        uppers = [y for y in range(0, height - self.nn_input_height, i_stride)] + [height - self.nn_input_height]
        upper_left_corners_of_all_tiles = [(left, upper) for left in lefts for upper in uppers]

        # print(f"The {width}x{height} image will be covered by these nn_input_tiles:")
        # for left, upper in upper_left_corners_of_all_tiles:
        #     right = left + self.nn_input_width
        #     lower = upper + self.nn_input_height
        #     print(f"nn_input_tile = image[{upper}:{lower}, {left}:{right}]")


        # print(f"num_frames_per_gpu_batch {self.num_frames_per_gpu_batch}")

        # for each frame and each pixel in a frame, total_score stores the score for that pixel.
        # TODO: the name total_score is totally inappropriate at this point
        total_score = np.zeros(shape=[self.num_frames_per_gpu_batch, height, width], dtype=np.uint8)
        original_image_array = np.zeros(shape=(self.num_frames_per_gpu_batch, self.original_height, self.original_width, 3), dtype=np.int8)

        # we have num_frames_per_gpu_batch * 6 tiles = 24 tiles.  Each has bounds [upper, lower, left, right]
        # so that it was cut out via tile = image[upper:lower, left:right]
        tile_index_to_tile_bounds = {}

        num_tiles_per_batch = self.num_frames_per_gpu_batch * num_patches_per_frame
        # print(f"num_tiles_per_batch = {num_tiles_per_batch}")

        batch_of_tiles = np.zeros(shape=(num_tiles_per_batch, 3, self.nn_input_height, self.nn_input_width), dtype=np.float32)

        # we are going to do the simplest thing that gives decent speed,
        # a group of, say 4, images will be inferred at a time
        # by putting a single tensor, batch_of_tiles, through the neural network:
    
        tile_index_to_within_the_group_frame_index = dict()

        tile_index = 0  # this should go from 0 to 23 since there are 4 * 6 = 24 tiles to put through the NN

        # within_the_group_frame_index ranges from 0 to 3:
        for within_the_group_frame_index, image_pil in enumerate(pil_frames):
            # the index of the image we are cutting into tiles:
            img_pil = image_pil.convert("RGB")
            if self.downsample_factor != 1:
                smaller_pil = img_pil.resize((small_width, small_height), Image.ANTIALIAS)  # 960x540
            else:
                smaller_pil = img_pil
                hwc_np_uint8 = np.array(smaller_pil)  # as a hwc numpy array
                original_image_array[within_the_group_frame_index, :, :, :] = np.array(img_pil)
           

            chw_np_uint8 = np.transpose(hwc_np_uint8, axes=(2, 0, 1))

            # convert the image to float32s ranging over [0,1]:
            chw_np_float32 = chw_np_uint8[:, :, :].astype(np.float32) / 255.0

            # normalize it like AlexNet:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (chw_np_float32 - mean[..., None, None]) / std[..., None, None]

            # chop out the tiles, stick them into the batch:
            for (left, upper) in upper_left_corners_of_all_tiles:
                right = left + self.nn_input_width
                lower = upper + self.nn_input_height
                batch_of_tiles[tile_index, :, :, :] = normalized[:, upper:lower, left:right]
                tile_index_to_within_the_group_frame_index[tile_index] = within_the_group_frame_index
                tile_index_to_tile_bounds[tile_index] = [upper, lower, left, right]
                tile_index += 1
        # At this point, the batch tensor should be ready to go through

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast():
        # predict that batch
        
        xb_cpu = torch.tensor(batch_of_tiles)
        xb = xb_cpu.to(self.torch_device)
        out = self.model(xb)
        log_probs_torch = F.log_softmax(out.type(torch.DoubleTensor), dim=1)
        probs_times_255_gpu = torch.exp(log_probs_torch[:, 1, :, :]) * 255
        probs_times_255 = probs_times_255_gpu.detach().cpu().numpy().astype(np.uint8)
        

        for tile_index in range(num_tiles_per_batch):
            upper, lower, left, right = tile_index_to_tile_bounds[tile_index]
            within_the_group_frame_index = tile_index_to_within_the_group_frame_index[tile_index]
            total_score[within_the_group_frame_index, upper:lower, left:right] = probs_times_255[tile_index, :, :]

       

        return total_score


