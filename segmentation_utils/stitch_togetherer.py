"""
Oftentimes these image segmentation neural networks can
take in only a relatively small image, say 224x224,
so that GPU memory usage is acceptable.
We need to apply it at various translations so that
all the strided 224x224 patches cover
the full-sized image, possibly with lots of overlap.
Some kind of voting / ensembling must be done for pixels
that are contained by several patches.
"""
import numpy as np
import torch
import torch.nn.functional as F
import time
from fastai.vision.all import *


def stitch_togetherer(
    hwc_np_uint8,
    j_min,
    i_min,
    j_max,
    i_max,
    torch_device,
    model,
    nn_input_width,
    nn_input_height,
    threshold,
    stride,
    batch_size
):
    """
    Suppose hwc_np_uint8 is a full-size i.e. 1920x1080 color image.
    We might only want to score the subrectangle [i_min:i_max, j_min:j_max].
    Returns a binary 0-1 np.uint8 np.array of the same height and width as
    the input image hwc_np_uint8.
    Typically the neural network input is much smaller than a full image,
    say nn_input_width wide by nn_input height tall,
    so we have to cover the part of the image we wish to
    score with chunk_w by chunk_h tiles (possibly with much overlap).
    Choose a batch_size that your GPU can handle.
    """
    chunk_w = nn_input_width
    chunk_h = nn_input_height

    height = i_max - i_min
    width = j_max - j_min
    lefts = [x for x in range(0, width - chunk_w, stride)] + [width - chunk_w]
    uppers = [y for y in range(0, height - chunk_h, stride)] + [height - chunk_h]

    num_times_scored = np.zeros(shape=(height, width), dtype=np.int16)
    total_score = np.zeros(shape=(height, width), dtype=np.int16)

    channel_change_start = time.time()
    chw_np_uint8 = np.transpose(
         hwc_np_uint8,
        axes=(2, 0, 1)
    )
    channel_change_end = time.time()
    # print(f"channel change time {channel_change_end - channel_change_start}")

    convert_to_float_32_start = time.time()
    chw = chw_np_uint8[:, i_min:i_max, j_min:j_max].astype(np.float32) / 255.0  # float [0,1] the whole image
    # chw = np.transpose(np_float32,
    #                    axes=(2, 0, 1))  # transpose the whole image to chw
    convert_to_float_32_end = time.time()
    # print(f"convert to float32 time {convert_to_float_32_end - convert_to_float_32_start}")
        
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalize_start = time.time()
    normalized = (chw - mean[..., None, None]) / std[..., None, None]
    normalize_end = time.time()
    # print(f"normalize time {normalize_end - normalize_start}")

    chunk_start = time.time()
    all_chunks = [(left, upper) for left in lefts for upper in uppers]
    chunk_end = time.time()
    # print(f"chunking time {chunk_end - chunk_start}")

    batch_alloc_start = time.time()
    batch = np.zeros(shape=(batch_size, 3, chunk_h, chunk_w), dtype=np.float32)
    batch_alloc_end = time.time()
    # print(f"batch allocation time {batch_alloc_end - batch_alloc_start}")

    batch_start = time.time()
    batches = [
        all_chunks[x:x + batch_size]
        for x in range(0, len(all_chunks), batch_size)
    ]
    batch_end = time.time()
    # print(f"batch assignment time {batch_end - batch_start}")

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            stride_counter = 0
            for batch_of_chunks in batches:
                bs = len(batch_of_chunks)  # for the remainder, it can be smaller than batch_size
                assert bs <= batch_size
                # load up as much as a full batch:
                for cntr, (left, upper) in enumerate(batch_of_chunks):
                    right = left + chunk_w
                    lower = upper + chunk_h
                    batch[cntr, :, :, :] = normalized[:, upper:lower, left:right]
                
                s = np.sum(batch[0])  # get the sum of the first tile right before it goes through the nn
                # print(f"Sum of the 0-ith tile {upper} {lower} {left} {right} is is {s}")
                s = np.sum(batch[1])  # get the sum of the first tile right before it goes through the nn
                # print(f"Sum of the 1-ith tile is is {s}")
                # predict that batch
                tensor_start = time.time()
                xb_cpu = torch.tensor(batch)
                torch.cuda.synchronize()
                tensor_end = time.time()
                print(f"convert to tensor time {tensor_end - tensor_start}")
                gpu_start = time.time()
                xb = xb_cpu.to(torch_device)
                torch.cuda.synchronize()
                gpu_end = time.time()
                print(f"gpu load time {gpu_end - gpu_start}")
                solve_start = time.time()
                out = model(xb)
                torch.cuda.synchronize()
                solve_end = time.time()
                print(f"Inference time {solve_end - solve_start}")
                detach_start = time.time()
                out_cpu_torch = out.detach().cpu()
                torch.cuda.synchronize()
                detach_end = time.time()
                print(f"detach from gpu time {detach_end - detach_start}")
                tensor_log_start = time.time()
                log_probs_torch = F.log_softmax(out_cpu_torch.type(torch.DoubleTensor), dim=1)
                tensor_log_end = time.time()
                # print(f"tensor log time {tensor_log_end - tensor_log_start}")
                log_probs_start = time.time()
                log_probs_np = log_probs_torch.numpy()
                log_probs_end = time.time()
                # print(f"logs probs time {log_probs_end - log_probs_start}")
                # probs_np = np.exp(log_probs_np)
                # chunk_probs = probs_np[:, 1, :, :]
                threshold_start = time.time()
                chunk_binary_prediction = (log_probs_np[:, 1, :, :] > np.log(threshold)).astype(np.uint8)
                threshold_end = time.time()
                # print(f"thresholding time {threshold_end - threshold_start}")

                out_start = time.time()
                for cntr, (left, upper) in enumerate(batch_of_chunks):
                    # print(f"{upper}:{upper + chunk_h}, {left}:{left + chunk_w}")
                    total_score[upper:upper + chunk_h, left:left + chunk_w] += chunk_binary_prediction[cntr, :, :]
                    num_times_scored[upper:upper + chunk_h, left:left + chunk_w] += 1
                    stride_counter += 1
                out_end = time.time()
                # print(f"batch results collection time {out_end - out_start}")

    print(f"tot num strides {stride_counter}")
    assert np.min(
        num_times_scored) >= 1, "ERROR: not every pixel was scored at least once!"
    fraction_of_votes = total_score / num_times_scored
    uint8_start = time.time()
    binary_prediction = (fraction_of_votes >= 0.5).astype(np.uint8)  # any positive??!!
    uint8_end = time.time()
    # print(f"uint8 time {uint8_end - uint8_start}")
    assert np.all(
        np.logical_or(
            binary_prediction == 0,
            binary_prediction == 1
        )
    )
    return binary_prediction
