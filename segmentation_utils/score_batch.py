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
from torch.cuda.amp import autocast, GradScaler


def score_batch(
    torch_device,
    model,
    threshold,
    batch_of_color_tiles,
    batch_of_segmented_tiles  # answer goes here
):
    """
    batch_size is 64. Choose a batch_size that your GPU can handle.
    Suppose batch_of_color_tiles is a 64 x height x width x 3rgb channels uint8 np.array.
    Returns binary segmentation answers
    in the 64 x height width np.uint8 np.array batch_of_segmented_tiles.
    """
    chunk_w = 224
    chunk_h = 224
    assert batch_of_color_tiles.shape == (64, 224, 224, 3), f"{batch_of_color_tiles.shape}" # b x h x w x c

    np_float32 = batch_of_color_tiles.astype(np.float32) / 255.0  # float [0,1] the whole image
    bchw = np.transpose(np_float32, axes=(0, 3, 1, 2))  # transpose the whole image to chw
    assert bchw.shape == (64, 3, 224, 224)  # pytorch wants batch x channels x height x width
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean_extra_axes = mean[None, ..., None, None]
    assert mean_extra_axes.shape == (1, 3, 1, 1)
    std_extra_axes = std[None, ..., None, None]
    assert std_extra_axes.shape == (1, 3, 1, 1)
    bchw_normalized = (bchw - mean_extra_axes) / std_extra_axes
    xb_cpu = torch.tensor(bchw_normalized)
    xb = xb_cpu.to(torch_device)
    solve_start = time.time()
    torch.cuda.synchronize()
    # with torch.no_grad():
    with autocast():
        out_gpu = model(xb)
    torch.cuda.synchronize()  # without this, timings will be very misleading
    solve_end = time.time()
    print(f"Inference time for one batch of 64: {solve_end - solve_start}")
    gpu_log_probs_torch = F.log_softmax(out_gpu, dim=1)
    pos_channel_of_gpu_log_probs_torch = gpu_log_probs_torch[:, 1, :, :]
    probs_gpu = torch.exp(pos_channel_of_gpu_log_probs_torch)
    binary_gpu = torch.ge(probs_gpu, threshold)
    start_detach = time.time()
    binary_cpu = binary_gpu.cpu()
    torch.cuda.synchronize()  # without this, timings will be very misleading
    end_detach = time.time()
    print(f"detach time for one batch of 64: {end_detach - start_detach}")
    start_detach = time.time()
    out_cpu_torch = out_gpu.cpu()
    torch.cuda.synchronize()  # without this, timings will be very misleading
    end_detach = time.time()
    print(f"detach time for one batch of 64: {end_detach - start_detach}")
    log_probs_torch = F.log_softmax(out_cpu_torch.float(), dim=1)
    log_probs_np = log_probs_torch.detach().numpy()
    probs_np = np.exp(log_probs_np)
    probs = probs_np[:, 1, :, :]  # extract the probability of foreground i.e. nonfloor
    
    # this is how we return the answer:
    batch_of_segmented_tiles[:, :, :] = (probs > threshold).astype(np.uint8)

    assert np.all(
        np.logical_or(
            batch_of_segmented_tiles == 0,
            batch_of_segmented_tiles == 1
        )
    )
    