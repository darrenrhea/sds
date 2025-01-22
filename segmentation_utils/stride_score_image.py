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

def stride_score_image(
    hwc_np_uint8,
    j_min,
    i_min,
    j_max,
    i_max,
    torch_device,
    model,
    threshold,
    stride,
    batch_size
):
    """
    Suppose hwc_np_uint8 is a full-size i.e. 1920x1080 color image.
    Returns a binary 0-1 np.uint8 np.array of the same height and width as
    the input image hwc_np_uint8.
    The neural network input is much smaller than a full image, so we have to
    stride over it (possibly with much overlap).
    Choose a batch_size that your GPU can handle.
    """
    chunk_w = 224
    chunk_h = 224
    height = i_max - i_min
    width = j_max - j_min
    lefts = [x for x in range(0, width - chunk_w, stride)] + [width - chunk_w]
    uppers = [y for y in range(0, height - chunk_h, stride)] + [height - chunk_h]

    num_times_scored = np.zeros(shape=(height, width), dtype=np.int32)
    total_score = np.zeros(shape=(height, width), dtype=np.int32)

    np_float32 = hwc_np_uint8.astype(
        np.float32)[i_min:i_max, j_min:j_max, :] / 255.0  # float [0,1] the whole image
    chw = np.transpose(np_float32,
                       axes=(2, 0, 1))  # transpose the whole image to chw
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (chw - mean[..., None, None]) / std[..., None, None]
    all_chunks = [(left, upper) for left in lefts for upper in uppers]

    batch = np.zeros(shape=(batch_size, 3, chunk_h, chunk_w), dtype=np.float32)

    batches = [
        all_chunks[x:x + batch_size]
        for x in range(0, len(all_chunks), batch_size)
    ]

    for batch_of_chunks in batches:
        bs = len(batch_of_chunks)  # for the remainder, it can be smaller than batch_size
        assert bs <= batch_size
        # load up as much as a full batch:
        for cntr, (left, upper) in enumerate(batch_of_chunks):
            right = left + chunk_w
            lower = upper + chunk_h
            batch[cntr, :, :, :] = normalized[:, upper:lower, left:right]
        
        # predict that batch
        xb_cpu = torch.tensor(batch)
        xb = xb_cpu.to(torch_device)
        out = model(xb)
        out_cpu_torch = out.detach().cpu()
        log_probs_torch = F.log_softmax(out_cpu_torch, dim=1)
        log_probs_np = log_probs_torch.numpy()
        probs_np = np.exp(log_probs_np)
        chunk_probs = probs_np[:, 1, :, :]
        chunk_binary_prediction = (chunk_probs > threshold).astype(np.uint8)

        for cntr, (left, upper) in enumerate(batch_of_chunks):
            total_score[upper:upper + chunk_h, left:left + chunk_w] += chunk_binary_prediction[cntr, :, :]
            num_times_scored[upper:upper + chunk_h, left:left + chunk_w] += 1

    assert np.min(
        num_times_scored) >= 1, "ERROR: not every pixel was scored at least once!"
    fraction_of_votes = total_score / num_times_scored
    binary_prediction = (fraction_of_votes >= 0.5).astype(np.uint8)  # any positive??!!
    assert np.all(
        np.logical_or(
            binary_prediction == 0,
            binary_prediction == 1
        )
    )
    return binary_prediction
