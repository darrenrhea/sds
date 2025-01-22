from IPython.core.display import display
import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F

def pred_for_chunk(
    torch_device,
    model,
    threshold,
    img_hwc_np_uint8,
    left,
    upper,
    chunk_w,
    chunk_h,
    show_plots=False
):
    """
    There are better versions of this that use batches.
    we load a (presumably 3840 x 2160 x 3) image from the file.
    a chunk_w wide x chunk_h tall cropping is cut out as [upper:lower, left:right, :]
    The channels are permuted so that it is CHW.
    The channels are ImageNet normalized.
    Then it is predicted and a mask is shown if show_plots = True.
    The mask is returned.
    """
    assert left <= 3840 and left >= 0
    assert upper <= 2160 and upper >= 0
    right = left + chunk_w
    lower = upper + chunk_h
    assert right <= 3840
    assert lower <= 2160

    chunk_hwc_np_uint8 = img_hwc_np_uint8[upper:lower, left:right, :]
    
    np_float32 = chunk_hwc_np_uint8.astype(np.float32) / 255.0

    chw = np.transpose(np_float32, axes=(2, 0, 1))

    assert chw.shape == (3, chunk_h, chunk_w), f"{chw.shape} is wrong"

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized = (chw - mean[..., None, None]) / std[..., None, None]

    batch = np.zeros(shape=(1, 3, chunk_h, chunk_w), dtype=np.float32)  # batch size is not fixed
    batch[0, :, :, :] = normalized
    xb_cpu = torch.tensor(batch)
    xb = xb_cpu.to(torch_device)
    out = model(xb)
    out_cpu_torch = out.detach().cpu()
    log_probs_torch = F.log_softmax(out_cpu_torch, dim=1)
    log_probs_np = log_probs_torch.numpy()
    probs_np = np.exp(log_probs_np)
    # print(f"prob_np has shape {probs_np.shape} and max {np.max(probs_np)}")
    # print(np.min(probs_np[0, 1, :, :] + probs_np[0, 0, :, :]))  # check probabilities adds to 1
    binary_prediction = (probs_np[0, 1, :, :] > threshold).astype(np.uint8)

    assert binary_prediction.shape == (chunk_h, chunk_w)
    if show_plots:
        display(PIL.Image.fromarray(chunk_hwc_np_uint8))
        print("prediction")
        display(PIL.Image.fromarray(binary_prediction * 255))
        display(PIL.Image.fromarray(binary_prediction[:, :, np.newaxis] * chunk_hwc_np_uint8))
        display(PIL.Image.fromarray(np.logical_not(binary_prediction)[:, :, np.newaxis] * chunk_hwc_np_uint8))

    return binary_prediction
