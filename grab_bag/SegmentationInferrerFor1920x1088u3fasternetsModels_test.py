from SegmentationInferrerFor1920x1088u3fasternetsModels import (
     SegmentationInferrerFor1920x1088u3fasternetsModels
)
import torch
import cv2
import numpy as np
from pathlib import Path


def test_SegmentationInferrerFor1920x1088u3fasternetsModels_1():
    """
    This is a test for the SegmentationInferrerFor1920x1088u3fasternetsModels class.
    It will load a model and run inference on an image.
    """

    pt_weights_file_path = Path(
        "/shared/checkpoints/u3fasternets-floor-95frames-512x512-buccaneersfloorrev0_epoch000197.pt"
    )
    
    device = torch.device("cuda:0")

    inferrer = SegmentationInferrerFor1920x1088u3fasternetsModels(
        pt_weights_file_path=pt_weights_file_path,
        device=device,
    )
    
    frame_index = np.random.randint(0, 100000)
    
    input_image_file_path = Path(
        f"/hd2/clips/nfl-59778-skycam/frames/nfl-59778-skycam_{frame_index:06d}_original.jpg"
    )
    print(f"Inferring image:\n   {pt_weights_file_path}")

    # Load as RGB with values in [0.0, 1.0]
    # prii(input_image_file_path)
    
    bgr_hwc_np_u8 = cv2.imread(str(input_image_file_path.resolve()), cv2.IMREAD_COLOR)
    rgb_hwc_np_u8 = bgr_hwc_np_u8[:, :, ::-1]
    # pad with 8 more rows of black:
    padded_rgb_hwc_np_u8 = np.zeros(
        (1088, 1920, 3), dtype=np.uint8
    )
    padded_rgb_hwc_np_u8[:1080, :, :] = rgb_hwc_np_u8

    # 2. Convert to float32 and scale to [0,1]
    rgb_hwc_np_f32 = padded_rgb_hwc_np_u8.astype(np.float32) / 255.0

    assert rgb_hwc_np_f32.dtype == np.float32
    assert rgb_hwc_np_f32.min() >= 0.0
    assert rgb_hwc_np_f32.max() <= 1.0

    padded_mask_hw_np_f32 = inferrer.infer(
        rgb_hwc_np_nonlinear_f32=rgb_hwc_np_f32,
    )
    
    assert padded_mask_hw_np_f32.dtype == np.float32
    assert padded_mask_hw_np_f32.shape == (1088, 1920)
    print(f"{padded_mask_hw_np_f32.min()=}, {padded_mask_hw_np_f32.max()=}")
    assert padded_mask_hw_np_f32.max() >= - 0.001
    assert padded_mask_hw_np_f32.max() <=   1.001
    
    # depad and convert to uint8 in [0, 255]:
    mask_hw_np_u8 = np.clip(
        np.round(padded_mask_hw_np_f32[:1080, :] * 255.0),
        0,
        255
    ).astype(np.uint8)

    # Save the mask
    output_mask_file_path = Path(
        "temp.png"
    ).resolve()

    bgra_hwc_np_u8 = np.zeros(
        (1080, 1920, 4), dtype=np.uint8
    )
    bgra_hwc_np_u8[:, :, :3] = bgr_hwc_np_u8[:, :, :]
    bgra_hwc_np_u8[:, :, 3] = mask_hw_np_u8[:, :]

    bgra_hwc_np_u8 = np.zeros(
        (1080, 1920, 4), dtype=np.uint8
    )
    bgra_hwc_np_u8[:, :, :3] = bgr_hwc_np_u8[:, :, :]
    bgra_hwc_np_u8[:, :, 3] = mask_hw_np_u8[:, :]

    # prii(bgra_hwc_np_u8[:, :, [2, 1, 0, 3]])
    cv2.imwrite(str(output_mask_file_path), bgra_hwc_np_u8)
    
    print(f"Mask saved to {output_mask_file_path}")



if __name__ == "__main__":
    test_SegmentationInferrerFor1920x1088u3fasternetsModels_1()
    print("Test passed successfully.")