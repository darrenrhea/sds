from get_final_model_from_id import (
     get_final_model_from_id
)
from get_cuda_devices import (
     get_cuda_devices
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)


def grirosffmi_get_ram_in_ram_out_segmenter_from_final_model_id(
    final_model_id: str
):
    """
    When speed is less of a concern, you just want to infer with the model.
    """
    final_model = get_final_model_from_id(
        final_model_id=final_model_id,
    )

    model_architecture_family_id = final_model.model_architecture_family_id
    weights_file_path = final_model.weights_file_path
    inference_width = final_model.original_width
    inference_height = final_model.original_height
    patch_width = final_model.patch_width
    patch_height = final_model.patch_height
    patch_stride_width = final_model.patch_stride_width
    patch_stride_height = final_model.patch_stride_height
    pad_height = final_model.pad_height
    pad_width = 0

    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    device = devices[0]

    ram_in_ram_out_segmenter = RamInRamOutSegmenter(
        device=device,
        fn_checkpoint=weights_file_path,
        model_architecture_id=model_architecture_family_id,
        inference_height=inference_height,
        inference_width=inference_width,
        pad_height=pad_height,
        pad_width=pad_width,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_stride_height=patch_stride_height,
        patch_stride_width=patch_stride_width,
    )

    return ram_in_ram_out_segmenter

    