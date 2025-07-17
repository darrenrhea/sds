from get_final_model_from_id import (
     get_final_model_from_id
)
from get_cuda_devices import (
     get_cuda_devices
)
from Segmenter import (
     Segmenter
)


def segmenter_from_final_model_id(
    final_model_id: str
) -> Segmenter:

    print(f"{final_model_id=}")
    final_model = get_final_model_from_id(
        final_model_id=final_model_id,
    )



    model_architecture_family_id = final_model.model_architecture_family_id
    weights_file_path = final_model.weights_file_path
    original_width = final_model.original_width
    original_height = final_model.original_height
    patch_width = final_model.patch_width
    patch_height = final_model.patch_height
    patch_stride_width = final_model.patch_stride_width
    patch_stride_height = final_model.patch_stride_height
    pad_height = final_model.pad_height
    pad_width = 0

    print(f"{model_architecture_family_id=}")
    print(f"{weights_file_path=}")
    print(f"{original_width=}")
    print(f"{original_height=}")
    print(f"{patch_width=}")
    print(f"{patch_height=}")
    print(f"{patch_stride_width=}")
    print(f"{patch_stride_height=}")
    print(f"{pad_height=}")

          

    devices = get_cuda_devices()
    device = devices[0]

    segmenter = Segmenter(
        device=device,
        fn_checkpoint=weights_file_path,
        model_architecture_id=model_architecture_family_id,
        original_height=original_height,
        original_width=original_width,
        inference_height=original_height,
        inference_width=original_width,
        pad_height=pad_height,
        pad_width=pad_width,
        patch_height=patch_height,
        patch_width=patch_width,
        patch_stride_height=patch_stride_height,
        patch_stride_width=patch_stride_width,
    )

    return segmenter