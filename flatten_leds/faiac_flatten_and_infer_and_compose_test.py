from get_final_model_from_id import (
     get_final_model_from_id
)
from get_cuda_devices import (
     get_cuda_devices
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)
from faiac_flatten_and_infer_and_compose import (
     faiac_flatten_and_infer_and_compose
)




if __name__ == "__main__":

    final_model_id = "brewcubflattenedvip4"

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

    frame_ranges = [
        # [295987, 296000, 1], # ball infront of right board
        # [24000, 300000, 500],  # generally
        #[125500, 125500+1, 1],
        # [249649, 250687, 50],
        [24689, 24802, 1]
    ]

    for a, b, c in frame_ranges:       
        for frame_index in range(a, b, c):
            faiac_flatten_and_infer_and_compose(
                ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
                clip_id="brewcub",
                frame_index=frame_index,
                board_ids=["left", "right"],
                board_id_to_rip_height={"left": 256, "right": 256},
                board_id_rip_width={"left": 1024, "right": 1024},
            )

    """
    mpv_make_plain_video --frames_dir /shared/clips/brewcub/compositions --clip_id brewcub --original_suffix .png --first_frame_index 24689 --last_frame_index 24801 --fps 59.94 --out temp.mp4
    """