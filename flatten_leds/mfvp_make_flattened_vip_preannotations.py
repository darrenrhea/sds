from masfvp_make_a_single_flattened_vip_preannotation import (
     masfvp_make_a_single_flattened_vip_preannotation
)
from get_cuda_devices import (
     get_cuda_devices
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)




if __name__ == "__main__":
    pad_height = 0
    pad_width = 0
    patch_width = 512
    patch_height = 256
    inference_width = 1024
    inference_height = 256
    patch_stride_width = 256
    patch_stride_height = 256
    model_id_suffix = ""
    original_height = 256
    original_width = 1024
    model_architecture_id = "effl"
    # fn_checkpoint = "/shared/checkpoints/effl-flatled-1256frames-512x256-somefake_epoch000013.pt"
    fn_checkpoint = "/shared/checkpoints/effl-flatled-3256frames-512x256-morefake_epoch000004.pt"
    devices = get_cuda_devices()
    num_devices = len(devices)
    assert num_devices > 0, "no cuda devices found"
    device = devices[0]

    ram_in_ram_out_segmenter = RamInRamOutSegmenter(
        device=device,
        fn_checkpoint=fn_checkpoint,
        model_architecture_id=model_architecture_id,
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
        [295987, 296000, 1], # ball infront of right board
        [23100, 300800, 100],  # generally
        # [125500, 125500+1, 1],
        # [249649, 250687, 50],
    ]

    for a, b, c in frame_ranges:       
        for frame_index in range(a, b, c):
            masfvp_make_a_single_flattened_vip_preannotation(
                ram_in_ram_out_segmenter=ram_in_ram_out_segmenter,
                clip_id="brewcub",
                frame_index=frame_index,
                board_ids=["left", "right"],
                board_id_to_rip_height={"left": 256, "right": 256},
                board_id_rip_width={"left": 1024, "right": 1024},
            )