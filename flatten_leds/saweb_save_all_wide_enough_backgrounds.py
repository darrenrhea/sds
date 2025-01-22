from edub_extract_discovered_unoccluded_background import (
     edub_extract_discovered_unoccluded_background
)


def saweb_save_all_wide_enough_backgrounds():
    clip_id = "hou-sas-2024-10-17-sdi"
    frame_indices = [
        146000,
        151000,
        153000,
        160000,
        168000,
        169000,
        180500,
        185000,
        186500,
        191000,
        197000,
        210000,
        211000,
        216000,
        242000,
        252500,
        258000,
        274500,
        275000,
        277000,
        283000,
        301000,
        301500,
        310000,
        312000,
        312500,
        316500,
        321000,
        348500,
        384000,
        398500,
        403000,
        416000,
    ]
    rip_height = 256
    rip_width = 2560
    min_width = 512

    for frame_index in frame_indices:
        edub_extract_discovered_unoccluded_background(
            clip_id=clip_id,
            frame_index=frame_index,
            rip_height=rip_height,
            rip_width=rip_width,
            min_width=min_width
        )


if __name__ == "__main__":
    saweb_save_all_wide_enough_backgrounds()