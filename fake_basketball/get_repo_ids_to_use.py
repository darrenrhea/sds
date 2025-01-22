def get_repo_ids_to_use(
    floor_id: str,
    segmentation_convention: str,
):
    """
    """
    assert floor_id in ["munich"]
    assert segmentation_convention in ["led_not_led"]

    repo_ids_to_use = [
        # "bay-czv-2024-03-01_led",
        # "bay-efs-2023-12-20_led",
        "bay-mta-2024-03-22-mxf_led",
        # "bay-mta-2024-03-22-part1-srt_led",
        # "bay-zal-2024-03-15-yt_led",
        "maccabi_fine_tuning",
        "maccabi1080i_led",
        "munich1080i_led",
    ]

    return repo_ids_to_use
