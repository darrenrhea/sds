def get_repo_ids_to_use_for_fake_data(
    floor_id: str,
    segmentation_convention: str,
):
    """
    Often we do not call this but just hard code the repo_ids_to_use.
    We need repos that have LED convention with camera poses.
    """
    assert floor_id in ["munich"]
    assert segmentation_convention in ["led_not_led"]

    repo_ids_to_use = [
        # these first two have the lighting condition:
        "bay-zal-2024-03-15-mxf-yadif_led",
        "bay-mta-2024-03-22-mxf_led",
        "maccabi_fine_tuning",
        "maccabi1080i_led",
        "munich1080i_led",


        # "bay-czv-2024-03-01_led",  # practically empty
        # "bay-efs-2023-12-20_led",
        # "bay-mta-2024-03-22-part1-srt_led",
        # "bay-zal-2024-03-15-yt_led",
        
    ]

    return repo_ids_to_use
