def source_video_validator(
    info
):
    required_keys = [
        "quality_level",
        "file_name",
    ]
    for key in required_keys:
        if key not in info:
            reason = f"The source_video section does not contain the {key} key"
            return False, reason
    return True, None
  
             