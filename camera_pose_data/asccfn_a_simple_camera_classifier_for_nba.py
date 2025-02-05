def asccfn_a_simple_camera_classifier_for_nba(
    x: float,
    y: float,
    z: float,
):
    """
    This function classifies a camera pose as being one of the types
    of NBA camera: C01, C02, NETCAM_LEFT, NETCAM_RIGHT, or SPIDER.
    """
    if y < -102.0:  # it is either C01 or C02:
        if x > 3.0:
            return "C02"
        else:
            return "C01"
    else:  # it is either NETCAM_LEFT or NETCAM_RIGHT or SPIDER:
        if z > 22.0:
            return "SPIDER"
        else:  # it is either NETCAM_LEFT or NETCAM_RIGHT:
            if x < 0.0:
                return "NETCAM_LEFT"
            else:
                return "NETCAM_RIGHT"