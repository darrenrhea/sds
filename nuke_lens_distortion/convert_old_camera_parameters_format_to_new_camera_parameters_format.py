def convert_old_camera_parameters_format_to_new_camera_parameters_format(old: dict) -> dict:
    rodrigues_x = old["rodrigues_x"]
    rodrigues_y = old["rodrigues_y"]
    rodrigues_z = old["rodrigues_z"]
    camera_location_x = old["camera_location_x"]
    camera_location_y = old["camera_location_y"]
    camera_location_z = old["camera_location_z"]
    focal_length = old["focal_length"]
    k1 = old["k1"]
    k2 = old["k2"]
    k3 = old["k3"]

    new_camera_parameters = dict(
        rod=[rodrigues_x, rodrigues_y, rodrigues_z],
        loc=[camera_location_x, camera_location_y, camera_location_z],
        f=focal_length,
        k1=k1,
        k2=k2,
        k3=k3,
    )
    return new_camera_parameters


