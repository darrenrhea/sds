def create_ij_displacement_and_weight_pairs(
    radius: int,
):
    """
    Needed for blur_both_original_and_mask_u8
    """
    assert radius >= 0
    ij_displacement_and_weight_pairs = []
    for j in range(-radius, radius + 1):
        ij_displacement_and_weight_pairs.append(
            (
                (0, j),
                1.0,
            )
        )
    return ij_displacement_and_weight_pairs
