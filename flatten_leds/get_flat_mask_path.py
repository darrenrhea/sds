from pathlib import Path


def get_flat_mask_path(
    final_model_id: str,
    clip_id: str,
    frame_index: int,
    rip_width: int,
    rip_height: int,
    board_id: str,
) -> Path:
    """
    To what file path on the hard drive do we want to save
    the inference of a particular final segmentation model
    on a particular frame_index
    of a particular clip_id
    of a particular ad board
    flattened to a particular rip_width and rip_height?
    The "primary key" here is a big cartesian product,
    so there really is no canonical way to do this, only a convention.
    """
    assert isinstance(clip_id, str)
    assert isinstance(frame_index, int)
    assert isinstance(final_model_id, str)
    assert isinstance(rip_width, int)
    assert isinstance(rip_height, int)
    assert isinstance(board_id, str)
    shared_dir =  Path(
        f"/shared"
    )
   
    output_dir = shared_dir / "clips" / clip_id / "flat" / board_id / f"{rip_width}x{rip_height}" / "masks" / final_model_id

    flat_mask_path = output_dir / f"{clip_id}_{frame_index:06d}_{final_model_id}.png"

    # we don't want to assert it exists, because writers of the mask also use
    # this function to know where to save

    return flat_mask_path


def test_get_flat_mask_path():
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 442105
    rip_width = 1856
    rip_height = 256
    final_model_id = "effs94"
    flat_mask_path = get_flat_mask_path(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_width=rip_width,
        rip_height=rip_height,
        final_model_id=final_model_id,
    )
    assert flat_mask_path.exists(), f"{flat_mask_path=} does not exist"

    assert flat_mask_path.exists()
    print(f"{flat_mask_path=}")

if __name__ == "__main__":
    test_get_flat_mask_path()
    print("get_flat_mask_path PASSED")