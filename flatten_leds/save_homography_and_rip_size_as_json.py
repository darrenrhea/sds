from pathlib import Path
import better_json as bj


def save_homography_and_rip_size_as_json(
        H,
        rip_height: int,
        rip_width: int,
        out_path: Path,
 ):
    """
    We want the way we flatten to infer to match the way we unflatten,
    so we save it for now.
    Move this to elsewhere later.
    """
    jsonable = dict(     
        homography_in_pixel_units=[
            [float(H[i, j]) for j in range(3)]
            for i in range(3)
        ],
        rip_height=rip_height,
        rip_width=rip_width,
    )
    bj.dump(fp=out_path, obj=jsonable)

