from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from prii import (
     prii
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import shutil
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from pathlib import Path


def fix_cutouts_color_cli_tool():
    valid_clip_ids = [
        "bos-ind-2024-01-30-mxf",
        "bos-mia-2024-04-21-mxf",
        "cle-mem-2024-02-02-mxf",
        "dal-bos-2024-01-22-mxf",
        "dal-lac-2024-05-03-mxf",
        "dal-min-2023-12-14-mxf",
    ]
    folder = Path.cwd()
    print(folder)

    def predicate(s):
        return (
            s[-4:] == ".png"
            and
            all([c in "01234566789" for c in s[-6:-4]])
            and
            all([c in "01234566789" for c in s[-13:-7]])
        )
    
    paths = [
        x for x in folder.glob('*.png')
        if predicate(x.name)
    ]
    triplets = []

    for p in paths:
        print(str(p))
        clip_id = p.name[:-14]
        frame_index = int(p.name[-13:-7])
        print(clip_id)
        print(frame_index)
        if clip_id not in valid_clip_ids:
            print("Invalid clip id")
            continue
        triplets.append((clip_id, frame_index, p))

    for clip_id, frame_index, png_path in triplets:
        color_source = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
            force_redownload=True,
        )

        shutil.copy(
            src=color_source,
            dst=folder
        )

        rgb = open_as_rgb_hwc_np_u8(color_source)
        rgba_cutout = open_as_rgba_hwc_np_u8(png_path)
        rgba_cutout[:, :, :3] = rgb

        write_rgba_hwc_np_u8_to_png(
            rgba_hwc_np_u8=rgba_cutout,
            out_abs_file_path=png_path,
            verbose=False
        )




    
