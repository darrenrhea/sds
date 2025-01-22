import pprint as pp

from prii import (
     prii
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from pathlib import Path


from get_clip_id_and_frame_index_from_file_name import (
     get_clip_id_and_frame_index_from_file_name
)

def check_enumeration_cli_tool():
    # make a list of abs_file_paths, clip_id, frame_index
    # for each, get the color by calling get_video_frame_path_from_clip_id_and_frame_index
    # check if the color is essentially correct where alpha is 255
    
    folder = Path.cwd()
    print(folder)

    
    paths = (
        [
            x for x in folder.glob('bos-dal-2024-06-06-srt*.png')
        ]
        +
        [
            x for x in folder.glob('bos-dal-2024-06-06-srt*.jpg')
           
        ]
    )


    triplets = []

    for p in paths:
        clip_id, frame_index = get_clip_id_and_frame_index_from_file_name(p.name)
        triplets.append((clip_id, frame_index, p))

    pp.pprint(triplets)
    
    for clip_id, frame_index, png_path in triplets:
        if frame_index > 239939:
            continue
    
        color_source = get_video_frame_path_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
            force_redownload=False,
        )
        dst = folder / color_source.name
        # if not dst.exists():
        #     shutil.copy(
        #         src=color_source,
        #         dst=dst
        #     )

       

        rgb = open_as_rgb_hwc_np_u8(color_source)
        rgba_cutout = open_as_rgba_hwc_np_u8(png_path)
        print(png_path)
        prii(rgba_cutout)
        print(color_source)
        prii(rgb)




    
