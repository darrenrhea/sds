from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from extract_all_frames_from_video_as_png import (
     extract_all_frames_from_video_as_png
)

from pathlib import Path


video_repo_dir = Path(
    "~/r/munich_led_videos"
).expanduser()

subdir_name_to_width0 = {
    "SKWEEK.COM": 1016,
    "ADIDAS": 1016,
    "DOWNTOWN": 1016,
    "EL CORPORATE": 1016,
    "DENIZBANK": 1024,
}

subdir_name_to_width1 = {
    "DOWNTOWN": 1144,
    "SKWEEK.COM": 1144,
    "ADIDAS": 1136,
    "EL CORPORATE": 1144,
    "DENIZBANK": 1158,

}

subdir_name_to_height = {
    "SKWEEK.COM": 144,
}

subdir_name_to_offset = {
    "TURKISH AIRLINES": 18,
}


description = dict()

subdirs = [x for x in video_repo_dir.iterdir() if x.is_dir() and x.name != ".git"]
print(subdirs)
for subdir in subdirs:
    subdir_name = subdir.name
    if subdir_name != "BKT":
        continue

    description[subdir_name] = dict()

    print(f"Doing {subdir}")
    video_files = [x for x in subdir.iterdir() if x.is_file() and x.suffix == ".mp4"]
    print(video_files)
    for video_file in video_files:
        if video_file.name == "Bayern-skweek":  # we think this one is wrong
            continue
        print(f"Doing {video_file}")
        new_dir = video_file.with_suffix("")
        new_dir.mkdir(exist_ok=True)
        print(f"Made {new_dir}")
        stackframes_dir = new_dir / "stackframes"
        stackframes_dir.mkdir(exist_ok=True)
        print(f"Made {stackframes_dir}")
        # extract_all_frames_from_video_as_png(
        #     input_video_abs_file_path=video_file,
        #     out_dir_abs_path=stackframes_dir
        # )

        for image_index in range(100000):
            image_file_path = stackframes_dir / f"{image_index:05d}.png"
            if not image_file_path.exists():
                break
            offset = subdir_name_to_offset.get(subdir.name, 0)
            height = subdir_name_to_height.get(subdir.name, 144)
            widths = [0, 0]
            widths[0] = subdir_name_to_width0.get(subdir.name, 1024)
            widths[1] = subdir_name_to_width1.get(subdir.name, 1152)

            print(f"Doing {image_file_path}")
            print(f"{offset=}, {widths[0]} x {height}")
            print(f"{offset=}, {widths[1]} x {height}")
            rgb = open_as_rgb_hwc_np_u8(image_file_path)
            top =  rgb[offset:offset + height, :widths[0], :]
            crap = rgb[offset:offset + height, widths[0]:, :]
            bottom = rgb[offset + height:offset + 2*height, :widths[1], :]
            crap2  = rgb[offset + height:offset + 2*height, widths[1]:, :]

            prii(top, caption="top")
            prii(crap, caption="crap")
            prii(bottom, caption="bottom")
            prii(crap2, caption="crap2")

            for i, rgb_hwc_np_u8 in [(0, top), (1, bottom)]:
                frames_dir = new_dir / f"{widths[i]}x{height}"
                frames_dir.mkdir(exist_ok=True)
                save_path = frames_dir / f"{image_index:05d}.png"
                write_rgb_hwc_np_u8_to_png(rgb_hwc_np_u8=rgb_hwc_np_u8, out_abs_file_path=save_path)





# filepath=$1

# filename=$(basename "$filepath") 
# filenameStem="${filename%.*}" 

# ffmpeg -i "${filepath}" \
# -filter_complex "\
#   [0:v]crop=1024:144:0:0[out1]; \
#   [0:v]crop=1152:144:0:144[out2]; \
#   [0:v]crop=1024:144:0:288[out3]; \
#   [0:v]crop=1024:144:0:432[out4]; \
#   [0:v]crop=1152:144:0:576[out5]; \
#   [0:v]crop=1024:144:0:720[out6]" \
# -map "[out1]" ${filenameStem}_1.mp4 \
# -map "[out2]" ${filenameStem}_2.mp4 \
# -map "[out3]" ${filenameStem}_3.mp4 \
# -map "[out4]" ${filenameStem}_4.mp4 \
# -map "[out5]" ${filenameStem}_5.mp4 \
# -map "[out6]" ${filenameStem}_6.mp4
