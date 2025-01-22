from open_as_hwc_rgb_np_uint8 import (
     open_as_hwc_rgb_np_uint8
)
import subprocess
from pathlib import Path
import numpy as np
import argparse


def extract_jpeg(
     input_video_abs_file_path,
     out_frame_abs_file_path
):
    """
    Trying to make JPEG frame extraction come out not too dark!!!???
    """
    assert out_frame_abs_file_path.suffix == ".jpg"
    args = [
        "/opt/homebrew/bin/ffmpeg",
        "-report",
        "-y",
        "-nostdin",
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
         "-vsync",
        "0",
        "-pix_fmt",
        "yuvj420p",
        "-q:v",
        "2",
        "-qmin",
        "2",
        "-qmax",
        "2",
        # "-src_range",
        # "0",
        # "-dst_range",
        # "1",
        "-frames:v",
        "1",
        # "-filter:v",
        # "scale=1920:-1:out_color_matrix=bt601:out_range=pc",
        "-update",
        "1",
        str(out_frame_abs_file_path)
    ]

    print(" \\\n".join(args))
    subprocess.run(args=args)
    print(f"pri {out_frame_abs_file_path}")


def extract_png(
     input_video_abs_file_path,
     out_frame_abs_file_path
):
    """
    Or is this PNG extraction too bright?
    """

    args = [
        "/opt/homebrew/bin/ffmpeg",
        "-report",
        "-y",
        "-nostdin",
        "-i",
        str(input_video_abs_file_path),
        "-vf",
        # "scale=1920:1080,zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p",
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,scale=1920:1080"
        "-f",
        "image2",
         "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(out_frame_abs_file_path)
    ]
    print(" \\\n".join(args))
    subprocess.run(args=args)
    print(f"pri {out_frame_abs_file_path}")


if __name__ == "__main__":
    input_video_abs_file_path = Path(
        # "~/temp7/22-23_BOS_CORE_30_second.mp4"
        # "~/a/clips/hou-lac-2023-11-14/hou-lac-2023-11-14_from_152000_to_155999.mp4"
        # "~/a/clips/DSCF0241/videos/DSCF0241_frame_numbers.mp4"
        # "/shared/clips/DSCF0241/videos/DSCF0241_from_000000_to_008549.mp4"
        "/Volumes/videos/nba/SummerLeague/SL00_0001309096_NBA202200021184430001r.mxf"
    ).expanduser()
    
    temp_dir = Path("temp").resolve()
    temp_dir.mkdir(exist_ok=True)

    extract_jpeg(
        input_video_abs_file_path=input_video_abs_file_path,
        out_frame_abs_file_path=temp_dir / "out.jpg"
    )

    extract_png(
        input_video_abs_file_path=input_video_abs_file_path,
        out_frame_abs_file_path=temp_dir / "out.png"
    )

    x = open_as_hwc_rgb_np_uint8(temp_dir / "out.png")
    y = open_as_hwc_rgb_np_uint8(temp_dir / "out.jpg")
    z = open_as_hwc_rgb_np_uint8(
        Path("~/r/ffmpeg_book/images/vlcsnap-00001.jpg").expanduser()
    )
    print(x.astype(np.double).mean())
    print(y.astype(np.double).mean())
    print(z.astype(np.double).mean())

   