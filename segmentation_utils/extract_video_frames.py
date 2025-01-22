from extract_single_frame_from_video import extract_single_frame_from_video
import subprocess
from pathlib import Path

# write_dir = "SL_2022_00"
# in_path_video = Path("/mnt/nas/volume1/videos/nba/SummerLeague/SL00_0001309096_NBA202200021184430001r.mxf").expanduser()
write_dir = "SL_2022_02"
in_path_video = Path("/mnt/nas/volume1/videos/nba/SummerLeague/SL02_0001309141_NBA202200021184450001r.mxf").expanduser()
out_path = Path(f"/mnt/nas/volume1/videos/frame_samples/{write_dir}").expanduser()

for i in range(155200, 600000):
    out_path_frame = out_path / f"{write_dir}_{i}.jpg"
    extract_single_frame_from_video(abs_video_file_path=in_path_video, frame_index_to_extract=i, frame_abs_out_path=out_path_frame)
    # cmds = [
    #     "extract_single_frame_from_video",
    #     f"{in_path}",
    #     f"{i}",
    #     f"{out_path}"
    # ]
    # subprocess.run(cmds)