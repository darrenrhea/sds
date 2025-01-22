import subprocess
import os
from pathlib import Path, PurePath
import sys


video_ids = [
    # "35a61d80-72d6-4dd8-ba6d-df84207dc6d1",
    # "a338ca9e-0c54-4b32-a1f4-891560760b3f",
    # "73899023-679e-4a7a-9196-0e195a6c4b7c",
    # "e150850b-647d-4be3-8592-53dd65ac558f",
    # "3adb3fd6-6210-453b-a91c-ba7b97870654", brooklyn city gray court
    # "9b5a57e5-f798-4e4d-b032-2beb98f940b2",
    # "dea14b46-e003-4d85-b532-89d34aed882c",
    # "5b7ce295-510b-48c4-aec7-fef60c6cdb1c",
    # "726ee95f-97e8-4fa6-8046-66c52fcea89f",
    # "e03f6534-652b-45e5-913f-a716b105f3b5",
    # "31e145a3-6f0f-4046-9f38-c6df8aa6ddc6",
    # "2cc07cd6-4964-4893-85c1-f4ee0f609361",
    # "f148723e-7199-40f4-acd7-a1df3e870fdf",
    # "fefceb75-a6f7-4d2e-8a22-f2da2b7f2ecd",
    "67dde4f6-b303-456c-97aa-a7ca1142fba5",
    "2062d907-8f6c-48a4-9396-e906f01a259f",
    "d005a5af-461d-4aa0-852b-99a45d289d69",
    "0477d8ad-3f08-4d56-8527-807511661524",
    "ceefd8b9-f123-46e6-ab7e-ebec7ed3da8c",
    "1e0c187d-ea57-4b2d-9167-aeed85df4394",
    "f03d906a-55f6-4a1a-a676-7fa4fe9273e9",
    "ad3522d3-fb14-4a54-a802-894717e18c54",
    "26768d32-e754-439d-9728-8010c1f2f136",
    "2e7e7b2e-f406-4d17-8c98-ab0d6212cabf",
    "0c0f070b-4e37-428d-8981-a4d987ca1e39",
    "6145b011-b383-4fc8-90b8-744d84a23849",
    "c47e6c27-6186-498a-ba0e-bfb6646c635c",
    "83d6674a-69c2-4eff-8c35-ed6b92a72b66",
    "3d1f3e9c-3450-44b7-bad6-a7bb432cdf40",
    "31bfaa44-0824-472d-8c0c-f95d84759dd6",
    "7eb66122-a6a9-45d4-8c62-d39e78ac55fd",
    "6e982ce8-3588-4c97-8818-f22c8178e0c5",
    "ab2fe3ce-0263-4671-834d-654e7ea28d11",
    "149ca0c0-0917-47cb-9a38-817273b8433d",
    "95a1e72b-9a77-4592-8bd5-17d7b7c29d95",
    "e4839041-b017-4fee-97a7-8d2568b94504",
    "bcfc1b0c-74dc-4f90-8373-f2db8bf03418",
    "0de0562d-e9da-4b8d-9352-5a912d0371c5",
    "54eee6bf-a903-4c04-bd9c-6fe98584b422",
    "130e4413-a1a2-48c6-80aa-efe9749a3b91",
    "3be395fd-3551-446a-8d2f-cdf0a9567a66",
    "bdc2798c-c66d-498e-b12c-6167d448f977",
    "3628b9db-36ba-4e0a-ba0c-50033ed42bc1",
    "b30e8b73-013d-4d52-a760-df704a7da5ec",
    "59821995-4a61-4a3f-b5aa-02bf2b209dad",
    "1b8e96a5-0a85-4904-81cb-890563a443dc",
    "41387e3f-0788-4b67-b1c7-4f38b920234c",
    "297ed7e5-080a-40c9-9972-bd7cdb59f5f6",
    "5aff4c24-6a93-494e-b5e3-28e07fa823eb",
    "955f3952-bf3d-4399-94cd-fd738a856463",
    "f553ad71-c1c2-4f02-acb7-aec02e61e056"
]

frame_blowout = 0
infer = 0
make_video = 0
downsample_originals = 1
downsampled_inference = 0

# frame_samples_path = Path("/mnt/nas/volume1/videos/porridge").expanduser()
frame_samples_path = Path("/mnt/nas/volume1/videos/frame_samples").expanduser()
# model_cookie_id = "28c6-702f-89ab-17d6-no-overlap" # gray_400p_400p_res34_13e_41f_half_res.pth
# model_cookie_id = "8ffe-b4f0-18a7-7298-no-overlap" # gray_400p_400p_res34_18e_41f_full_res.pth
model_cookie_id = "c97a-c300-93d3-8459-no-overlap" # wood_generic_400p_400p_res34_55e_178f_full_res.pth
# model_cookie_id = "74a2-cbc8-6775-133b-no-overlap" # wood_light_normal_400p_400p_res34_13e_423f_half_res.pth
# model_cookie_id = "4fce-1b89-497d-505c-no-overlap" # wood_light_normal_400p_400p_res34_34e_423f_full_res.pth 
# video_id = "hlt_22-23_BKN_city_Awecom_WSC_BKN"

for video_id in video_ids:
    original_frames_dir = frame_samples_path / video_id
    original_frames_dir_movie = frame_samples_path / (video_id + ".mp4")
    out_dir = original_frames_dir / model_cookie_id
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_dir = Path("~/r/infer_frames").expanduser()
    downsample_width = 1920//2
    downsample_height = 1080//2
    downsample_subdir = f"{downsample_width}x{downsample_height}"
    downsample_originals_dir = original_frames_dir / downsample_subdir

    if frame_blowout:
        args = [
            "time",
            "/usr/local/bin/ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            f"{original_frames_dir_movie}",
            "-vsync",
            "0",
            "-q:v",
            "2",
            "-start_number",
            "0",
            "-t",
            "11",
            f"{frame_samples_path / video_id}/{video_id}_%06d.jpg"
        ]
        subprocess.run(args)

    if downsample_originals:
        downsample_originals_dir.mkdir(parents=True, exist_ok=True)
        for full_frame_path in original_frames_dir.iterdir()[0:3600]:
            if full_frame_path.is_file():
                frame_name = full_frame_path.name
                print(f"video name {frame_name}")
                print(f"{original_frames_dir}/{frame_name}")
                print(f"{downsample_originals_dir}/{frame_name}")
                args = [
                    "convert",
                    f"{original_frames_dir}/{frame_name}",
                    "-resize",
                    f"{downsample_width}x{downsample_height}",
                    f"{downsample_originals_dir}/{frame_name}"
                ]
                subprocess.run(args)
    if infer:
        if downsampled_inference:
            infer_args = [
                "python",
                f"{infer_dir}/flexible_caller.py",
                f"{infer_dir}/A5000_0.jsonc",
                f"{downsample_originals_dir}",
                f"{out_dir}",
                "False",
                f"{model_cookie_id}"
            ]
        else:
            infer_args = [
                "python",
                f"{infer_dir}/flexible_caller.py",
                f"{infer_dir}/A5000_0.jsonc",
                f"{original_frames_dir}",
                f"{out_dir}",
                "False",
                f"{model_cookie_id}"
            ]

        subprocess.run(infer_args)
    if make_video:
        first_frame_index = 0
        last_frame_index = 659
        outfile = f"{out_dir}/{model_cookie_id}_{video_id}_from_{first_frame_index}_to_{last_frame_index}.mp4"
        num_frames= last_frame_index - first_frame_index + 1 

        if downsampled_inference:
            args = [
                "time",
                "/usr/local/bin/ffmpeg",
                "-y",
                "-start_number",
                f"{first_frame_index}",
                "-framerate",
                "59.94",
                "-i",
                f"{out_dir}/{video_id}_%06d_{model_cookie_id}.png",
                "-start_number",
                f"{first_frame_index}",
                "-framerate",
                "59.94", 
                "-i",
                f"{downsample_originals_dir}/{video_id}_%06d.jpg",
                "-frames",
                f"{num_frames}",
                "-filter_complex",
                f"[1][0]alphamerge[fg];[1]eq=contrast=1.0:brightness=0.0[lowcont];[lowcont]colorchannelmixer=.0:.0:.0:.0:.3:.4:.3:.0:.0:.0:.0:.0[green];[green][fg]overlay[final];[final]drawtext=fontfile=/awecom/misc/arial.ttf: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuvj420p",
                "-crf",
                "18",
                f"{outfile}"
            ]
        else:
            args = [
                "time",
                "/usr/local/bin/ffmpeg",
                "-y",
                "-start_number",
                f"{first_frame_index}",
                "-framerate",
                "59.94",
                "-i",
                f"{out_dir}/{video_id}_%06d_{model_cookie_id}.png",
                "-start_number",
                f"{first_frame_index}",
                "-framerate",
                "59.94", 
                "-i",
                f"{original_frames_dir}/{video_id}_%06d.jpg",
                "-frames",
                f"{num_frames}",
                "-filter_complex",
                f"[1][0]alphamerge[fg];[1]eq=contrast=1.0:brightness=0.0[lowcont];[lowcont]colorchannelmixer=.0:.0:.0:.0:.3:.4:.3:.0:.0:.0:.0:.0[green];[green][fg]overlay[final];[final]drawtext=fontfile=/awecom/misc/arial.ttf: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuvj420p",
                "-crf",
                "18",
                f"{outfile}"
            ]

        print(" \\\n".join(args))

        subprocess.run(args)
        print(f"on laptop do:\n\ncd ~/show_n_tell\nrsync -P lam:{outfile} .\nopen {Path(outfile).name}\n")