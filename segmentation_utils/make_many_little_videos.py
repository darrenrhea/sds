import pprint as pp
import sys
from pathlib import Path
from make_temporary_ad_insertion_config_file import (
    make_temporary_ad_insertion_config_file,
)
import subprocess


# These are in Priebe numbers:
# things = [
#     [149261, 149276, "segmenting shadows on sam adams"],
#     [158294, 158335, "number 31 screwing up chime shorts flicker legs flicker"],
#     [162815, 162852, "major shorts and legs flickering of 31 over sam adams"],
#     [163664, 163756, "ref getting impaled by awecom"],
#     [
#         165466,
#         165539,
#         "chime shorts flickering keeps happening intermittently for a few dozen frames after this range",
#     ],
#     [166257, 166296, "intermittent leg and shorts flickering"],
#     [167131, 167227, "ref struggling with awecom overwriting his world"],
#     [167551, 167617, "white cav players shoulder flickering on sam adams"],
#     [170754, 170769, "sam adams shorts and hands flickering"],
#     [170889, 170908, "shorts flickering over sam adams"],
#     [173516, 173522, "shorts edge flickering over sam adams"],
#     [176266, 176352, "shorts flickering on sam adams"],
#     [176398, 176417, "rough segmentation edges slash spikes ripping into sam adams"],
#     [180675, 180752, "shorts flickering over sam adams"],
#     [182385, 182393, "dr pepper segmented for no apparent reason"],
#     [183912, 183941, "sam adams flickering around shoes"],
#     [188155, 188253, "TD impaling player leg"],
#     [188911, 188911, "shorts flicker around here on sam adams"],
#     [189136, 189189, "more shorts flickering at sam adams"],
# ]

things = [
    [270960, 270961, "dr_pepper_overwriting_somebody"],
    [568961, 568976, "segmenting shadows on sam adams"],
    [577994, 578035, "number 31 screwing up chime shorts flicker legs flicker"],
    [582515, 582552, "major shorts and legs flickering of 31 over sam adams"],
    [583364, 583456, "ref getting impaled by awecom"],
    [585166, 585239, "chime shorts flickering keeps happening intermittently for a few dozen frames after this range"],
    [585957, 585996, "intermittent leg and shorts flickering"],
    [586831, 586927, "ref struggling with awecom overwriting his world"],
    [587251, 587317, "white cav players shoulder flickering on sam adams"],
    [590454, 590469, "sam adams shorts and hands flickering"],
    [590589, 590608, "shorts flickering over sam adams"],
    [593216, 593222, "shorts edge flickering over sam adams"],
    [595966, 596052, "shorts flickering on sam adams"],
    [596098, 596117, "rough segmentation edges slash spikes ripping into sam adams"],
    [600375, 600452, "shorts flickering over sam adams"],
    [602085, 602093, "dr pepper segmented for no apparent reason"],
    [603612, 603641, "sam adams flickering around shoes"],
    [607855, 607953, "TD impaling player leg"],
    [608600, 608620, "shorts flicker around here on sam adams"],
    [608836, 608889, "more shorts flickering at sam adams"],
]



gpu_index = 0
for model_name in ["fastai_29e_66f"]:

    output_masking_attempt_id = "temp"

    for thing in things:
        
        assert thing[0] <= thing[1]
        first_frame_index = thing[0] - 60
        last_frame_index = thing[1] + 60
        if first_frame_index < 351367:
            tracking_attempt_id= "blend_first_half"
        else:
            tracking_attempt_id= "blend_second_half"

        name_of_short_video = thing[2].replace(" ", "_")

        if model_name != "old":  # no need to infer for old, it is whatever is in final_bw
            inference_command_pieces = [
                "python",
                "run_stitcher.py",
                "gsw1",
                f"{first_frame_index}",
                f"{last_frame_index}",
                f"{gpu_index}",
                model_name,
                output_masking_attempt_id,
            ]

            print("To infer the foreground/background segmentation, we do this:")
            print(" ".join(inference_command_pieces))

            working_directory_for_inference = Path("~/r/segmentation_utils").expanduser()

            ans = subprocess.run(
                inference_command_pieces,
                capture_output=False,
                cwd=working_directory_for_inference,
            )
        if model_name == "old":
            ad_insertion_masking_attempt_id="final_bw"
            insertion_attempt_id="temp_for_old_little_videos"
        else:
            ad_insertion_masking_attempt_id = output_masking_attempt_id
            insertion_attempt_id="temp_for_little_videos"

        ad_insertion_config_file_path = make_temporary_ad_insertion_config_file(
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            tracking_attempt_id=tracking_attempt_id,
            masking_attempt_id=ad_insertion_masking_attempt_id,
            insertion_attempt_id=insertion_attempt_id
        )

        print(f"bat {ad_insertion_config_file_path}")

        ad_insertion_command_pieces = [
            "bin/ad_insertion",
            str(ad_insertion_config_file_path),
        ]

        print("To actually insert ads, we run this:")
        print(" ".join(ad_insertion_command_pieces))

        working_directory_for_ad_insertion = Path(
            "~/felix/ad_insertion_video_maker/build"
        ).expanduser()

        ans = subprocess.run(
            ad_insertion_command_pieces,
            capture_output=False,
            cwd=working_directory_for_ad_insertion,
        )

        ffmpeg_command_pieces = [
            "/usr/local/bin/ffmpeg",
            "-start_number",
            f"{first_frame_index}",
            "-framerate",
            "60",
            "-y",
            "-i",
            f"/home/drhea/awecom/data/clips/gsw1/insertion_attempts/{insertion_attempt_id}/gsw1_%06d_ad_insertion.jpg",
            "-vf",
            f"drawtext=fontfile=Arial.ttf: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-frames",
            f"{last_frame_index - first_frame_index + 1}",
            f"/home/drhea/show_n_tell/little_videos/{name_of_short_video}_{first_frame_index:06d}_{last_frame_index:06d}_{model_name}.mp4",
        ]

        print("To actually ffmpeg it into a video, we run this:")
        print(" ".join(ffmpeg_command_pieces))

        ans = subprocess.run(
            ffmpeg_command_pieces,
            capture_output=False,
        )

        print("cd ~/show_n_tell")
        print("mkdir -p little_videos")
        print(
            "mkdir -p ~/show_n_tell/little_videos/",
            "rsync -r --progress lam:/home/drhea/show_n_tell/little_videos/ ~/show_n_tell/little_videos/"
        )
