import argparse
import textwrap
from validate_one_camera_pose import (
     validate_one_camera_pose
)
import pprint as pp

from prii import (
     prii
)

def validate_camera_poses_cli_tool():
    """
    This is a tool to check if the a track, i.e. the camera poses,
    are correct.
    """
    argparser = argparse.ArgumentParser(
        usage=textwrap.dedent(
            """\
            do something like:

                validate_camera_poses -c hou-sas-2024-10-17-sdi -a 146000 -b 300000 -s 500
                validate_camera_poses -c hou-sas-2024-10-17-sdi -a 146000 -b 300000 -s 500

                 
                validate_camera_poses -c was-uta-2024-11-12-ingest -a 0 -b 2000 -s 100
                validate_camera_poses -c chi-den-2022-11-13-ingest -a 0 -b 2000 -s 100
                validate_camera_poses -c atl-bos-2022-11-16-ingest -a 0 -b 2000 -s 100

            .
            """
        )
    )
    argparser.add_argument(
        "-c", "--clip_id",
        type=str,
        help="clip_id",
        required=True,
    )
    argparser.add_argument(
        "-a", "--first_frame_index",
        type=int,
        help="first_frame_index",
        default=200000
    )
    argparser.add_argument(
        "-b", "--last_frame_index",
        type=int,
        help="last_frame_index",
        default=300000
    )
    argparser.add_argument(
        "-s", "--frame_index_step",
        type=int,
        help="frame_index_step",
        default=10000
    )
        
    args = argparser.parse_args()
    clip_id = args.clip_id
    first_frame_index = args.first_frame_index
    last_frame_index = args.last_frame_index
    frame_index_step = args.frame_index_step

    # repo_name = f"{clip_id}_led"

    # approvals = bj.load(
    #     f"~/r/{repo_name}/approvals.json5"
    # )

    # approved = approvals["approved"]



    # base_names = [
    #     x.split("/")[1] for x in approved
    # ]

    # pp.pprint(base_names)

    # clip_id_frame_index_pairs = [
    #     (
    #         x.split("_")[0],
    #         int(x.split("_")[1]),
    #     )
    #     for x in base_names
    # ]
    # clip_id_frame_index_pairs = sorted(clip_id_frame_index_pairs, key=lambda x: x[1])

    frame_indices = range(
        first_frame_index,
        last_frame_index + 1,
        frame_index_step
    )


    clip_id_frame_index_pairs = [
        (
            clip_id,
            frame_index,
        )
        for frame_index in frame_indices
    ]

    pp.pprint(clip_id_frame_index_pairs)


    for clip_id, frame_index in clip_id_frame_index_pairs:
        print(f"{clip_id=} {frame_index=}")

        drawn_on = validate_one_camera_pose(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        prii(
            drawn_on,
            caption="the landmarks better line up or the camera pose is wrong:"
        )
