from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import better_json as bj
from CameraParameters import CameraParameters


# this is memoization of loading the large jsonlines files:
clip_id_to_camera_poses_and_index_offset = dict()


def get_camera_pose_from_clip_id_and_frame_index(
    clip_id,
    frame_index
):
    """
    If you specify a clip_id and a frame_index, this function will return the camera pose at that frame.
    """

    clip_id_to_track_sha256_and_offset = {
        "bay-zal-2024-03-15-mxf-yadif": ("18f36f43eb42e48fc4a0df2450918d6635bff28a002c2ccad66e27d627d936b7", 0),
        "bos-dal-2024-06-09-mxf": ("44c003e82c067a63bbd3806b2a2a55912b52b889e303a7bf7f4b8f1c98abcfa7", 276962),
        "bos-mia-2024-04-21-mxf": ("a5de551ceb1a8770920fc57005dd45cb5e9f3522a77a175110845c733ec1bf87", 359638),
        "dal-bos-2024-01-22-mxf": ("7a9baa5daa022f40705cbc4a139a60b9b837ca82631d2288becbdefa865ba7e6", 71585),
        "dal-bos-2024-06-12-mxf": ("142ca42fbcbf44d0c4b19747b5b113740329dd6b4f9f8702f991341b399fd88d",  355146),
        "slgame1": ("a3d764848dfbd42abc459630a05cfa297b21251c1abe06ae51e2ab029f5e1a51", 0),  # mathieu's newest 2024-07-20
        "slday2game1": ("a97465c7968a83e8b7e4434e6ac5db34d85a62b7880401931c9131afd1830d1b", 0),
        "slday3game1": ("d349bd7c315c402da25ef676181c393352177eff73fa52f39e9d21f8466df50b", 0),
        "slday4game1": ("91750a91a9d7f8af6bcfb5785764e4b92eb89cb0b01c45471e1ad8c36af658d5", 0),
        "slday5game1": ("8ecd2cde593e98b536abdce358be9f2b6e9709f590ac332a9aacd2d4777df39d", 0),
        "slday6game1": ("8c2001da8488ab118b3c4c813c561f02350a5130e4687f4e23574ac138399db6", 0),
        "slday7game1": ("b8dedf1158e942a80580843dcf58b720b4a06bbc3bec0979c1b3d227c1d84547", 0), # maybe d06313adb047b31105543aba162be23ce91a0679cb63cbb686f3e8d6796c100e
        "slday8game1": ("1d02e8dcf506e76468a4bb8745be3d025cd48c56c58bf423dbc7620bcc8e3495", 0),  # chaz redid it.
        # "slday8game1": ("81d5fddcaa9969638276e3b0696d48edb6d9faf742901b7eff3127972f80a637", 0),  # chaz did this one
        "slday9game1": ("080911f9b3a002b3d15d6b7e8931474ed90e6118cde8a5eeadbbd0575f627394", 0),
        "slday10game1" : ("becb60b2ce83d0e715b10637206cfcba66423406e990348c769fbd4c9659bf1e", 0),
        "hou-sas-2024-10-17-sdi": ("f31e30f615f91b51e3c64d96876827a0ca4ba431949900855efacace2b02fbae", 0),  # chaz tracked this 2024-10-19
        "hou-was-2024-11-11-sdi": ("f18840b7166b9081ae19ff108327e9a62ddc0164bfef0472da9cd8ab5610986a", 0),
        
        # WASvUTA_PGM_core_nbc_11-12-2022_game_from148409_to609375_50m_out-track_24-11-19_05-23-31":
        "was-uta-2022-11-12-ingest": ("26f9cabe19da71851f9b40a547d24acd9244d56b4217de9d729d3c069e5e7cf6", 0),
        

        # CHIvDEN_PGM_core_nbc_11-13-2022_game_from143971_to596217_50m_out-track_24-11-19_16-18-31.ts
        "chi-den-2022-11-13-ingest": ("5d9c02c50ef1ffea3ba5d80703418280af12f1ca7365f09daf1b7432c3536fc9", 0),  

        # ATLvBOS_PGM_core_esp_11-16-2022_game_from162085_to604850_50m_out-track_24-11-19_19-40-03.ts
        "atl-bos-2022-11-16-ingest": ("ecddaf0e40b8743d37a99817a695eec5877a1f8111a3feceaaf674ac91f1a395", 0),
    }
    assert (
        clip_id in clip_id_to_track_sha256_and_offset
    ), f"{clip_id=} not in {clip_id_to_track_sha256_and_offset=}, suggest you edit get_camera_pose_from_clip_id_and_frame_index.py"

    if clip_id not in clip_id_to_camera_poses_and_index_offset:
        track_sha256, index_offset = clip_id_to_track_sha256_and_offset[clip_id]
        
        # print(f"{track_sha256=}")

        jsonlines_path = get_file_path_of_sha256(
            sha256=track_sha256
        )
        # print("Starting to load the jsonlines file.")
        camera_poses = bj.load_jsonlines(
            jsonlines_path=jsonlines_path
        )
        # print("Finished loading the jsonlines file.")


        clip_id_to_camera_poses_and_index_offset[clip_id] = (camera_poses, index_offset)
    
    camera_poses, index_offset = clip_id_to_camera_poses_and_index_offset[clip_id]

    mathieu_frame_index = frame_index - index_offset
    
    camera_pose = camera_poses[mathieu_frame_index]
    
    camera_pose = CameraParameters(
        rod=camera_pose["rod"],
        loc=camera_pose["loc"],
        f=camera_pose["f"],
        ppi=0.0,
        ppj=0.0,
        k1=camera_pose["k1"],
        k2=camera_pose["k2"],
    )

    return camera_pose
