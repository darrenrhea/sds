from get_game_id_from_video_source_file_name import (
     get_game_id_from_video_source_file_name
)

import textwrap


def test_get_game_id_from_video_source_file_name_1():
    video_source_file_names = [
        "EB_23-24_R01_BAY-BER.mxf",
        "EB_23-24_R20_BAY-RMB.mxf",
        "EB_23-24_R27_BAY-CZV.mxf",
        "EB_23-24_R29_BAY-ZAL.mxf",
        "EB_23-24_R31_BAY-MTA.mxf",
        "EB_23-24_R32_BAY-BAR.mxf",
        "EB_23_24_R15_BAY-EFS.mxf",
    ]

    for file_name in video_source_file_names:
        game_id = get_game_id_from_video_source_file_name(
            file_name=file_name
        )
        print(
            textwrap.dedent(
                f"""\
                
                video_source_file_name = {file_name} has
                game_id =                {game_id}


                """
            )
        )

if __name__ == "__main__":
    test_get_game_id_from_video_source_file_name_1()