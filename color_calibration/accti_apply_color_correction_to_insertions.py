from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import pyperclip
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from show_color_correction_result_on_insertion_description_id import (
     show_color_correction_result_on_insertion_description_id
)
from pathlib import Path


def accti_apply_color_correction_to_insertions():
    """
    Suppose you have already saved a color correction polynomial to json.

    This function accti will load the color correction polynomial from the json,

    and apply it to the self-reproducing-insertions listed here.

    At which point, you can flipflip between an original and the reproduction-of-reality.

    Ideally the reproduction-of-reality will be indistinguishable from the original.
    """
    insertion_description_ids = [
        "0e4d1982-baeb-4b04-8c18-774f3bce4084",  # Playoffs_Title  NBA Playoffs Presented by Google Pixel, bos-mia-2024-04-21-mxf_365000
        "41bb4d0b-d2cf-4d15-8482-abb6493520ba",  # Playoffs_Title  NBA Playoffs Presented by Google Pixel, bos-mia-2024-04-21-mxf_365500
        "e8d4d8d2-0409-4eb1-a993-f0a7d8ce58ab",  # NBA_APP_MSFT       get the App powered by microsoft bos-mia-2024-04-21-mxf_397000
        "60820db1-9028-4566-86a0-84d6056168fb",  # ESPN_DAL_LAC_NEXT_ABC    Black version Playoffs Dallas Mavericks x LA Clippers COMING UP NEXT abc bos-mia-2024-04-21-mxf_412000
        "cbc658ec-a9c3-4f67-8d07-667c03a35f27",  # ESPN_NBA_Finals     abc home of the NBA Finals begin june 6 bos-mia-2024-04-21-mxf_440000
        "fd217680-07d7-4f0f-a8ff-8a2bdf2a25e4",  # different_here    Green Different Here bos-mia-2024-04-21-mxf_471000
        "e4e347df-b49a-4f00-8990-d5d7489b0812",  # NBA_ID      NBA ID Sign up. Score. bos-mia-2024-04-21-mxf_501500
        "c275ba31-0faf-4564-9134-a1e1d03e8805",  # NHL_Playoffs      NHL Stanley Cup playoffs bos-mia-2024-04-21-mxf_539000
        "48964eb8-573e-4455-9db6-e6874c66ef62",  # ESPN_APP GET ESPN+ bos-mia-2024-04-21-mxf_557000
        "2a04b7dd-8d83-4455-927e-002b16b11128",  # ESPN_MIL_IND_FRI     Milwaukee Bucks x Indiana Pacers bos-mia-2024-04-21-mxf_628500
        "20f0eff2-34e0-4921-96d2-7f2b83ff1b7a",  # NBA_Store     NBA Store bos-mia-2024-04-21-mxf_657000
        "1b9f9cd9-0d15-4965-ab74-a6b2626dbd23",  # Playoffs_PHI_NYK_TOM_TNT      NBA Playoffs 76ers versus Knicks bos-mia-2024-04-21-mxf_712000
        "c93bd561-f628-4043-86d0-9c601ce23993",  # PickEm_Bracket_Challenge Pickem Bracket Challenge Play now bos-mia-2024-04-21-mxf_770500
        "7466b2fe-0b71-437e-a5d4-5189f968469c",  # Playoffs_DAL_LAC_NEXT_ABC white version of Playoffs Dallas Mavericks x LA Clippers NEXT
    ]
    
    # color_correction_sha256 = "bd545cba8ac10558b8a5a4eeba40bc3be9f1e809975fd7e6ad38d6a3ac598140"
    
    # what was used for the past two days:
    color_correction_sha256 = "4edceff5771335b7a64b1507fa1d31f38f5148f71322092c4db5ecd8ec6e985b"
    
    color_correction_json_path = get_file_path_of_sha256(color_correction_sha256)
    print(f"loading color correction from {color_correction_json_path}")
    
    degree, coefficients = load_color_correction_from_json(
        json_path=color_correction_json_path
    )
    
    out_dir = Path(
        "~/temp"
    ).expanduser()
    
    out_dir.mkdir(exist_ok=True, parents=True)

    for insertion_description_id in insertion_description_ids:
        print(f"Doing {insertion_description_id}")
        show_color_correction_result_on_insertion_description_id(
            insertion_description_id=insertion_description_id,
            degree=degree,
            coefficients=coefficients,
            out_dir=out_dir,
        )
   
    s = "flipflop ~/temp"
    pyperclip.copy(s)
    print("We suggest you run the following command:")
    print(s)
    print("you can just paste since it is on the clipboard")
 

if __name__ == "__main__":
    accti_apply_color_correction_to_insertions()