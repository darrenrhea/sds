from attempt_to_annotate_a_cutout import (
     attempt_to_annotate_a_cutout
)
import argparse
from color_print_json import (
     color_print_json
)
from colorama import Fore, Style
import textwrap
from pathlib import Path
import better_json as bj


 
def asoc_annotate_scale_on_cutouts_cli_tool():
    # we tend to scale annotate in "monochromatic batches",
    # usually a folder filled with cutouts of the same kind, team, league, and uniform colors.
    # this way the metadata of each cutout is the same except for the bottom_of_foot and six_feet_above_that.
    argp = argparse.ArgumentParser(
        description="annotate the scale of a bunch of cutouts of the same kind, team, league, and uniform colors",
        usage=textwrap.dedent(
            """\
            python annotate_scale_on_cutouts.py nba ~/r/boston_cutouts_approved/green boston-celtics_green_1
            """
        )
    )
    argp.add_argument("league_id", type=str)
    argp.add_argument("dir_of_the_same_jersey", type=str)
    argp.add_argument("jersey_id", type=str)
    opt = argp.parse_args()
    league_id = opt.league_id
    assert league_id in ["nba", "euroleague", "mlb"], f"{league_id=} is not supported"
    jersey_id = opt.jersey_id
    dir_of_one_kind = Path(opt.dir_of_the_same_jersey).resolve()

  
  
    # go to https://www.euroleaguebasketball.net/en/euroleague/game-center and then click on a game by the team
    # to get the official team name.
    # team = "panathinaikos-aktor-athens"
   
    jersey_file_path = Path(f"~/r/jersey_ids/{jersey_id}.json5").expanduser()

    print(f"Based on {jersey_file_path=!s}")
    jersey = bj.load(jersey_file_path)
    color_print_json(jersey)
    ans = input("You are about to label all the guys in this folder as wearing this uniform / outfit (N/s/y)")
    if ans.lower() != "y":
        print("exiting")
        exit(0)

    # note that we get the kind from the jersey these days, because in baseball,
    # pitchers and batters and catchers and umpires have different kinds of xy point annotations.
    kind = jersey["kind"]  
    league = jersey["league_id"]
    

    if league in ["nba", "euroleague"]:
        assert kind in ["referee", "player", "coach", "ball"]
    else:
        assert kind in ["pitcher", "batter", "umpire", "baseball"]
    assert (
        kind != "ball"
    ), "Use annotate_scale_on_balls.py for balls, not this script." 

    # Set the subfolder of munich_cutout_approveds they come from:

    # arguably meaningful for even coaches and referees:
   
    assert dir_of_one_kind.is_dir(), f"{dir_of_one_kind} does not exist."



    cutout_paths = list(
        dir_of_one_kind.glob("*.png")
    )

    print(f"Going to annotate {len(cutout_paths)} cutouts of {kind=}, namely:")
    
    for cutout_path in cutout_paths:
        print(cutout_path)

    for cutout_path in cutout_paths:
        try:
            clip_id = cutout_path.stem.split("_")[0]
            sixdigits = cutout_path.stem.split("_")[1]
            assert len(sixdigits) == 6
            frame_index = int(sixdigits)
        except Exception as _:
            clip_id = None
            frame_index = None
        
        print(f"{cutout_path=}")
        out_path = cutout_path.with_suffix(".json")
        print(f"{Fore.CYAN}Does {out_path=} already exist?{Style.RESET_ALL}")
        if out_path.exists():
            print(
                textwrap.dedent(
                    f"""\
                    skipping {cutout_path} because {out_path} already exists.
                    If you really want a do-over, delete the json file. Or hand edit.
                    """
                )
            )
            continue
        

        metainfo = dict(
            clip_id=clip_id,
            frame_index=frame_index,
            kind=kind,
            jersey_id=jersey_id,
        )
        if league_id in ["nba", "euroleague"]:
            keypoint_id_to_instruction = dict(
                bottom_of_lowest_foot="Please click on bottom of the lowest foot",
                six_feet_above_that="Please click on a point six feet above that",
            )
        elif league_id == "mlb":
            if kind == "pitcher":
                keypoint_id_to_instruction = dict(
                    interfering_point="Please click on the interfering point",
                    six_feet_below_that="Please click on the point six feet below the interfering point",
                )
            elif kind == "batter":
                keypoint_id_to_instruction = dict(
                    interfering_point="Please click on the bat tip or other interfering point",
                    headtop="Please click on the top of the head",
                    six_feet_below_that="Please click on the point six feet below the head top",
                )
            else:
                raise ValueError(f"{kind=} is not supported yet")
        else:
            raise ValueError(f"{league_id=} is not supported")

        name_to_xy = attempt_to_annotate_a_cutout(
            cutout_path=cutout_path,
            keypoint_id_to_instruction=keypoint_id_to_instruction
        )

        if name_to_xy is not None:
            metainfo["name_to_xy"] = name_to_xy
            bj.color_print_json(metainfo)
            bj.dump(obj=metainfo, fp=out_path)
            print(f"wrote {out_path}")
