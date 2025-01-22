
from collections import OrderedDict


def get_ad_id_to_frames_in_which_it_appears(clip_id: str):
    ad_id_to_frames_in_which_it_appears = None

    if clip_id == "dal-bos-2024-06-14-mxf":
        ad_id_to_frames_in_which_it_appears = OrderedDict([
            ("Finals_Friday_830_ABC_CS_DAL", [846000, 852000, 855000]),
            ("DAL_NBA_Finals_Courtside_2560x96_v01", [357000, ]),
            ("WNBA_NYL_LVA_SAT_3_ABC_CS_DAL", [383000, 384000, 385000, 387000, 391000, 392000, ]),
            ("ABC_WNBA_NYL_LVA_CS_DAL", [723000, 741000, 749000, 750000, ]),
            ("Hotels_dot_com_CS_DAL", [395000, 403000, ]),
            ("YTTV_CS_DAL", [432000, 433000, 436000, 443000, 445000]),
            ("one_for_all_dallas_CS_DAL", [466000, 470000, 471000, ]),
            ("Summer_League_Awareness_CS_DAL", [485000, 502000, 507000, ]),
            ("BB4_Sony_CS_DAL", [511000, 512000, 515000, 516000, 523000, ]),
            ("Statefarm_CS_DAL", [543000, 544000, 549000]),
            ("ABC_Finals_Game4_Fri_CS_DAL", [618000, 621000]),
            ("ABC_NHL_Stanley_Cup_Game4_CS_DAL", [649000, 650000, 654000]),
            ("NBA_Store_Finals_CS_DAL", [664000, 668000, 689000]),
            ("NBA_Draft_Awareness_CS_DAL", [690000,692000, 698000, 699000]),
            ("Tissot_CS_DAL", [772000, 774000, 776000]),
        ])
    
    elif clip_id == "bos-dal-2024-06-09-mxf":
        ad_id_to_frames_in_which_it_appears = OrderedDict([
            ("nba_draft_2024_presented_by_state_farm_begins_june_26", [
                410000,
                413000,
            ]),
            ("follow_every_golden_trophy_finals_moment_nba_app_black", [720000, 727000, 728000]),
            ("emirates_fly_better_emirates_fly_better", [419000, 773000, 788000]),
            ("get_in_the_game_vote", [501000, 502000, 506000,507000, 523000]),
            ("bad_boys_ride_or_die_will_smith", [531000, 532000, 540000, 555000, 559000]),
            ("state_farm_state_farm", [560000, 573000, 578000]),
            ("ABC_NBA_Finals_Gm3_CS_BOS", [646000, 647000, 648000]),
            ("ABC_NHL_StanleyCup_Gm2_CS_BOS", [652000, 654000, 659000, 677000, 682000]),
            ("golden_different_here", [684000, 702000, 703000, 706000]),
            ("in_the_arena_serena_williams", [758000, 761000, 762000, 764000]),
            ("golden_trophy_tshirt_nba_store", [818000, 820000, 821000, 829000,]),
            ("NBA_Finals_WED_ABC_CS_BOS",[834000, 835000, 841000, 844000, ]), 
        ])

    return ad_id_to_frames_in_which_it_appears