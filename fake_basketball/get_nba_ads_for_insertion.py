from prii import (
     prii
)
from pathlib import Path


def get_nba_ads_for_insertion() -> dict:
    """
    This is what decides which ads get stuck into NBA games.
    """

    ads_not_needing_color_correction_dir = Path(
        "~/r/nba_ads_that_dont_need_color_correction"
    ).expanduser()
    
    ads_needing_color_correction_dir = Path(
        "~/r/nba_ads/summer_league_ads_they_sent_to_us"
    ).expanduser()

    assert ads_not_needing_color_correction_dir.is_dir(), f"ERROR: {ads_not_needing_color_correction_dir=} is not a directory."

    assert ads_needing_color_correction_dir.is_dir(), f"ERROR: {ads_needing_color_correction_dir=} is not a directory."
    
    # only list the ads that you have a rip for here:
    rips_from = {}

    rips_from["two_years_ago"] = [
        "golden_trophy_nba_finals_2022-06-08",
        "golden_trophy_youtube_2022",
        "nba_foundation_2022",
        "bullet_train",
        "download_the_app_2022",
        "finals_friday_9pm_et_abc_2022",
        "state_farm",
        "summer_league_2022",
    ]

    rips_from["bos-mia-2024-04-21-mxf"] = [
        "ESPN_APP",
        "ESPN_DAL_LAC_NEXT_ABC",
        "ESPN_MIL_IND_FRI",
        "ESPN_NBA_Finals",
        "NBA_APP_MSFT",
        "NBA_ID",
        "NBA_Store",
        "NHL_Playoffs",
        "PickEm_Bracket_Challenge",
        "Playoffs_DAL_LAC_NEXT_ABC",
        "Playoffs_PHI_NYK_TOM_TNT",
        "Playoffs_Title",
        "different_here",
    ]

    rips_from["bos-dal-2024-06-06-srt"] = [
        "bokeh_nba_finals_presented_by_youtube_golden_trophy",
        "nba_draft_2024_presented_by_state_farm_begins_june_26",
        "youtube_tv_youtube_tv",
        "2k25_summer_league_july_12_22_espn_espn2",
        "get_in_the_game_vote",
        "bad_boys_ride_or_die_will_smith",
        "state_farm_state_farm",
        "golden_team_logos_nba_finals_presented_by_youtube_game_2_sunday_8_et_abc",
        "stanley_cup_final_2024_oilers_florida_saturday_8_et_abc_espn",
        "golden_different_here",
        "follow_every_finals_moment_nba_app",
        "mlb_world_tour_london_series_mets_vs_phillies_sunday_10am_et_espn",
        "emirates_fly_better_emirates_fly_better",
        "golden_trophy_tshirt_nba_store",
        "golden_trophy_nba_finals_presented_by_youtube_sunday_8_pm_et_abc",
    ]

    rips_from["bos-dal-2024-06-09-mxf"] = [
        "24_BOS_Finals_CSS_v01",
        "bokeh_nba_finals_presented_by_youtube_golden_trophy",
        "nba_draft_2024_presented_by_state_farm_begins_june_26",
        "youtube_tv_youtube_tv",
        "2k25_summer_league_july_12_22_espn_espn2",
        "get_in_the_game_vote",
        "bad_boys_ride_or_die_will_smith",
        "state_farm_state_farm",
        "golden_team_logos_nba_finals_presented_by_youtube_game_2_sunday_8_et_abc",
        "stanley_cup_final_2024_oilers_florida_saturday_8_et_abc_espn",
        "golden_different_here",
        "follow_every_finals_moment_nba_app",
        "mlb_world_tour_london_series_mets_vs_phillies_sunday_10am_et_espn",
        "emirates_fly_better_emirates_fly_better",
        "golden_trophy_tshirt_nba_store",
        "golden_trophy_nba_finals_presented_by_youtube_sunday_8_pm_et_abc",
    ]

    rips_from["for_dallas"] = [
        "ABC_Finals_Game4_Fri_CS_DAL",
        "ABC_Stanley_Cup_Game3_CS_DAL",
        "ABC_WNBA_NYL_LVA_CS_DAL",
        "BB4_Sony_CS_DAL",
        "DAL_NBA_Finals_Courtside_2560x96_v01",
        "Hotels_dot_com_CS_DAL",
        "NBA_Draft_Awareness_CS_DAL",
        "NBA_Store_Finals_CS_DAL",
        "NBA_Vote_CS_DAL",
        "one_for_all_dallas_CS_DAL",
        "Tissot_CS_DAL",
        "WNBA_NYL_LVA_SAT_3_ABC_CS_DAL",
        "YTTV_CS_DAL",
    ]

    ad_ids_that_you_want_to_use_without_any_color_correction = list(set(
        rips_from["bos-dal-2024-06-09-mxf"]
        +
        rips_from["bos-dal-2024-06-06-srt"]
    ))

    # This is in the order of their banner stack 7a67723a45f770d08a15ffbbe0057477c9dfd7b43da76c289c18acbfa4451ebe
    # Some are analogous rather than exact

    ads_for_bos_dal_2024_06_09 = [
        "follow_every_finals_moment_nba_app",
        "mlb_world_tour_london_series_mets_vs_phillies_sunday_10am_et_espn",
        "Flag",
        "golden_different_here",
        "bokeh_nba_finals_presented_by_youtube_golden_trophy",
        "golden_trophy_nba_finals_presented_by_youtube_sunday_8_pm_et_abc",
        "nba_draft_2024_presented_by_state_farm_begins_june_26",
        "2k25_summer_league_july_12_22_espn_espn2",
        "get_in_the_game_vote",
        "follow_every_golden_trophy_finals_moment_nba_app_black",
        "golden_trophy_tshirt_nba_store",
        "youtube_tv_youtube_tv",
        "bad_boys_ride_or_die_will_smith",
        "state_farm_state_farm",
        "emirates_fly_better_emirates_fly_better",
        "golden_team_logos_nba_finals_presented_by_youtube_game_2_sunday_8_et_abc",
        "stanley_cup_final_2024_oilers_florida_saturday_8_et_abc_espn",
        "in_the_arena_serena_williams",
    ]

    ads_for_dal_bos_2024_06_12 = [
        "ABC_Finals_Game4_Fri_CS_DAL",
        "ABC_Stanley_Cup_Game3_CS_DAL",
        "ABC_WNBA_NYL_LVA_CS_DAL",
        "BB4_Sony_CS_DAL",
        "DAL_NBA_Finals_Courtside_2560x96_v01",
        "Finals_Friday_830_ABC_CS_DAL",
        "Flag_CS_DAL",
        "Hotels_dot_com_CS_DAL",
        "NBA_Draft_Awareness_CS_DAL",
        "NBA_Store_Finals_CS_DAL",
        "NBA_Vote_CS_DAL",
        "Statefarm_CS_DAL",
        "Summer_League_Awareness_CS_DAL",
        "Tissot_CS_DAL",
        "WNBA_NYL_LVA_SAT_3_ABC_CS_DAL",
        "YTTV_CS_DAL",
        "one_for_all_dallas_CS_DAL",
    ]  

    ads_for_dal_bos_2024_06_14 = [
        "one_for_all_dallas_CS_DAL",
        "BB4_Sony_CS_DAL",
        "DAL_NBA_Finals_Courtside_2560x96_v01",
        "Flag_CS_DAL",
        "Hotels_dot_com_CS_DAL",
        "NBA_Draft_Awareness_CS_DAL",
        "NBA_Store_Finals_CS_DAL",
        "NBA_Vote_CS_DAL",
        "Statefarm_CS_DAL",
        "Summer_League_Awareness_CS_DAL",
        "YTTV_CS_DAL",
        "ABC_NBA_Finals_Game5_IfNec_CS_DAL",
        "ABC_NHL_Stanley_Cup_Game4_CS_DAL",
        "NBA_Finals_Monday_IfNec_CS_DAL",
        "WNBA_NYL_LVA_SAT_3_ABC_CS_DAL",
        "Gatorade_Finals_CS_DAL",
        "ABC_MLB_NYY_BOS_CS_DAL",
    ]

    ads_for_bos_dal_2024_06_17 = [
        "Flag",
        "golden_different_here",
        "24_BOS_Finals_CSS_v01",
        "NBA_Finals_THURS_Game6_IfNec_CS_BOS",
        "nba_draft_2024_presented_by_state_farm_begins_june_26",
        "2k25_summer_league_july_12_22_espn_espn2",
        "get_in_the_game_vote",
        "follow_every_golden_trophy_finals_moment_nba_app_black",
        "golden_trophy_tshirt_nba_store",
        "youtube_tv_youtube_tv",
        "bad_boys_ride_or_die_will_smith",
        "state_farm_state_farm",
        "emirates_fly_better_emirates_fly_better",
        "ABC_NBA_Finals_THURS_Game6_IfNec",
        "ABC_NHL_Stanley_Cup_Game5_CS_BOS",
        "ABC_NBA_Draft_CS_BOS",
    ]



    # ads that they sent to us as rasterized images, but we need to color correct them
    # since the OETFs and the reflections and diffuse ambient light need to be added on top:
    ads_for_bos_cle_2024_05_09 = [
        "Finals_Awareness_June6",
        "Flag",
        "NBA_ID",
        "NBA_Pickem",
        "NBA_Store",
        "NBA_Tickets",
        "Playoff_Tunein_DAL_OKC_Next_ESPN",
        "Playoff_Tunein_NYK_IND_Tomorrow_ESPN",
        "Playoffs_Title",
        "WNBA_Tunein_IND_CON_Tuesday_ESPN2",
        "different_here",
        "emirates",
        "hisense",
        "nonstop_playoff_access",
    ]

    ad_ids_that_you_want_to_use_via_color_correction = ads_for_bos_dal_2024_06_17
   


    # a map from the ad_id to a list of image paths that need color correction:
    ad_name_to_paths_that_do_need_color_correction = {
        ad_id:
        [
            ads_needing_color_correction_dir / f"{ad_id}.jpg"
        ]
        for ad_id in ad_ids_that_you_want_to_use_via_color_correction
    }

    for ad_id, paths in ad_name_to_paths_that_do_need_color_correction.items():
        for p in paths:
            assert p.is_file(), f"ERROR: {p=} is not a file."

    # In the NBA, we don't have any ads that don't need color correction because we don't rip ads:
    ad_name_to_paths_that_dont_need_color_correction = {
        ad_id: (
            [
                p for p in (ads_not_needing_color_correction_dir / ad_id).glob("*.png")
            ]
            +
            [
                p for p in (ads_not_needing_color_correction_dir / ad_id).glob("*.jpg")
            ]
        )
        for ad_id in ad_ids_that_you_want_to_use_without_any_color_correction
    }
     
    return dict(
        ad_name_to_paths_that_do_need_color_correction=ad_name_to_paths_that_do_need_color_correction,
        ad_name_to_paths_that_dont_need_color_correction=ad_name_to_paths_that_dont_need_color_correction,
    )

   
if __name__ == "__main__": 
    # this is a test / demo: 
    dct = get_nba_ads_for_insertion()
    ad_name_to_paths_that_dont_need_color_correction = dct["ad_name_to_paths_that_dont_need_color_correction"]
    ad_name_to_paths_that_do_need_color_correction = dct["ad_name_to_paths_that_do_need_color_correction"]

    print("Ads that do not need color correction:")

    for ad_name, paths in ad_name_to_paths_that_dont_need_color_correction.items():
        print(f"\n{ad_name} has these rips:")
        assert len(paths) > 0, f"ERROR: {paths=} is empty."
        for cntr, p in enumerate(paths):
            print(p)
            assert p.is_file(), f"ERROR: {p=} is not a file."
            prii(p)
            if cntr  == 2:
                break

    print("\n\n\n\nRasterized RGB images that they sent us, which do need color correction:")
    for ad_name, paths in ad_name_to_paths_that_do_need_color_correction.items():
        for p in paths:
            print(p)
            assert p.is_file(), f"ERROR: {p=} is not a file."
            prii(p)


    
    