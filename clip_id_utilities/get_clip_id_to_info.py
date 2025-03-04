def get_clip_id_to_info():
    """
    There can and often are multiple videos of a single game from different cameras, or program feed.
    
    TODO: The clip_id should reference a game_id,
    and only the game_id should mention youtube_url, date, teams, court edition, jerseys, final_score, home_team, away_team, etc.

    The clip_id can talk about quality like sdi mxf youtube, or the blowout details.

    Each clip_id that occurs in the segmentation data needs to map to what it means.
    """

    clip_id_to_info = {
        "template": {
            "youtube_url": "",
            "home_team": "",
            "away_team": "",
            "date": "",
            "quality": "sdi",
            "court": "",
            "jerseys": {
                "hou": "",
                "lac": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "BOSvMIA_30-03-2022_C01_CLN_MXF": {
            "youtube_url": "https://www.youtube.com/watch?v=k8CwqSyMgRk",
            "home_team": "bos",
            "away_team": "mia",
            "date": "2022-03-30",
            "quality": "mxf",
            "court": "bos_core_2122",
            "jerseys": {
                "bos": "bos_association_2122",
                "mia": "mia_city_2122",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "hou-gsw-2024-11-02-sdi": {
            "game_id": "hou-gsw-2024-11-02",
            "quality": "sdi",
            "youtube_url": "https://www.youtube.com/watch?v=HhiKwj94TPE",
            "home_team": "hou",
            "away_team": "gsw",
            "date": "2024-11-02",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "hou_icon_2425",
                "lac": "https://appimages.nba.com/p/tr:n-slnfre/2024/uniform/Golden%20State%20Warriors/GSW_IE.jpg",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "hou-por-2024-11-23-sdi": {
            "youtube_url": "https://www.youtube.com/watch?v=mzZbPTmzVTk",
            "home_team": "hou",
            "away_team": "por",
            "date": "2024-11-23",
            "quality": "sdi",
            "court": "hou_city_2425",
            "jerseys": {
                "hou": "hou_city_2425",
                "por": "por_city_2425",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "HOUvBOS_LAR_city_ssn_01_03_2024": {
            "viewpoint": "LAR",
            "youtube_url": "https://www.youtube.com/watch?v=3qDaleDJ8rE",
            "home_team": "hou",
            "away_team": "bos",
            "date": "2025-01-03",
            "quality": "mxf",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "hou_icon_2425",
                "bos": "bos_association_2425",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "HOUvBOS_RAR_city_ssn_01_03_2024": {
            "viewpoint": "RAR",
            "youtube_url": "https://www.youtube.com/watch?v=3qDaleDJ8rE",
            "home_team": "hou",
            "away_team": "bos",
            "date": "2025-01-03",
            "quality": "mxf",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "hou_icon_2425",
                "bos": "bos_association_2425",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        
        "HOUvLAL_LAR_city_ssn_01_05_2024": {
            "viewpoint": "LAR",
            "youtube_url": "https://www.youtube.com/watch?v=sruAMHYutvU",
            "home_team": "hou",
            "away_team": "lal",
            "date": "2025-01-05",
            "quality": "mxf",
            "court": "hou_city_2425",
            "jerseys": {
                "hou": "",
                "lal": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "HOUvLAC_RAR_core_ssn_11-13-2024": {
            "viewpoint": "RAR",
            "youtube_url": "https://www.youtube.com/watch?v=gg-trSRWNM8",
            "home_team": "hou",
            "away_team": "lac",
            "date": "2024-11-13",
            "quality": "mxf",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "",
                "lac": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
            "comments": [
                "This is a test comment",
            ]
        },
        "HOUvLAC_LAR_core_ssn_11-13-2024": {
            "viewpoint": "LAR",
            "youtube_url": "https://www.youtube.com/watch?v=gg-trSRWNM8",
            "home_team": "hou",
            "away_team": "lac",
            "date": "",
            "quality": "mxf",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "",
                "lac": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "hou-mem-2025-01-13-sdi": {
            "youtube_url": "https://www.youtube.com/watch?v=Es0AvuWIzdQ",
            "home_team": "hou",
            "away_team": "mem",
            "date": "2025-01-13",
            "quality": "sdi",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "",
                "mem": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",
        },
        "hou-det-2025-01-20-sdi": {
            "youtube_url": "https://www.youtube.com/watch?v=B5rhv0W471s",
            "home_team": "hou",
            "away_team": "det",
            "date": "2025-01-20",
            "quality": "sdi",
            "court": "hou_core_2425",
            "jerseys": {
                "hou": "",
                "det": "",
            },
            "clip_sha256": "",
            "old_s3_path": "",
            "league": "nba",
            "camera_poses": "",

        },
        "hou-was-2024-11-11-sdi": {
            "youtube_url": "https://www.youtube.com/watch?v=UeoYoG9ZMP8",
            "home_team": "hou",
            "away_team": "was",
            "date": "2024-11-11",
            "quality": "sdi",
            "court": "hou_icon_2425",
            "jerseys": {
                "hou": "",
                "was": "",
            },
            "clip_sha256": "10fdd02dfa8ba380745bfa2955e29029adce0d9054d401dba72beae37b71669e",
            "old_s3_path": "s3://awecomai-original-videos/20241111HoustonRockets/out_NEWvsHOU_20241112_noaugment.ts",
            "league": "nba",
            "camera_poses": "f18840b7166b9081ae19ff108327e9a62ddc0164bfef0472da9cd8ab5610986a",
        },
        "hou-lac-2024-11-15-sdi": {
            "court": "hou_cup_2425",
            "home_team": "hou",
            "away_team": "lac",
            "jerseys": {
                "hou": "hou_statement_2425",
                "lac": "lac_association_2425",
            },
            "date": "2024-11-15",
            "quality": "sdi",
            "youtube_url": "https://www.youtube.com/watch?v=tT5ms06AWTA",
            "clip_sha256": "3b24498fcd1c94cecc9072a2873bf9023bb6fa4cba49b019b6aa517612fd7b7c",
            "old_s3_path": "s3://awecomai-original-videos/20241115HoustonRockets/out_NEWvsHOU_20241116_noaugment.ts",
            "league": "nba",
        },
        "den1": {
            "court": "den_city_1920",
            "home_team": "den",
            "away_team": "min",
            "date": "2020-02-23",
            "youtube_url": "https://www.youtube.com/watch?v=T35Pa770aBM",
            "footage_name": "0000954524_MIN_at_DEN_02232020.mxf",
            "league": "nba",
        },
        "BOSvDAL_PGM_core_esp_11-23-2022": {
            "league": "nba",
            "date": "2022-11-23",
            "court": "bos_core_2223",
            "youtube_url": "https://www.youtube.com/watch?v=4GoNOIrBmOM",
            "footage_name": "BOSvDAL_PGM_core_esp_11-23-2022.mxf",
            "home_team": "BOS",
            "away_team": "DAL",
        },
        "BOSvDEN_PGM_core_alt_11-11-2022": {
            "court": "bos_core_2223",
            "league": "nba",
            "date": "2022-11-11",
            "youtube_url": "https://www.youtube.com/watch?v=jesrOINX4PI",
            "footage_name": "BOSvDEN_PGM_core_alt_11-11-2022.mxf",
            "home_team": "BOS",
            "away_team": "DEN",
        },
        "CHAvNYK_PGM_city_bal_12-09-2022": {
            "court": "cha_city_2223",
            "league": "nba",
            "date": "2022-12-09",
            "youtube_url": "https://www.youtube.com/watch?v=bnZjTzAcLNY",
            "footage_name": "CHAvNYK_PGM_city_bal_12-09-2022.mxf",
            "home_team": "CHA",
            "away_team": "NYK",
        },
        "DENvBOS_PGM_core_alt_01-01-2023": {
            "court": "den_icon_2223",
            "league": "nba",
            "date": "2023-01-01",
            "youtube_url": "https://www.youtube.com/watch?v=5OFCDbI_YoI",
            "footage_name": "DENvBOS_PGM_core_alt_01-01-2023.mxf",
            "home_team": "DEN",
            "away_team": "BOS",
        },
        "DENvDET_PGM_core_alt_11-22-2022": {
            "court": "den_icon_2223",
            "league": "nba",
            "date": "2022-11-22",
            "youtube_url": "https://www.youtube.com/watch?v=zFZ1CQzYba4",
            "footage_name": "DENvDET_PGM_core_alt_11-22-2022.mxf",
            "home_team": "DEN",
            "away_team": "DET",
        },
        "DENvNYK_PGM_core_alt_11-16-2022": {
            "court": "den_icon_2223",
            "league": "nba",
            "date": "2022-11-16",
            "youtube_url": "https://www.youtube.com/watch?v=nB6e8BEBIJY",
            "footage_name": "DENvNYK_PGM_core_alt_11-16-2022.mxf",
            "home_team": "DEN",
            "away_team": "NYK",
        },
        "GSWvCLE_PGM_city_nbc_11-11-2022": {
            "court": "gsw_city_2223",
            "league": "nba",
            "date": "2022-11-11",
            "youtube_url": "https://www.youtube.com/watch?v=LgS_ccpXxMI",
            "footage_name": "GSWvCLE_PGM_city_nbc_11-11-2022.mxf",
            "home_team": "GSW",
            "away_team": "CLE",
        },
        "GSWvBOS_13-06-2022_C01_CLN_MXF": {
            "court": "gsw_city_2122",
            "court_url": "https://www.reddit.com/r/warriors/comments/qm0q8k/the_new_city_edition_court_and_jerseys_will_debut/",
            "league": "nba",
            "date": "2022-06-13",
            "youtube_url": "https://www.youtube.com/watch?v=fTpZSpG0_vg",
            "home_team": "GSW",
            "away_team": "BOS",
        },
        "LALvPOR_PGM_core_spe_11-30-2022": {
            "court": "lal_core_2223",
            "league": "nba",
            "date": "2022-11-30",
            "youtube_url": "https://www.youtube.com/watch?v=AQKUK-RHQp8",
            "footage_name": "LALvPOR_PGM_core_spe_11-30-2022.mxf",
            "home_team": "LAL",
            "away_team": "POR",
        },
        "LALvSAC_PGM_core_spe_11-11-2022": {
            "court": "lal_core_2223",
            "league": "nba",
            "date": "2022-11-11",
            "youtube_url": "https://www.youtube.com/watch?v=q7ZtIZHd2KI",
            "footage_name": "LALvSAC_PGM_core_spe_11-11-2022.mxf",
            "home_team": "LAL",
            "away_team": "SAC",
        },
        "MIAvNYK_PGM_core_bal_03-03-2023": {
            "court": "mia_core_2223",
            "league": "nba",
            "date": "2023-03-03",
            "youtube_url": "https://www.youtube.com/watch?v=hHVfgSJYZ7k",
            "footage_name": "MIAvNYK_PGM_core_bal_03-03-2023.mxf",
            "home_team": "MIA",
            "away_team": "NYK",
        },
        "MIAvPHX_PGM_core_bal_11-14-2022": {
            "court": "mia_core_2223",
            "league": "nba",
            "date": "2022-11-14",
            "youtube_url": "https://www.youtube.com/watch?v=CG-fzJpTJ18",
            "footage_name": "MIAvPHX_PGM_core_bal_11-14-2022.mxf",
            "home_team": "MIA",
            "away_team": "PHX",
        },
        "PHI_CORE_2022-03-14_DEN_PGM": {
            "court": "phi_core_2122",
            "league": "nba",
            "date": "2022-03-14",
            "youtube_url": "https://www.youtube.com/watch?v=p19rl0LxAa0",
            "footage_name": "30Mbps_0001256078_NBA202200020920750001r.mp4",
            "home_team": "PHI",
            "away_team": "DEN",
            "away_color": "red",
            "home_color": "navy-blue",
        },
        "PHIvATL_PGM_core_nbc_11-12-2022": {
            "court": "phi_core_2223",
            "league": "nba",
            "date": "2022-11-12",
            "youtube_url": "https://www.youtube.com/watch?v=NLXkCHUPErU",
            "footage_name": "PHIvATL_PGM_core_nbc_11-12-2022.mxf",
            "home_team": "PHI",
            "away_team": "ATL",
        },
        "PHI_CORE_2022-04-16_TOR_PGM": {
            "court": "phi_core_2122",
            "league": "nba",
            "date": "2022-04-16",
            "youtube_url": "https://www.youtube.com/watch?v=zN3n0z9C878",
            "footage_name": "PHI_CORE_2022-04-16_TOR_PGM.mxf",
            "home_team": "PHI",
            "away_team": "TOR",
        },
        "SL_2022_00": {
            "league": "nba",
            "court": "summer_league_22",
            "court_example_image": "ff8d49821b890e6cc0729381b16f4cae90652d11753324e952da129c07c59446",
            "youtube_url": "https://www.youtube.com/watch?v=FfmhosW66E0",
            "date": "2022-07-10",
            "footage_name": "/mnt/nas/volume1/videos/nba/SummerLeague/SL00_0001309096_NBA202200021184430001r.mxf",
            "home_team": "BKN",
            "away_team": "PHI",
            "away_color": "blue",
            "home_color": "white",
        },
        "SL_2022_02": {
            "court": "summer_league_22",
            "court_example_frame": "a84657a7b865bd8510abd4a07b5d5e3f58631d7bfa55ba8e17645ea08f798e74",
            "league": "nba",
            "youtube_url": "https://www.youtube.com/watch?v=FfmhosW66E0",
            "date": "2022-07-10",
            "footage_name": "/mnt/nas/volume1/videos/nba/SummerLeague/SL02_0001309141_NBA202200021184450001r.mxf",
            "home_team": "GSW",
            "away_team": "SAS",
            "home_color": "blue",
            "away_color": "white",
        },
        "bay-mta-2024-03-22-part1-srt": {
            "league": "euroleague",
            "date": "2024-03-22",
            "footage_name": "aws s3 ls s3://awecomai-original-videos/EB_23-24_R31_BAY-MTA_hq.ts",
            "youtube_url": "https://www.youtube.com/watch?v=H6e6X0T6zOo",
            "home_team": "BAY",
            "away_team": "MTA",
        },
        "bos-dal-2024-06-09-mxf": {
            "court": "bos_core_2324",
            "league": "nba",
            "date": "2024-06-09",
            "footage_name": "/media/drhea/corsair4tb3/Downloads/DAL_at_BOS_6924.mxf",
            "youtube_url": "https://www.youtube.com/watch?v=6NBaHv8uRUQ",
            "home_team": "BOS",
            "away_team": "DAL",
        },
        "dal-bos-2024-06-12-mxf": {
            "court": "dal_core_2425",
            "league": "nba",
            "date": "2024-06-12",
            "footage_name": "/media/drhea/muchspace/s3/awecomai-original-videos/DALvBOS_PGM_core_esp_06-12-2024.mxf",
            "youtube_url": "https://www.youtube.com/watch?v=E6I0td7GEOw",
            "home_team": "DAL",
            "away_team": "BOS",
        },
        "hou-lac-2023-11-14": {
            "court": "hou_core_2324",
            "league": "nba",
            "date": "2022-11-14",
            "footage_name": "/mnt/nas/volume1/videos/nba/2022-2023_Season_Videos/HOUvLAC_PGM_core_att_11-14-2022.mxf",
            "footage_sha256": "e5e5314bffcc8a0f13d004b3d5013645ed642a19255b65ba2826177a43b08f0b",
            "youtube_url": "https://www.youtube.com/watch?v=B0hAYFUtBnc",
            "home_team": "HOU",
            "away_team": "LAC",
        },
        "hou-sas-2024-10-17-sdi": {
            "court": "hou_core_2425",
            "league": "nba",
            "date": "2024-10-17",
            "footage_name": "/shared/s3/awecomai-original-videos/20241017HoustonRockets.ts",
            "footage_s3": "s3://awecomai-original-videos/20241017HoustonRockets.ts",
            "footage_sha256": "e5e5314bffcc8a0f13d004b3d5013645ed642a19255b65ba2826177a43b08f0b",
            "youtube_url": "https://www.youtube.com/watch?v=lIH7fQprFtU",
            "home_team": "HOU",
            "away_team": "SAS",
        },
        "DSCF0236": {
            "league": "euroleague",
            "date": "2024-01-29",
            "footage_s3": "s3://awecomai-test-videos/nba/Mathieu/MUNICH/DSCF0236.MOV",
            "home_team": "BAY",
        },
        "DSCF0240": {
            "league": "euroleague",
            "date": "2024-01-29",
            "footage_s3": "s3://awecomai-test-videos/nba/Mathieu/MUNICH/DSCF0240.MOV",
            "home_team": "BAY",
        },
        "DSCF0241": {
            "league": "euroleague",
            "date": "2024-01-29",
            "footage_s3": "s3://awecomai-test-videos/nba/Mathieu/MUNICH/DSCF0241.MOV",
            "home_team": "BAY",
        },
        "MUN_ASVEL_CALIB_VID": {
            "league": "euroleague",
            "date": "2024-01-29",
            "footage_s3": "s3://awecomai-test-videos/nba/Mathieu/MUN_ASVEL_CALIB_VID.ts",
            "home_team": "BAY",
            "away_team": "ASVEL",
        },
        "lon-lei-2024-03-03-mov-yadif": {
            "league": "london",
            "date": "2024-03-03",
            "footage_s3": "s3://awecomai-test-videos/epl/lon-lei-2024-03-03-mov-yadif.ts",
            "home_team": "LON",
            "away_team": "LEI",
        },
        "london20240208": {
            "league": "london",
            "date": "2024-02-08",
            "footage_s3": "/media/drhea/muchspace/s3/awecomai-test-videos/nba/Mathieu/20240208_CHA_LON_SHE_DIRTY_1_gameplay.mp4 ",
            "home_team": "LON",
        },
    }

    return clip_id_to_info