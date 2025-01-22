from make_ad_placement_descriptor_from_origin_u_v_width_height import (
     make_ad_placement_descriptor_from_origin_u_v_width_height
)
from typing import List
import numpy as np

from AdPlacementDescriptor import (
     AdPlacementDescriptor
)

from get_euroleague_geometry import (
     get_euroleague_geometry
)

def get_world_coordinate_descriptors_of_ad_placements(
    clip_id: str,
    with_floor_as_giant_ad: bool,
    overcover_by: float = 0.0  # sometimes we want the ad to be much bigger behind its hole, or for relevance masks that are slightly bigger.
) -> List[AdPlacementDescriptor]:
    """
    This function returns a list of AdPlacementDescriptor objects,
    usually for the LED boards in the arena,
    although apparently there is a with_floor_as_giant_ad option as well, god knows what that does.
    """
    ad_descriptors = []
    # can we really determine the world coordinates of the LED board corners from the context_id alone?
    # this is assuming things don't change between games!
    clip_id_to_context_id = {
        "munich2024-01-25-1080i-yadif": "munich2024",
        "munich2024-01-09-1080i-yadif": "munich2024",
        "bay-zal-2024-03-15-mxf-yadif": "munich2024",
        "bay-mta-2024-03-22-mxf": "munich2024",
        "bos-mia-2024-04-21-mxf": "boston_celtics",
        "bos-dal-2024-06-09-mxf": "boston_celtics",
        "dal-bos-2024-01-22-mxf": "dallas_mavericks",
        "dal-bos-2024-06-12-mxf": "dallas_mavericks",
        "slgame1": "summer_league_2024",
        "slday2game1": "summer_league_2024",
        "slday3game1": "summer_league_2024",
        "slday4game1": "summer_league_2024",
        "slday5game1": "summer_league_2024",
        "slday6game1": "summer_league_2024",
        # chaz may not have tracked slday7game1 well
        "slday8game1": "summer_league_2024",
        "slday9game1": "summer_league_2024",
        "slday10game1": "summer_league_2024",
        "hou-sas-2024-10-17-sdi": "houston_rockets_2024",
    }

    if clip_id not in clip_id_to_context_id:
        raise Exception(f"You need to say what the context_id of {clip_id=} is in {clip_id_to_context_id=}")

    context_id = clip_id_to_context_id[clip_id]

   
    if context_id == "munich2024":
        geometry = get_euroleague_geometry()
        points = geometry["points"]
        placementname_tl_bl_br_and_u_and_v = [
            ["LEDBRD0", "led0_tl", "led0_bl", "led01_b", [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], # 0 
            ["LEDBRD1", "led01_t", "led01_b", "led1_br", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], # 0 
            ["LEDBRD2a", "led2a_tl", "led2a_bl", "led2a_br", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ["LEDBRD2b", "led2b_tl", "led2b_bl", "led2b_br", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ["LEDBRD3", "led3_tl", "led3_bl", "led34_b", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], # 1
            ["LEDBRD4", "led34_t", "led34_b", "led4_br", [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
        if with_floor_as_giant_ad:
            placementname_tl_bl_br_and_u_and_v.insert(
                0,
                ["FLOOR", "floor_texture_tl", "floor_texture_bl", "floor_texture_br", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )
    
        # landmark_names = [key for key in points.keys()]
        for name, tl_name, bl_name, br_name, u, v in placementname_tl_bl_br_and_u_and_v:
            tl = points[tl_name]
            bl = points[bl_name]
            br = points[br_name]
            
            if overcover_by > 0:
                delta_x = overcover_by
                delta_y = overcover_by
                delta_z = overcover_by
                if name in ["LEDBRD0"]:
                    tl += np.array([0.0, -delta_y, delta_z])
                    bl += np.array([0.0, -delta_y, -delta_z])
                    br += np.array([0.0,  delta_y, -delta_z])
                if name in ["LEDBRD1"]:
                    tl += np.array([-delta_x, 0.0, delta_z])
                    bl += np.array([-delta_x, 0.0, -delta_z])
                    br += np.array([delta_x, 0.0, -delta_z])
                if name in ["LEDBRD2a", "LEDBRD2b", "LEDBRD3"]:
                    tl += np.array([-delta_x, 0.0, delta_z])
                    bl += np.array([-delta_x, 0.0, -delta_z])
                    br += np.array([delta_x, 0.0, -delta_z])
                if name in ["LEDBRD4"]:
                    tl += np.array([0.0, delta_y, delta_z])
                    bl += np.array([0.0, delta_y, -delta_z])
                    br += np.array([0.0, -delta_y, -delta_z])
            if clip_id in ["bay-zal-2024-03-15-mxf-yadif"]:
                if name == "LEDBRD1":
                    delta_x = -0.165
                    delta_z = -0.03
                    tl += np.array([delta_x, 0.0, delta_z])
                    delta_z = -0.01
                    bl += np.array([delta_x, 0.0, 0])
                    # move br to the left quite a bit:
                    delta_x = -1.945
                    #delta_z = 0.07
                    br += np.array([delta_x, 0.0, delta_z])
                if name == "LEDBRD2a":
                    delta_x = -0.44
                    tl += np.array([delta_x, 0.0, 0.0])
                    bl += np.array([delta_x, 0.0, 0.0])
                if name == "LEDBRD2b":
                    delta_x = 0.41
                    tl += np.array([0.0, 0.0, 0.0])
                    bl += np.array([0.0, 0.0, 0.0])
                    br += np.array([delta_x, 0.0, 0.0])

            if clip_id in ["bay-mta-2024-03-22-mxf"]:
                if name == "LEDBRD2a":
                    delta_x = -0.44
                    tl += np.array([delta_x, 0.0, 0.0])
                    bl += np.array([delta_x, 0.0, 0.0])
                if name == "LEDBRD2b":
                    delta_x = 0.41
                    tl += np.array([0.0, 0.0, 0.0])
                    bl += np.array([0.0, 0.0, 0.0])
                    br += np.array([delta_x, 0.0, 0.0])


            descriptor = AdPlacementDescriptor(
                name=name,
                origin=bl,
                u=u,
                v=v,
                height=np.linalg.norm(tl - bl),
                width=np.linalg.norm(br - bl) * 1.00 #
            )
            ad_descriptors.append(descriptor)



    elif context_id == "boston_celtics":
        led_screen_y_coordinate = 30.339

        ad_descriptors = []
        ad_placement_descriptor_jsonable = {
            "tl": [
                -9.77,
                led_screen_y_coordinate,
                2.605
            ],
            "bl": [
                -9.77,
                led_screen_y_coordinate,
                0.21
            ],
            "tr": [
                9.95,
                led_screen_y_coordinate,
                2.605
            ],
            "br": [
                9.95,
                30.339,
                0.21
            ],
            "origin": [
                -9.81125,
                led_screen_y_coordinate,
                0.202239
            ],
            "u": [
                1.0,
                0.0,
                0.0
            ],
            "v": [
                0.0,
                0.0,
                1.0
            ],
            "height": 2.458,
            "width": 19.74
        }
        
        tl = np.array(ad_placement_descriptor_jsonable["tl"])
        bl = np.array(ad_placement_descriptor_jsonable["bl"])
        tr = np.array(ad_placement_descriptor_jsonable["tr"])
        br = np.array(ad_placement_descriptor_jsonable["br"])

        if overcover_by > 0:
            delta_x = overcover_by
            delta_y = overcover_by
            delta_z = overcover_by
            tl += np.array([-delta_x, 0.0, delta_z])
            bl += np.array([-delta_x, 0.0, -delta_z])
            br += np.array([delta_x, 0.0, -delta_z])

        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 1.0])

        ad_placement_descriptor = AdPlacementDescriptor(
            name="LED",  # so far only one LED board in NBA
            origin=bl,
            u=u,
            v=v,
            height=np.linalg.norm(tl - bl),
            width=np.linalg.norm(br - bl) * 1.00 #
        )
        
        ad_descriptors.append(ad_placement_descriptor)

    elif context_id == "dallas_mavericks":
        # dallas only had one LED board:
        origin = [0.18643665313720703, 29.4282808303833, 1.3350388407707214]
 
        u = [1.0, 0.0, 0.0]

        v = [0.0, 0.16780186922657853, 0.9858207406440921]

        width = 50.34819984436035
        height = 1.8224297205803919


        ad_descriptors = []
        ad_placement_descriptor = make_ad_placement_descriptor_from_origin_u_v_width_height(
            origin=bl,
            u=u,
            v=v,
            width=width,
            height=height,
            overcover_by=overcover_by
        )
        
        ad_descriptors.append(ad_placement_descriptor)

    elif context_id == "summer_league_2024":
        # taken from Chaz's LED-01
        # TODO: pull this from Chaz Garfinkles' config files
        planes = {
            "LED-00":
                {
                    "origin": [
                        0.1666661688804627,
                        28.98753407892227,
                        1.4789171227490903
                    ],
                    "x": [
                        1.0,
                        0.0,
                        0.0
                    ],
                    "y": [
                        0.0,
                        0.20990269420225008,
                        0.9777222811037073
                    ]
                }
            ,
            "LED-01":
                {
                    "name": "LED-01",
                    "origin": [
                        0.12828140218734774,
                        28.98753407892227,
                        1.552085395399928
                    ],
                    "x": [
                        1.0,
                        0.0,
                        0.0
                    ],
                    "y": [
                        0.0,
                        0.2557832826252587,
                        0.9667341477001042
                    ]
                }
            ,
        }
        plane = planes["LED-00"]
        origin = plane["origin"]
        u = plane["x"]
        v = plane["y"]
        
        capture_plane = {
                "name": "LED-00",
                "bottom_left": [-25, -1.5],
                "top_right": [25, 1.5],
                "size": [4267, 256]
        }
        width = capture_plane["top_right"][0] - capture_plane["bottom_left"][0] + 0.0
        height = capture_plane["top_right"][1] - capture_plane["bottom_left"][1] + 0.0
        
        # height = 2.482585284986467
        # width = 49.30000137535095

        # -24.23623788220601, 29.664822678218343
        # 26.837863524155804
        ad_descriptors = []
        ad_placement_descriptor = make_ad_placement_descriptor_from_origin_u_v_width_height(
            origin=origin,
            u=u,
            v=v,
            width=width,
            height=height,
            overcover_by=overcover_by
        )
        
        ad_descriptors.append(ad_placement_descriptor)
    elif context_id == "houston_rockets_2024":
        # taken from Chaz's LED-01
        # TODO: pull this from Chaz Garfinkles' config files
        # "LED-00": {
        #     "bl": [
        #         -14.760334968566895,
        #         30.155729293823242,
        #         0.3668877184391022
        #     ],
        #     "br": [
        #         14.761262893676758,
        #         30.33211898803711,
        #         0.3223923146724701
        #     ],
        #     "tr": [
        #         14.774378776550293,
        #         30.356229782104492,
        #         2.7177863121032715
        #     ],
        #     "tl": [
        #         -14.74721908569336,
        #         30.179838180541992,
        #         2.762281656265259
        #     ]
        # }
        planes = {
            "LED-00":
                {
                    "origin": [
                        0.0,
                        30.5,
                        1.5,
                    ],
                    "x": [
                        1.0,
                        0.0,
                        0.0
                    ],
                    "y": [
                        0.0,
                        0.0,
                        1.0,
                    ]
                }
            ,
            "LED-01":
                {
                    "name": "LED-01",
                    "origin": [
                        0.0,
                        29.0,
                        1.5,
                    ],
                    "x": [
                        1.0,
                        0.0,
                        0.0
                    ],
                    "y": [
                        0.0,
                        0.0,
                        1.0,
                    ]
                }
            ,
        }
        plane = planes["LED-00"]
        origin = plane["origin"]
        u = plane["x"]
        v = plane["y"]
        
        w = 15
        screen_height = 3
        width = 2 * w
        suggested_rip_width = int(np.round(256 * width / screen_height))
        print(f"{suggested_rip_width=}")
        capture_plane = {
                "name": "LED-00",
                "bottom_left": [-w, -1.5],
                "top_right": [w, 1.5],
                "size": [4267, 256]
        }
        width = capture_plane["top_right"][0] - capture_plane["bottom_left"][0] + 0.0
        height = capture_plane["top_right"][1] - capture_plane["bottom_left"][1] + 0.0
        
        # height = 2.482585284986467
        # width = 49.30000137535095

        # -24.23623788220601, 29.664822678218343
        # 26.837863524155804
        ad_descriptors = []
        ad_placement_descriptor = make_ad_placement_descriptor_from_origin_u_v_width_height(
            origin=origin,
            u=u,
            v=v,
            width=width,
            height=height,
            overcover_by=overcover_by
        )
        
        ad_descriptors.append(ad_placement_descriptor)

    else:
        raise Exception(f"unknown context_id {context_id}")


    return ad_descriptors    
            
