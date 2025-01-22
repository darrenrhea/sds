from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
import numpy as np


def get_random_world_coordinates_for_a_cutout_of_this_kind(
    cutout_kind: str,
    league: str,
    context_id: str,  # this should be pasting_distribution_context_id
) -> np.ndarray:
    """
    coaches tend to lurk in certain places near the LED board,
    whereas players tend to be on the court,
    and referees tend to be on the court but in the periphery.
    balls are all over, but we might want to place them for maximum learning.
    led_screen_occluding_objects are in front of the LED screen.
    This determines the attachment points for cutouts of a given kind, and
    will help order them by distance from the camera.
    """
    
    assert (
        league in ["euroleague", "nba"]
    ), f"{league=} is not in ['euroleague', 'nba'], and we hae only configured those two leagues."

    valid_cutout_kinds = get_valid_cutout_kinds()
    assert cutout_kind in valid_cutout_kinds, f"{cutout_kind=} is not in {valid_cutout_kinds=}"

    # for Euroleague it is in meters:
    if league == "euroleague":
        if cutout_kind == "player":
            p_giwc = np.array([
                np.random.uniform(-14, 14),
                np.random.uniform(0, 8.5),
                # np.random.uniform(3, 7.5),
                0.0
            ])
        elif cutout_kind == "referee":
            p_giwc = np.array([
                np.random.uniform(-14, 14),
                np.random.uniform(-7.5, 7.5),
                0.0
            ])
        elif cutout_kind == "coach":
            n = np.random.randint(0, 100)
            if n < 98:  # really close to the LED board
                p_giwc = np.array([
                    np.random.uniform(-14, 14),
                    np.random.uniform(7.5, 8.5),
                    0.0
                ])
            # these are mainly an escape hatch for rare camera poses:
            elif n == 98: 
                p_giwc = np.array([
                    np.random.uniform(-15, -14),
                    np.random.uniform(-7.5, 7.5),
                    0.0
                ])
            elif n == 99:
                p_giwc = np.array([
                    np.random.uniform(-15, -14),
                    np.random.uniform(-7.5, 7.5),
                    0.0
                ])
        elif cutout_kind == "referee":
            n = np.random.randint(0, 2)
            if n == 0:
                p_giwc = np.array([
                    np.random.uniform(-14, 14),
                    np.random.uniform(6.5, 8.5),
                    0.0
                ])
            elif n == 1:
                p_giwc = np.array([
                    np.random.uniform(-15, -14),
                    np.random.uniform(-7.5, 7.5),
                    0.0
                ])
            elif n == 2:
                p_giwc = np.array([
                    np.random.uniform(14, 15),
                    np.random.uniform(-7.5, 7.5),
                    0.0
                ])
            else:
                raise Exception(f"ERROR: {n=}")
        
        elif cutout_kind == "ball":
            n = np.random.randint(0, 2)
            if n == 0:
                p_giwc = np.array([
                    np.random.uniform(-14, 14),
                    np.random.uniform(6.5, 8.5),
                    np.random.uniform(0.0, 1.5),
                ])
            elif n == 1:
                p_giwc = np.array([
                    np.random.uniform(-15, -14),
                    np.random.uniform(-7.5, 7.5),
                    np.random.uniform(0.0, 1.5),
                ])
            elif n == 2:
                p_giwc = np.array([
                    np.random.uniform(14, 15),
                    np.random.uniform(-7.5, 7.5),
                    np.random.uniform(0.0, 1.5),
                ])
            else:
                raise Exception(f"ERROR: {n=}")

        else:
            raise ValueError(f"Unknown cutout_kind: {cutout_kind}")
    
    # for NBA it is in feet:
    elif league == "nba":
        if context_id == "boston_celtics":
            player_x_min = -16
            player_x_max = 16
            player_y_min = 0
            player_y_max = 27
            referee_x_min = -16
            referee_x_max = 16
            referee_y_min = 0
            referee_y_max = 27
            
            coach_x_min = -16
            coach_x_max = 16
            coach_y_min = 24
            coach_y_max = 27

            ball_x_min = -16
            ball_x_max = 16
            ball_y_min = 27
            ball_y_max = 30
            ball_z_min = 0.0
            ball_z_max = 4.0

            led_screen_occluding_object_x_min = -10
            led_screen_occluding_object_x_max = 10
            led_screen_occluding_object_y_min = 28.5
            led_screen_occluding_object_y_max = 29.5

        elif context_id == "dallas_mavericks":
            player_x_min = -16
            player_x_max = 16
            player_y_min = 0
            player_y_max = 27
            referee_x_min = -16
            referee_x_max = 16
            referee_y_min = 0
            referee_y_max = 27
            
            coach_x_min = -16
            coach_x_max = 16
            coach_y_min = 24
            coach_y_max = 27

            ball_x_min = -16
            ball_x_max = 16
            ball_y_min = 27
            ball_y_max = 30
            ball_z_min = 0.0
            ball_z_max = 4.0

            led_screen_occluding_object_x_min = -26
            led_screen_occluding_object_x_max = 26
            led_screen_occluding_object_y_min = 28.5
            led_screen_occluding_object_y_max = 29.5

        elif context_id == "summer_league_2024":
            player_x_min = -26
            player_x_max = 26
            player_y_min = 0
            player_y_max = 27
            referee_x_min = -26
            referee_x_max = 26
            referee_y_min = 0
            referee_y_max = 27
            
            coach_x_min = -26
            coach_x_max = 26
            coach_y_min = 24
            coach_y_max = 27

            ball_x_min = -26
            ball_x_max = 26
            ball_y_min = 27
            ball_y_max = 30
            ball_z_min = 0.0
            ball_z_max = 4.0

            led_screen_occluding_object_x_min = -26
            led_screen_occluding_object_x_max = 26
            led_screen_occluding_object_y_min = 28.5
            led_screen_occluding_object_y_max = 29.5
        elif context_id == "nba_floor_not_floor_pasting":
            player_x_min = -47
            player_x_max = 47
            player_y_min = -25
            player_y_max = 25
            referee_x_min = -47
            referee_x_max = 47
            referee_y_min = -25
            referee_y_max = 25
            
            coach_x_min = -47
            coach_x_max = 47
            coach_y_min = -25
            coach_y_max = 25

            ball_x_min = -47
            ball_x_max = 47
            ball_y_min = -25
            ball_y_max = 25
            ball_z_min = 0.0
            ball_z_max = 4.0

            led_screen_occluding_object_x_min = -26
            led_screen_occluding_object_x_max = 26
            led_screen_occluding_object_y_min = 28.5
            led_screen_occluding_object_y_max = 29.5

        else:
            raise ValueError(f"Unknown context_id: {context_id}")
        
        if cutout_kind == "player":
            p_giwc = np.array([
                np.random.uniform(player_x_min, player_x_max),
                np.random.uniform(player_y_min, player_y_max),
                0.0,
            ])        
        elif cutout_kind == "referee":
            p_giwc = np.array([
                np.random.uniform(referee_x_min, referee_x_max),
                np.random.uniform(referee_y_min, referee_y_max),
                0.0,
            ])

        elif cutout_kind == "coach":  
            p_giwc = np.array([
                np.random.uniform(coach_x_min, coach_x_max),
                np.random.uniform(coach_y_min, coach_y_max),
                0.0
            ])
            
        elif cutout_kind == "referee":
            p_giwc = np.array([
                np.random.uniform(-16, 16),
                np.random.uniform(0, 27),
                0.0
            ])
           
        # choose a random ball position that makes it somewhat likely to be in front of the LED board:
        elif cutout_kind == "ball":
            p_giwc = np.array([
                np.random.uniform(ball_x_min, ball_x_max),
                np.random.uniform(ball_y_min, ball_y_max),
                np.random.uniform(ball_z_min, ball_z_max),
            ])
        
        elif cutout_kind == "led_screen_occluding_object":
            p_giwc = np.array([
                np.random.uniform(led_screen_occluding_object_x_min, led_screen_occluding_object_x_max),
                np.random.uniform(led_screen_occluding_object_y_min, led_screen_occluding_object_y_max),
                0.0,
            ])
          
        else:
            raise ValueError(f"Unknown cutout_kind: {cutout_kind}")
    
    return p_giwc