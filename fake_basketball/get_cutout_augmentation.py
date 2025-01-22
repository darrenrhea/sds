from get_led_screen_occluding_object_augmentation import (
     get_led_screen_occluding_object_augmentation
)
import albumentations as albu
from get_balls_augmentation import (
     get_balls_augmentation
)
from get_players_augmentation import (
     get_players_augmentation
)
from get_referees_augmentation import (
     get_referees_augmentation
)
from get_coaches_augmentation import (
     get_coaches_augmentation
)
def get_cutout_augmentation(augmentation_id: str):
    """
    What data augmentation to use during training the model.
    """
    valid_augmentation_ids = [
        "player",
        "referee",
        "coach",
        "ball",
        "justhorizontalflip",
        "led_screen_occluding_object",
    ]

    assert (
        augmentation_id in valid_augmentation_ids
    ), f"{augmentation_id=} must be in {valid_augmentation_ids=} but you gave {augmentation_id=}"

    if augmentation_id == "justhorizontalflip":
        train_transform = [
            albu.HorizontalFlip(p = 0.5),
        ]
    elif augmentation_id == "ball":
        train_transform = get_led_screen_occluding_object_augmentation() 
    elif augmentation_id == "led_screen_occluding_object":
        train_transform = get_balls_augmentation() 
    elif augmentation_id == "player":
        train_transform = get_players_augmentation()
    elif augmentation_id == "referee":
        train_transform = get_referees_augmentation()
    elif augmentation_id == "coach":
        train_transform = get_coaches_augmentation()
    else:
        raise ValueError(f"Unknown augmentation_id: {augmentation_id}")
    
    return albu.Compose(
        train_transform,
        p=1.0,
        additional_targets={'importance_mask': 'mask'},
        keypoint_params=albu.KeypointParams(
            format='xy',
            remove_invisible=False,
        )
    )