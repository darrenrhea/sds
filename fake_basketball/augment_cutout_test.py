from get_top_xy_for_cutout import (
     get_top_xy_for_cutout
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from prii import (
     prii
)
from get_cutouts import (
     get_cutouts
)
from augment_cutout import (
     augment_cutout
)
from get_cutout_augmentation import (
     get_cutout_augmentation
)
import albumentations as albu


def test_augment_cutout():
    cutouts = get_cutouts(
        just_this_kind="ball"
    )
    
    augmentation_id = "forballs"
    transform = get_cutout_augmentation(augmentation_id)

    assert isinstance(transform, albu.core.composition.Compose)

    for cutout in cutouts:
        rgba_np_u8 = cutout.rgba_np_u8
        top_xy = get_top_xy_for_cutout(
            cutout=cutout
        )
        prii(rgba_np_u8, caption="before augmentation")
        print("Examples of it being augmented")
        for _ in range(10):
            new_rgba_np_u8, new_top_xy = augment_cutout(
                rgba_np_u8=rgba_np_u8,
                top_xy=top_xy,
                transform=transform
            )

            prii_named_xy_points_on_image(
                image=new_rgba_np_u8,
                name_to_xy={
                    "top_xy": new_top_xy
                }
            )


if __name__ == "__main__":
    test_augment_cutout()
    print("augment_cutout_test.py has passed all tests")