from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path
import albumentations as A


def show_augmentations_cli_tool():
    # 1. Read the image from disk (as BGR)
    folder = Path("~/r/slgame1_floor/.approved").expanduser()
    image_path = folder / "slgame1_031500_nonfloor.png"
    image = open_as_rgb_hwc_np_u8(
        image_path=image_path,
    )
    prii(image, caption="original")

    # 2. Define the augmentation
    # train_transform = A.RandomBrightnessContrast(
    #     brightness_limit=(-0.02, 0.10),
    #     # brightness_limit=(-0.0001, 0.0001),
    #     contrast_limit=(-0.00001, 0.00001),  # turn it off
    #     p=1.0,
    # )
    train_transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.02, 0.10),
                contrast_limit=(-0.00001, 0.00001),  # turn it off
                p=0.95
            ),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=0.1),
        ]
    )

    # 3. Apply the augmentation
    for _ in range(30):
        augmented = train_transform(image=image)
        augmented_image = augmented["image"]

        # 4. Save the augmented image
        prii(augmented_image)
    print("BYE")