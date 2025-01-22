from pathlib import Path
from Drawable2DImage import Drawable2DImage
import numpy as np
import PIL
import better_json as bj


def main():

    data_repo_dir = Path("~/r/sac_core").expanduser()
    assert data_repo_dir.is_dir(), f"The directory {data_repo_dir} does not exist"

    annotated_photo_name = "sac_core_2022-23a_0035"

    json_path = Path(
        data_repo_dir / f"{annotated_photo_name}_keypoints.json"
    ).expanduser()

    assert json_path.is_file(), f"The file {json_path} does not exist"

    name_to_location = bj.load(json_path)

    original_image_path = data_repo_dir / "original_photos" / f"{annotated_photo_name}.jpg"
    assert original_image_path.is_file(), f"The file {original_image_path} does not exist"

    original_image_pil = PIL.Image.open(str(original_image_path)).convert("RGBA")
    rgba_np_uint8 = np.array(original_image_pil)
    annotation_color = np.array([255, 0, 0, 255], dtype=np.uint8)

    print(name_to_location)
    xs = []
    ys = []
    names = []


    for name, val in name_to_location.items():
        # the x coordinates in the json range over [-1, 1] where +1 is the far right of the photo
        # the y coordinates in the json range over [-h/w to h/w] i.e.
        # [-3648/5472, 3648/5472] where  +3648/5472 is the bottom of the photo
        x_float = val[0]
        y_float = val[1]
        x_pixel = (x_float + 1) / 2 * 5472
        y_pixel = (y_float + 3648/5472) / 2 * 5472
        xs.append(x_pixel)
        ys.append(y_pixel)
        names.append(name)
    
    print(xs)
    print(ys)
    print(names)
    drawable_image = Drawable2DImage(
        rgba_np_uint8=rgba_np_uint8,
        expand_by_factor=2
    )

    drawable_image.draw_line_from_point_to_point(
        (0,0),
        (1000, 1000),
        rgb=(0, 255, 0),
        width=2
    )

    drawable_image.draw_2d_curve(
        t_min=np.pi - np.arcsin(22 / (23 + 2 / 3)),
        t_max=np.pi + np.arcsin(22 / (23 + 2 / 3)),
        t_to_xy_function=lambda t: (
            1000.0 + 500 * np.cos(t),
            1000.0 + 500* np.sin(t)
        ),
        rgb=(0, 255, 0),
        num_steps=500,
    )

    for x, y, name in zip(xs, ys, names):
        drawable_image.draw_plus_at_2d_point(
            x_pixel=x,
            y_pixel=y,
            rgb=(0, 255, 0),
            size=10,
            text=name
        )

    drawable_image.save(output_image_file_path="out.png")
    print(f"open out.png")


if __name__ == "__main__":
    main()
