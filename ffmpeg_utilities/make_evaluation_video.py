from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)
from get_font_file_path import (
     get_font_file_path
)
from pathlib import Path
import subprocess


from colorama import Fore, Style


def make_evaluation_video(
    original_suffix: str,
    frames_dir: Path,
    masks_dir: Path,
    first_frame_index: int,
    last_frame_index: int,
    clip_id: str,
    model_id: str,
    fps: float,
    what_is_normal_color: str,
    out_video_file_path: Path,
    fill_color: str,
    draw_frame_numbers: bool,
):  
    """
    Suppose you already have video frames blown out into a directory called
    :python:`frames_dir` saved as
    as JPEGs with file extension .jpg and naming format

    .. code-block:: python

         frames_dir / f"{clip_id}_%06d.jpg"

    and you have segmentation masks blown out into a directory called
    :python:`masks_dir`
    as PNGs with file extension .png
    
    .. code-block:: python

         masks_dir / f"{clip_id}_%06d_{model_id}.png"

    Makes a video that shows the segmentation masks overlaid on the RGB frames.
    You can:
    make the foreground normal color,
    and the background greenified via
    do_background_green=True
    or
    make the foreground greenified, and
    the background normal color via
    do_background_green=False
    """
    if fill_color == "green":
        matrix_coefficients = ".0:.0:.0:.0:2.0:2.0:2.0:.0:.0:.0:.0:.0"
    elif fill_color == "black":
        matrix_coefficients = ".0:.0:.0:.0:.0:.0:.0:.0:.0:.0:.0:.0"
    else:
        raise ValueError(f"ERROR: fill_color must be one of ['black', 'green'] but was {fill_color=}")
    assert original_suffix in [".jpg", ".png", "_original.png", "_original.jpg", "_original_clahe.jpg"]
    if "_" in model_id:
        # print in yellow
        print(Fore.YELLOW)
        print(f"WARNING: model_id {model_id} contains an underscore. This is not a good idea, and it probably an error.")
        print(Style.RESET_ALL)
    
    assert out_video_file_path.parent.is_dir(), f"ERROR: out_video_file_path.parent {out_video_file_path.parent} is not an extant directory!"

    assert isinstance(what_is_normal_color, str), f"ERROR: what_is_normal_color must be a string but was {type(what_is_normal_color)}"
    assert what_is_normal_color in ["background", "foreground"], f"ERROR: what_is_normal_color must be one of ['background', 'foreground'] but was {what_is_normal_color}"

    assert isinstance(fps, float), f"ERROR: fps must be a float but was {type(fps)} with value {fps=}"
    assert fps in [25, 50, 59.94, 29.97], f"ERROR: fps must be one of [50, 59.94, 29.97] but was {fps}"

    assert isinstance(first_frame_index, int), f"ERROR: first_frame_index must be an int but was {type(first_frame_index)} with value {first_frame_index=}"
    assert isinstance(last_frame_index, int), f"ERROR: last_frame_index must be an int but was {type(last_frame_index)} with value {last_frame_index=}"
    assert first_frame_index <= last_frame_index, f"ERROR: first_frame_index must be less than or equal to last_frame_index but was {first_frame_index=}, {last_frame_index=}"

    assert (
        frames_dir.is_dir()
    ), f"ERROR: frames_dir {frames_dir} is not an extant directory!"

    assert (
        masks_dir.is_dir()
    ), f"ERROR: masks_dir {masks_dir} is not an extant directory!"

    font_file_path = get_font_file_path()
    ffmpeg = get_nonbroken_ffmpeg_file_path()

    num_frames = last_frame_index - first_frame_index + 1

    first_original_image = frames_dir / f"{clip_id}_{first_frame_index:06d}{original_suffix}"

    last_original_image = frames_dir / f"{clip_id}_{last_frame_index:06d}{original_suffix}"

    assert first_original_image.is_file(), f"ERROR: {first_original_image=} is not an extant file!"

    assert last_original_image.is_file(), f"ERROR: {last_original_image=} is not an extant file!"

    first_mask_image = masks_dir / f"{clip_id}_{first_frame_index:06d}_{model_id}.png"
    assert first_mask_image.is_file(), f"ERROR: {first_mask_image=} is not an extant file!"

    last_mask_image = masks_dir / f"{clip_id}_{last_frame_index:06d}_{model_id}.png"
    assert last_mask_image.is_file(), f"ERROR: {last_mask_image=} is not an extant file!"
    
    crf_str = "22"

    if what_is_normal_color == "foreground":
        if draw_frame_numbers:
            complex_filter_str = f"[1][0]alphamerge[fg];[1]eq=contrast=-1.0:brightness=0.2[lowcont];[lowcont]colorchannelmixer={matrix_coefficients}[green];[green][fg]overlay[final];[final]drawtext=fontfile={font_file_path}: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=3*lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5"
        else:
            complex_filter_str = f"[1][0]alphamerge[fg];[1]eq=contrast=-1.0:brightness=0.2[lowcont];[lowcont]colorchannelmixer={matrix_coefficients}[green];[green][fg]overlay"
        # https://superuser.com/questions/1330300/need-a-detail-explanation-for-ffmpeg-colorchannelmixer
        args = [
            str(ffmpeg),
            "-y",
            "-nostdin",
            "-start_number",
            str(first_frame_index),
            "-framerate",
            str(fps),
            "-i",
            str(masks_dir / f"{clip_id}_%06d_{model_id}.png"),
            "-start_number",
            str(first_frame_index),
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / f"{clip_id}_%06d{original_suffix}"),
            "-frames",
            str(num_frames),
            "-filter_complex",
            complex_filter_str,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            crf_str,
            str(out_video_file_path),
        ]

        print(Fore.GREEN)
        print("  \\\n".join(args))
        print(Style.RESET_ALL)
    
    elif what_is_normal_color == "background":
        if draw_frame_numbers:
            complex_filter_str = f"[0]lut=c0=negval[flipped];[1][flipped]alphamerge[fg];[1]eq=contrast=-1.0:brightness=0.2[lowcont];[lowcont]colorchannelmixer={matrix_coefficients}[green];[green][fg]overlay[final];[final]drawtext=fontfile={font_file_path}: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5"
        else:
            complex_filter_str = f"[0]lut=c0=negval[flipped];[1][flipped]alphamerge[fg];[1]eq=contrast=-1.0:brightness=0.2[lowcont];[lowcont]colorchannelmixer={matrix_coefficients}[green];[green][fg]overlay"

        args = [
            str(ffmpeg),
            "-y",
            "-nostdin",
            "-start_number",
            str(first_frame_index),
            "-i",
            str(masks_dir / f"{clip_id}_%06d_{model_id}.png"),
            "-framerate",
            str(fps),
            "-start_number",
            str(first_frame_index),
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / f"{clip_id}_%06d{original_suffix}"),
            "-frames",
            str(num_frames),
            "-filter_complex",
            complex_filter_str,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",  # why was there a j in there???
            "-crf",
            crf_str,
            str(out_video_file_path),
        ]

        print(Fore.GREEN)
        print("  \\\n".join(args))
        print(Style.RESET_ALL)

    else:
        raise ValueError(f"ERROR: what_is_normal_color must be one of ['background', 'foreground'] but was {what_is_normal_color}")
    
    subprocess.run(
        args=args,
        capture_output=False
    )

    assert (
        out_video_file_path.is_file()
    ), f"ERROR: video_file_path {out_video_file_path} is not an extant file! We should have produced a video file there."


