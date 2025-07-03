from get_font_file_path import (
     get_font_file_path
)
from PIL import ImageFont

def get_font_for_use_with_pil_image_drawable(
    font_size: int
):
    font_path = get_font_file_path()
    font = ImageFont.truetype(str(font_path), 16)
    return font

