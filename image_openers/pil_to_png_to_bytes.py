import PIL.Image
import io


def pil_to_png_to_bytes(
    image_pil: PIL.Image.Image
) -> bytes:
    """
    Write the image_pil to a png file but in memory / RAM.
    """
    fp = io.BytesIO()
    image_pil.save(fp, format="PNG")  
    data = fp.getvalue()
    assert isinstance(data, bytes)
    return data
