import PIL.Image
import io


def pil_to_jpg_to_bytes(
    image_pil: PIL.Image.Image
) -> bytes:
    """
    Write the image_pil to JPEG encoding but in memory / RAM.
    TODO: control the quality of the JPEG encoding, 4:2:2, 4:2:0, etc.
    """
    fp = io.BytesIO()
    

    image_pil.save(fp, format="JPEG")
    # save the image as 4:2:0 chroma subsampling:

    data = fp.getvalue()
    assert isinstance(data, bytes)
    return data
