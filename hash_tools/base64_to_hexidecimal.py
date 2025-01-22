from base64 import b64decode


def base64_to_hexidecimal(b64_str: str) -> str:
    """
    Converts a base64 string to hexadecimal.

    Parameters:
    b64_str (str): Base64 string to be converted.

    Returns:
    str: Hexadecimal representation of the base64 string.
    """
    assert isinstance(b64_str, str)
    for c in b64_str:
        assert c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    decoded = b64decode(b64_str)
    hex_str = decoded.hex()
    return hex_str
