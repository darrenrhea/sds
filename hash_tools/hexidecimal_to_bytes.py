def hexidecimal_to_bytes(hex_str: str) -> bytes: 
    """
    Converts a hexadecimal string to bytes.

    Parameters:
    hex_str (str): Hexadecimal string to be converted.

    Returns:
    bytes: Bytes representation of the hexadecimal string.
    """
    assert isinstance(hex_str, str)
    assert len(hex_str) % 2 == 0
    return bytes.fromhex(hex_str)



