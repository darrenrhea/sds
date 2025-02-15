def is_sha256(sha256: str):
    """
    Check if the given string is a valid sha256 hash,
    i.e. is 64 characters long and contains only hexadecimal characters.
    """
    
    assert isinstance(sha256, str), f"Expected a string but got {sha256=}"

    if not len(sha256) == 64:
        return False
    
    for c in sha256:
        if c not in "0123456789abcdef":
            return False
    return True
