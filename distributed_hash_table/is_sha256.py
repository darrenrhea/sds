def is_sha256(sha256: str):
    """
    Check if a string is a valid sha256 hash,
    i.e. is 64 characters long and contains only hexadecimal characters.
    """
    
    assert isinstance(sha256, str)

    assert len(sha256) == 64
    for c in sha256:
        if c not in "0123456789abcdef":
            return False
    return True
