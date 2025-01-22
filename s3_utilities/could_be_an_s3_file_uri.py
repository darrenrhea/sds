def could_be_an_s3_file_uri(s3_file_uri):
    """
    Returns whether or not the given string could be an s3 file uri.
    We insist on lower case s in s3://, i.e. we forbid S3://.
    TODO: check bucket name legality.
    """
    assert (
        isinstance(s3_file_uri, str)
    ), "ERROR: you gave a non-string to could_be_an_s3_file_uri"
    if s3_file_uri.endswith("/"):
        return False
    if not s3_file_uri.startswith("s3://"):
        return False
    suffix = s3_file_uri[5:]
    pieces = suffix.split("/")
    if len(pieces) < 2:
        return False
    return True

