import re
from typing import Optional


def pgallgd_parse_god_awful_lambda_labs_gpu_description(
    gpu_description_str: str
) -> Optional[int]:
    """
    Search for a substring in the format: an integer, a space, then 'GB'.
    If found, return the substring; otherwise, return None.
    """
    s = gpu_description_str
    # Use word boundaries to ensure proper matching
    pattern = r'\b(\d+)\s?GB\b'
    match = re.search(pattern, s)
    if match:
        return int(match.group(1))
    return None


    # Example usage:
    if __name__ == '__main__':
        xys = [
            ("This computer has 128 GB of storage.", 128),
            ("The smartphone comes with 256 GB memory.", 256),
            ("No storage info here.", None),
            ("It has 512GB which is not separated by a space.", 512),
        ]
        
        for x, should_be in xys:
            result = pgallgd_parse_god_awful_lambda_labs_gpu_description(x)
            assert result == should_be
        