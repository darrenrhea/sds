import struct
from pathlib import Path

def png_bit_depth(path: str | Path) -> int:
    """
    Return the **bit depth per channel** (8 or 16) of a PNG image by
    inspecting only its header.

    The function touches at most 33 bytes (the 8-byte PNG signature plus the
    fixed‐size 25-byte *IHDR* chunk header), making it essentially as fast as
    a single `stat` call.

    Parameters
    ----------
    path : str | Path
        File-system path to the `.png` file.

    Returns
    -------
    int
        `8` for an 8-bit PNG (or any PNG whose bit depth ≤ 8)  
        `16` for a 16-bit PNG.

    Raises
    ------
    ValueError
        If the file is not a valid PNG or the header is truncated.

    Notes
    -----
    *PNG layout recap*  
    ```
    Offset  Size  Description
    ------  ----  ------------------------------------------
      0     8     PNG signature (b'\x89PNG\r\n\x1a\n')
      8     4     Length of IHDR data  (must be 13)
     12     4     Chunk type           (b'IHDR')
     16     13    IHDR data:
                    [0-3] Width  (uint32)
                    [4-7] Height (uint32)
                    [8]   Bit-depth        <── we need this byte
                    …
     29     4     CRC (ignored here)
    ```
    The *bit-depth* field stores one of {1, 2, 4, 8, 16}.  Values 1–8 are
    encoded with 8 bits or fewer per channel; 16 denotes 16 bits per channel.
    """
    path = Path(path)
    with path.open("rb") as fp:
        # 1. Validate PNG signature (8 bytes)
        if fp.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path!s} is not a PNG file")

        # 2. Read IHDR chunk length and type (8 bytes)
        length_bytes = fp.read(4)
        if len(length_bytes) != 4:
            raise ValueError("File truncated before IHDR length")
        (ihdr_len,) = struct.unpack(">I", length_bytes)
        if ihdr_len != 13:  # Spec mandates exactly 13
            raise ValueError("Corrupted PNG: IHDR length ≠ 13")

        if fp.read(4) != b"IHDR":
            raise ValueError("Corrupted PNG: IHDR is not the first chunk")

        # 3. Grab IHDR data (13 bytes) and extract the 9th byte (bit-depth)
        ihdr_data = fp.read(13)
        if len(ihdr_data) != 13:
            raise ValueError("File truncated inside IHDR data")

        bit_depth_byte = ihdr_data[8]
        return 16 if bit_depth_byte == 16 else 8