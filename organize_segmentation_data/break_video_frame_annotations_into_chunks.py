from sjaios_save_jsonable_as_its_own_sha256 import (
     sjaios_save_jsonable_as_its_own_sha256
)
from osofpsaj_open_sha256_or_file_path_str_as_json import (
     osofpsaj_open_sha256_or_file_path_str_as_json
)
from chunk_list import (
     chunk_list
)
import numpy as np

uneven_chunks = [
    "37deb6dd165db2a0b1d1ea42ecffa1f1161656526ebc7b1fb0410f37718649b2",
    "f2ffa2041832a30582b2e3bfe9b609480f433a194cae53dfc13ecd2485ef634d",
    "09b0d09f4309313c70484847313b3c12b22c2f2923aa565cc71d261372fb7221",
    "d551828911e76b7e7ac2eae6a61dc96ac67791a28cc66baefd82ec3614b8f303",
]

total = []
for sha256 in uneven_chunks:
    x = osofpsaj_open_sha256_or_file_path_str_as_json(
       sha256
    )
    print(f"{sha256=} has {len(x)=}")
    total.extend(x)

np.random.shuffle(total)

even_chunks = chunk_list(
    lst=total,
    num_chunks=4
)



for chunk in even_chunks:
    sha256 = sjaios_save_jsonable_as_its_own_sha256(
        obj=chunk,
    )
    print(f"{sha256=}")



    