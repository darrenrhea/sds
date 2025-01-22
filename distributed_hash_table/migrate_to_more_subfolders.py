import shutil
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)



shared_dir = get_the_large_capacity_shared_directory()

# this is where we cache files locally:
sha256_local_cache_dir = shared_dir / "sha256"

for p in sha256_local_cache_dir.iterdir():
    if not p.is_file():
        continue
    sha256_hash = p.stem
    assert len(sha256_hash) == 64
    for c in sha256_hash:
        assert c in "0123456789abcdef", "ERROR: that is not a valid sha256 hash!"
    src = p

    d01 = sha256_hash[0:2]
    d23 = sha256_hash[2:4]
    d45 = sha256_hash[4:6]
    d67 = sha256_hash[6:8]

    new_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67
    new_dir.mkdir(parents=True, exist_ok=True)
    dst = new_dir / p.name

    print(f"Moving {src} to {dst}")
    shutil.move(src=src, dst=dst)

