from download_the_files_with_these_sha256s import (
     download_the_files_with_these_sha256s
)
import time
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)


def download_the_files_with_these_sha256s_speedtest():
    
    datapoints_sha256 = "37deb6dd165db2a0b1d1ea42ecffa1f1161656526ebc7b1fb0410f37718649b2"
    datapoints = ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
        sha256_of_the_jsonlike_file=datapoints_sha256,
        check=True
    )
    
    

    sha256s = []
    for datapoint in datapoints:
        label_name_to_sha256 = datapoint["label_name_to_sha256"]
        original_sha256 = label_name_to_sha256["original"]
        floor_not_floor_sha256 = label_name_to_sha256["floor_not_floor"]
        # camera_pose_sha256 = datapoint["camera_pose"]
        sha256s.append(original_sha256)
        sha256s.append(floor_not_floor_sha256)
        # sha256s.append(camera_pose_sha256)
    
    start = time.time()    
    download_the_files_with_these_sha256s(
        sha256s_to_download=sha256s,
        max_workers=10,
        verbose=True
    )
    stop = time.time()
    print(f"Elapsed time: {stop - start} seconds to download {len(sha256s)} files")
    
if __name__ == "__main__":
    download_the_files_with_these_sha256s_speedtest()
    