from SHA256ToS3FileURIResolver import (
     SHA256ToS3FileURIResolver
)
import time
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)
from gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256 import (
     gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256
)



def sha256_to_s3_file_uri_resolution_speed_test():
    
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
    
    # sha256s = sha256s[:20]

    sha256_to_s3_file_uri_resolver = SHA256ToS3FileURIResolver()

    start = time.time()
    for sha256 in sha256s:
        print(sha256)
    
        # answer = gaes3fuwts_get_an_extant_s3_file_uri_with_this_sha256(
        #     sha256=sha256
        # )
        answer = sha256_to_s3_file_uri_resolver.get(sha256)

        print(answer)
    end = time.time()
    duration = end - start
    time_per_resolution = duration / len(sha256s)
    print(f"Took {duration=} to resolve {len(sha256s)=} sha256s.")
    print(f"{time_per_resolution=}")
   
if __name__ == "__main__":
    sha256_to_s3_file_uri_resolution_speed_test()