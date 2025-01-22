"""
ssh lam

cd ~/felix/popsift/build


Linux-x86_64/match_keyframes /home/drhea/awecom/data/clips/swinney1/frames/swinney1_004613.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004700.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004800.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004900.jpg /home/drhea/awecom/data/clips/swinney1/frames out

mkdir ~/awecom/data/clips/swinney1/some_frames/

for x in `seq 8000 10000` ; do cp $(printf "$HOME/awecom/data/clips/swinney1/frames/swinney1_%06d.jpg" $x) $(printf "$HOME/awecom/data/clips/swinney1/some_frames/swinney1_%06d.jpg" $x); done

Linux-x86_64/match_keyframes /home/drhea/awecom/data/clips/swinney1/frames/swinney1_004613.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004700.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004800.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004900.jpg /home/drhea/awecom/data/clips/swinney1/some_frames out

Back on the laptop:
mkdir -p ~/felix/popsift/build/out/
rsync -r --progress lam:/home/drhea/felix/popsift/build/out/ ~/felix/popsift/build/out/

"""
import numpy as np
import better_json as bj
from pathlib import Path

sorted_keyframe_indices = [4613, 4700, 4800, 4900]

for frame_index in range(4600, 11000+1):
    file_path = Path(
        f"~/felix/popsift/build/out/swinney1_{frame_index:06d}.hom"
    ).expanduser()
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        assert len(lines) == len(sorted_keyframe_indices)
        for line in lines:
            keyframe_ind_str, num_desc_str, num_matched_str, H_str = line.strip().split(';')
            keyframe_ind = int(keyframe_ind_str)
            keyframe_index = sorted_keyframe_indices[keyframe_ind]
            num_desc = int(num_desc_str)
            num_matched = int(num_matched_str)
            success = num_matched > 10
            H_nums = [float(s) for s in H_str.split(',')]
            H = np.array(H_nums).reshape((3,3))
            jsonable = dict(
                success=success,
                homography_in_pixel_units=[
                    [float(H[i, j]) for j in range(3)]
                    for i in range(3)
                ],
                num_descriptions=num_desc,
                num_matched=num_matched,
            )

            out_path = Path(
                f"~/awecom/data/clips/swinney1/homography_attempts/felix/{frame_index:06d}_into_{keyframe_index:06d}.json"
            ).expanduser()

            bj.dump(fp=out_path, obj=jsonable)
            

