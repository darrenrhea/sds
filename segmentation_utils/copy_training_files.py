import shutil
import pathlib
from pathlib import Path


# avoid_list = [
#   "009408",
#   "048143",
#   "048546",
#   "048550",
#   "048573",
#   "048583",
#   "048669",
#   "048714",
#   "048728",
#   "048744",
#   "048747",
#   "048749",
#   "048752",
#   "048772",
#   "048788",
#   "048803",
#   "048819",
#   "048843",
#   "048863",
#   "048868",
#   "049815",
#   "049824",
#   "049846",
#   "049848",
#   "049849",
#   "059999",
#   "060015",
#   "060020",
#   "060043",
#   "060059",
#   "060077",
#   "060190",
#   "060209",
#   "060215",
#   "060225",
#   "060238",
#   "060270",
#   "060296",
#   "060433"
# ]

# copy_list = [
#   "151367",
#   "151316",
#   "150474",
#   "150309",
#   "150288",
#   "150270",
#   "150244",
#   "150209",
#   "150135",
#   "150123",
#   "150108",
#   "150106",
#   "150103",
#   "151466",
#   "156740",
#   "156796",
#   "560523",
#   "558605",
#   "515180",
#   "486384",
#   "151607",
#   "483391",
#   "352775",
#   "349896",
#   "341050",
#   "265654",
#   "638145",
#   "606879",
#   "567440",
#   "558476",
#   "638156",
#   "593013",
#   "263978",
#   "263963",
#   "638749",
#   "638554",
#   "637614",
#   "512679",
#   "511859",
#   "638992",
#   "638898",
#   "567063",
#   "515021",
#   "263878",
#   "335946",
#   "328822",
#   "328107",
#   "327864",
#   "639126",
#   "638381",
#   "592840",
#   "198395",
#   "194826",
#   "190596",
#   "190729",
#   "190784",
#   "191083",
#   "191197",
#   "191462",
#   "194811",
#   "195150",
#   "195776",
#   "197047",
#   "197112",
#   "197742",
#   "197779",
#   "198258",
#   "198292",
#   "198446",
#   "199063",
#   "199153",
#   "199176",
#   "199242",
#   "199281",
#   "199596",
#   "199625",
#   # "310850",
#   # "311042",
#   # "311180",
#   # "344942",
#   "189307",
#   "188917",
#   "188460",
#   "188341",
#   "188239",
#   "188045",
#   "188012",
#   "187207",
#   "187185",
#   "187130",
#   "185448",
#   "184432",
#   "181876",
#   "179887",
#   "502373",
#   "178814",
#   "178778",
#   "178212",
#   "633580",
#   "627735",
#   "626734",
#   "601423",
#   "575048",
#   "571538",
#   "504264",
#   "503654",
#   "503622",
#   "498445",
#   "498428",
#   "482396",
#   "633140",
#   # "601994",
#   # "601976",
#   "574173",
#   # "574100",
#   "482793",
#   "306978",
#   "571585",
#   "503712",
#   "498468",
#   "482745",
#   "482715",
#   "305900",
#   "220999",
#   "151403",
#   "158257",
#   "158302",
#   "158305",
#   "158322",
#   "159329",
#   "171913",
#   "172673",
#   "571563",
#   "574585",
#   "574994",
#   "575619",
#   "632707"
# ]

copy_list = [
    "DSCF0241_000102_nonfloor.png",
	"DSCF0241_001977_nonfloor.png",
	"DSCF0241_002334_nonfloor.png",
	"DSCF0241_002680_nonfloor.png",
	"DSCF0241_006596_nonfloor.png",
	"DSCF0241_007432_nonfloor.png",
	"DSCF0241_007688_nonfloor.png",
	"DSCF0241_008008_nonfloor.png",
	"DSCF0241_008288_nonfloor.png",
	"DSCF0241_001023_nonfloor.png",
	"DSCF0241_000471_nonfloor.png",
	"DSCF0241_000321_nonfloor.png",
	"DSCF0236_000475_nonfloor.png",
	"DSCF0240_000870_nonfloor.png",
	"DSCF0241_002082_nonfloor.png",
	"DSCF0241_004046_nonfloor.png",
	"DSCF0241_000392_nonfloor.png",
	"DSCF0240_000790_nonfloor.png",
	"DSCF0241_001109_nonfloor.png",
	"DSCF0241_002181_nonfloor.png",
	"DSCF0241_002286_nonfloor.png",
	"DSCF0241_008495_nonfloor.png",
	"DSCF0241_008322_nonfloor.png",
	"DSCF0241_004089_nonfloor.png",
	"DSCF0236_000900_nonfloor.png",
	"DSCF0240_001560_nonfloor.png",
	"DSCF0236_000614_nonfloor.png",
	"DSCF0241_008081_nonfloor.png",
    "DSCF0240_001100_nonfloor.png",
	"DSCF0241_000770_nonfloor.png",
    "DSCF0241_002560_nonfloor.png",
    "DSCF0236_000040_nonfloor.png",
	"DSCF0240_000660_nonfloor.png",
	"DSCF0240_001267_nonfloor.png",
	"DSCF0241_000280_nonfloor.png",
	"DSCF0241_000353_nonfloor.png"
]

# from_dir = Path(f"~/r/chicago4k").expanduser()
# to_dir = Path(f"	").expanduser()
# file_prefix = "chicago4k_inning1"
# frame_count = 0
# num_avoids = 0
# from_dir = Path(f"~/r/hou-lac-2023-11-14_alpha_mattes").expanduser()
# from_originals_dir = Path(f"/media/drhea/muchspace/clips/hou-lac-2023-11-14/frames").expanduser()
from_dir = Path(f"~/r/munich4k_floor").expanduser()
from_originals_dir = Path(f"/media/drhea/muchspace/clips/hou-lac-2023-11-14/frames").expanduser()
# to_dir = Path(f"~/hou-lac-2023-11-14_no-atnt_aematter").expanduser()
to_dir = Path(f"~/new_approved_alpha_mattes").expanduser()
file_prefix = "hou-lac-2023-11-14"
frame_count = 0
for p in from_dir.rglob("*"):
    parent_path = p.parent
    # print(parent_path)
    if p.is_file():
        if str(p).endswith('nonfloor.png'):
            # print(p)
            frame_index = p.stem.split('_')[-2]
            # print(f"frame index {frame_index}")
            # if frame_index not in avoid_list:
            if frame_index in copy_list:  
                print(f"{frame_index=}")
                print(f"{p=}")          
                # original_jpg = Path('_'.join(p.stem.split('_')[:-1]) + ".jpg")
                # from_original_path = from_originals_dir / original_jpg
                # print(f"{from_original_path=}")
                to_mask_path = to_dir / p.name
                print(f"{to_mask_path=}")
                to_jpg_path = to_dir / original_jpg
                print(f"{to_jpg_path=}")
                # shutil.copy(p, to_mask_path)
                # shutil.copy(from_original_path, to_jpg_path)
                frame_count += 1
print(f"found {frame_count} annotations")
print(f"num files to copy {len(copy_list)}")
# print(f"num avoids {num_avoids}")
# print(f"length of avoids {len(avoid_list)}")

# time python train.py effs /mnt/nas/volume1/videos/chicago4k_train_data --dataset-kind nonfloor --patch-width 1152 --patch-height 1152 --batch-size 8 --epochs 1000 --train --test-size 10

# time python infer2.py effs /mnt/nas/volume1/videos/checkpoints/effs_1152x1152_chicago4k_train_data_nonfloor_251f_epoch014.pt --original-size 3840,2160 --patch-width 1152 --patch-height 1152 --patch-stride-width 576 --patch-stride-height 576 --out-dir /mnt/nas/volume1/videos/baseball/masks/chicago4k_inning1 --model-id-suffix weight50-14epochs-251f-1152x1152-halfoverlap "/mnt/nas/volume1/videos/baseball/clips/chicago4k_inning1/43154to43726/chicago4k_inning1_*.jpg"

# mev_make_evaluation_video --clip_id chicago4k_inning1 --first_frame_index 43154 --last_frame_index 43726 --model_id weight50-14epochs-251f-1152x1152-halfoverlap --fps 50

# aws s3 cp /mnt/nas/volume1/videos/show_n_tell/chicago4k_inning1_segmentation_pitch_view.mp4 s3://awecomai-show-n-tell/

# time python train.py effs /mnt/nas/volume1/videos/asm-efs-2023-11-14_train_data --dataset-kind nonfloor --patch-width 768 --patch-height 768 --batch-size 16 --epochs 1000 --train --test-size 10

# time python infer2.py effs /mnt/nas/volume1/videos/checkpoints/effs_768x768_asm-efs-2023-11-14_train_data_nonfloor_22f_epoch066.pt --original-size 1920,1080 --patch-width 768 --patch-height 768 --patch-stride-width 384 --patch-stride-height 384 --out-dir /mnt/nas/volume1/videos/asm-efs-2023-11-14/masks --model-id-suffix weight99-66epochs-22f-768x768-halfoverlap "/mnt/nas/volume1/videos/asm-efs-2023-11-14/frames/118001to128000/asm-efs-2023-11-14_*.jpg"

# mev_make_evaluation_video --clip_id asm-efs-2023-11-14 --first_frame_index 118001 --last_frame_index 128000 --model_id weight99-66epochs-22f-768x768-halfoverlap --fps 50

# /usr/local/bin/ffmpeg
# -y
# -nostdin
# -start_number
# 108000
# -framerate
# 59.94
# -i
# {original_frames_path}/chicago4k_inning1_%06d.jpg
# -frames
# {frame_num}
# -vf
# drawtext=fontfile=/awecom/misc/arial.ttf: text='%{{frame_num}}': start_number={frame_ranges[i][1] + 1}: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5
# -vcodec
# libx264
# -pix_fmt
# yuv420p
# -crf
# 18
# show_n_tell_path/{file_prefix}_{frame_ranges[i][1] + 1}to{frame_ranges[i+1][0] - 1}.mp