"""
Loop through segmented frames and build the average flattened RHS lane 
This will be used for fake data generation.

TODO: floor-ripper code currently sets pp = (0, 0)!! Change this
"""
from pathlib import Path
import os
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage


def get_list_of_segmented_frames():
    mask_dir = Path("~/r/gsw1/segmentation").expanduser()
    mask_suffix = "_nonlane.png"
    #mask_suffix = "_players.png"
    list_of_frames = []
    for filename in os.listdir(mask_dir):
        if not filename.endswith(mask_suffix):
            continue
        else:
            list_of_frames.append(os.path.basename(filename)[:-len(mask_suffix)])
    return list_of_frames

if __name__ == "__main__":

    pad = 2
    x_bnds = dict(
        left = [-47-pad, -28.17+pad],
        right = [28.17-pad, 47+pad]
    )
    y_min = -7.83 - pad
    y_max = 7.83 + pad

    output_width = 720 # ripped lane image width

    os.system("mkdir tmp")
    # loop through segmentations and load the corresponding frame and camera solve
    cnt = 0
    list_of_segmented_frames = get_list_of_segmented_frames()
    for name in list_of_segmented_frames:
        # Load thee frrame, player mask, and solved camera
        try:
            path_to_frame = Path(f"~/r/gsw1/segmentation/{name}_color.png").expanduser()
            frame_img = Image.open(path_to_frame).convert("RGBA")
        except:
            print(f"Missig frame, not found at {path_to_frame}")
            continue
        try:
            path_to_mask = Path(f"~/r/gsw1/segmentation/{name}_nonlane.png").expanduser()
            mask_img = Image.open(path_to_mask).convert("RGBA")
        except:
            print(f"Missnig mask, not found at {path_to_mask}")
            continue
        try:
            path_to_camera = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/second/{name}_camera_parameters.json").expanduser()
        except:
            print(f"Missnig camera, not found at {path_to_camera}")
            continue
        print(name)

        # dilate the player mask
        num_dilation_iters = 7
        mask_arr = np.array(mask_img)[:,:,-1] # pull out alpha channel
        dilated_mask_arr = ndimage.binary_dilation(mask_arr>0, iterations=num_dilation_iters)
        dilated_mask_img = ImageOps.invert( Image.fromarray(dilated_mask_arr).convert("L") )
        frame_img.putalpha(dilated_mask_img)
        #print("Saving image")
        masked_frame_path = "./tmp/masked_frame_for_avg.png"
        frame_img.save(masked_frame_path)

        # rip both LHS and RHS lanes, reflecting the LHS lane into RHS
        # image will likely only have one of LHS or RHS lanes visible.
        for side in ["left", "right"]:
            x_min, x_max = x_bnds[side]

            output_path = "./tmp/canonical_lane.png"
            exec_string = "~/r/floor-ripper/rip_floor.exe %s %s %s %s %s %s %s %s" % (path_to_camera, x_min, x_max, y_min, y_max, output_width, masked_frame_path, output_path)
            os.system(exec_string)

            ripped_floor_arr = np.array(Image.open(output_path)).astype(int)
            if side=="left": # reflect LHS into RHS canonical form
                ripped_floor_arr = ripped_floor_arr[:,::-1,:]
            rgb = ripped_floor_arr[:,:,:3]
            mask = (ripped_floor_arr[:,:,-1] > 0).astype(int)

            if (mask!=0).sum() < 0.1 * mask.size:
                print(f"Not enough {side} lane visible. Skipping this side.")
            else:
                print(f"Enough {side} lane visible. Addiing to runniing average.")

            # let's not keep every such png
            os.remove(output_path)

            # Compute running average of canonical lane
            if cnt == 0:
                avg = np.zeros_like(ripped_floor_arr)
                avg[:,:,:3] = rgb * mask[:,:,None]
                avg[:,:,-1] = mask
            else:
                count = avg[:,:,-1]
                avg[:,:,:3] = (count[:,:,None] * avg[:,:,:3] + rgb * mask[:,:,None]) 
                avg[:,:,-1] = avg[:,:,-1] + mask
                denom = avg[:,:,-1].copy()
                denom[denom==0] = 1
                avg[:,:,:3] = avg[:,:,:3] / denom[:,:,None]

            # save result
            avg_img = Image.fromarray(avg[:,:,:3].astype(np.uint8))
            avg_img.save(f"./tmp/avg_{cnt}.png")
            cnt += 1

    os.system(f"mv ./tmp/avg_{cnt-1}.png ./avg_rhs_lane.png")
    os.system("rm -rf ./tmp")

    pixels_per_foot = output_width / (x_bnds["right"][1] - x_bnds["right"][0]) 
    lane_mask = np.zeros((avg_img.height, avg_img.width)).astype(np.uint8)
    lane_mask[int(pad*pixels_per_foot):-int(pad*pixels_per_foot), int(pad*pixels_per_foot):-int(pad*pixels_per_foot)] = 255
    avg_img.putalpha(Image.fromarray(lane_mask).convert("L"))
    avg_img.save("./avg_rhs_lane_mask.png")