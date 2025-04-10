#!/bin/bash

# Run this like:
# source scripts/install_python_libraries_in_editable_mode.sh

# install our own libraries into the conda environment in an editable way:
L=(
alpha_compositing_utils
annotation_utils
better_json
camera_parameters
camera_pose_data
clip_id_utilities
color_calibration
computer_quirks
connected_components
convex_hull_utilities
cutout_utilities
distributed_hash_table
dotflat
Drawable2DImage
fake_basketball
ffmpeg_utilities
file_shit
first_order_jets
flatten_leds
floor_texture_utilities
gpu_utilities
hash_tools
homography_utils
image_openers
image_writers
keypoint_matching_utilities
led_alignment
nuke_lens_distortion
nuke_texture_rendering
onnx_stuff
postgres_utils
primitive_logging
print_image_in_iterm2
rodrigues_utils
rsync_utils
s3_utilities
segmentation_data_utils
segmentation_utils
string_utilities
syntax_highlighting
texture_utils
train_segmentation_models
various_architectures
via_ssh
video_frame_data_utils
)

ls scripts || printf "Run this like source scripts/install_python_libraries_in_editable_mode.sh"

for i in "${L[@]}"
do
    echo installing $i
    (cd $i && pip install --no-deps -e .)
done

