#!/bin/bash

# This makes a conda environment for the project.
# Choose a name for the conda environment, let's say sds
# and then do:
# ./create_conda_environment.sh sds
conda_environment_name=$1
# bail as soon as something does not work:
set -e

# echo the commands as they happen:
set -x

# crash if conda not installed where we think it should be:
ls ~/miniconda3/bin/conda

# we need conda to work in this bash script:
__conda_setup="$(~/miniconda3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"

eval "$__conda_setup"

# crash if there is no conda:
which conda

# this does not error even if the environment does not exist:
# conda env remove --name $conda_environment_name

rm -rf /home/ubuntu/miniconda3/envs/sds

if [[ $(uname) = "Darwin" ]]
then
    echo "we seem to be on MacOS"
    pytorch_package='pytorch::pytorch'
else    
    echo "We seem to be on Linux"
    pytorch_package='pytorch-cuda=12.4'
    # added advantage of 12.4 is that you get cuDNN 9.x
fi

conda create -y \
-n $conda_environment_name \
python=3.11 \
pytorch \
torchvision \
torchaudio \
"$pytorch_package" \
numpy \
scipy \
ipython \
matplotlib \
ipykernel \
pandas \
pillow \
imageio \
notebook \
scikit-image \
scipy \
-c pytorch \
-c nvidia

