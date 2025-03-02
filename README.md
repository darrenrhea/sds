# sds: segmentation-data-synthesis / synthetic data sets


## Get some training data

```bash
mkdir -p /shared/all_training_data/floor_not_floor

time aws s3 sync \
s3://awecomai-shared/all_training_data/floor_not_floor \
/shared/all_training_data/floor_not_floor
```

And some fixtures we don't want to commit to the repo

```bash
mkdir -p /shared/fixtures/sds
time aws s3 sync \
s3://awecomai-shared/fixtures \
/shared/fixtures
```

## Docker install

```bash
cd ~
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:awecom/sds
cd ~/sds

# Kill the container if it already exists:
docker stop sds-dev && docker rm sds-dev

# start the container off of nvidia's image:
docker run -dit --gpus all --network host --ipc host --shm-size 128G --ulimit memlock=-1 -v $HOME/sds:/workspaces/sds -v /shared:/shared --name sds-dev nvcr.io/nvidia/pytorch:24.03-py3

# get inside the container:
docker exec -it sds-dev /bin/bash

cd /workspaces/sds

source install_over_nvidias_docker.sh
```

## Test that training on folders of images works:

```bash
cd /workspaces/sds/train_segmentation_models
CUDA_VISIBLE_DEVICES=0 python train_rockets_core.py
```


## Drop the .pt model checkpoint to .onnx


```bash
cd /workspaces/sds/onnx_stuff

export name=u3fasternets-floor-1541frames-1920x1088-darrenjerry_epoch000010
export CUDA_VISIBLE_DEVICES=0
export model_architecture_family_id=u3fasternets
export weights_file_path=/shared/checkpoints/${name}.pt
export onnx_file_path=/shared/onnx/${name}.onnx
mkdir -p /shared/onnx
export out_dir=/shared/onnx

time python dump_to_onnx.py \
${model_architecture_family_id} \
$weights_file_path \
--onnx ${onnx_file_path} \
--original-size 1920,1080 \
--patch-width 1920 \
--patch-height 1088 \
--patch-stride-width 1920 \
--patch-stride-height 1088 \
--pad-height 8 \
--out-dir ${out_dir} \
--model-id-suffix gray \
/shared/fixtures/sds/just_two_frames.json5
```

## Infer with that .onnx model

```bash
# try running the onnx on a frame.  Worry if sigmoid is built in.

mkdir -p /shared/clips/bos-mia-2024-04-21-mxf/frames
cp /shared/fixtures/sds/frames/* /shared/clips/bos-mia-2024-04-21-mxf/frames/

python onnx_infer.py \
--original_suffix _original.jpg \
--frames_dir /shared/clips/bos-mia-2024-04-21-mxf/frames \
--onnx_file_path /shared/onnx/${name}.onnx \
--clip_id bos-mia-2024-04-21-mxf \
--first_frame_index 440694 \
--last_frame_index 440695 \
--step 1 \
--is_logistic_sigmoid_baked_in True
```


## alternative install outside of Docker

```bash
# clone it somewhere:

# we don't want it asking us about recognizing the host, so:
ssh-keyscan -H github.com >> ~/.ssh/known_hosts

git clone git@github.com:awecom/sds

# cd into it:
cd sds

# make a conda environment appropriate for it:
./scripts/create_conda_environment.sh sds

# activate that conda environment:
conda activate sds

# install more public Python libraries:
pip install -r requirements.txt

# install private python libraries:
source scripts/install_python_libraries_in_editable_mode.sh
```
