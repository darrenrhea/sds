# sds: segmentation-data-synthesis / synthetic data sets

## Install

```bash
# clone it somewhere:
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

# train a segmentation model

```bash
cd ~/sds/train_segmentation_models
CUDA_VISIBLE_DEVICES=0 python train_rockets_core.py
```
