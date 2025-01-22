# sds: segmentation-data-synthesis / synthetic data sets

## Install

```bash
# clone it somewhere:
git clone git@github.com:darrenrhea/sds

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

Make fake important people convention
=====================================

cd fake_basketball
python make_people_free_backgrounds_by_overwriting_people_with_floor_texture_and_led_ads.py

python make_fake_important_people_annotations_from_people_free_backgrounds.py