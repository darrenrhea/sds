cd /workspaces/sds
pip install colorama

# consider not installing fusedAdan via this:
cd /
git clone https://github.com/sail-sg/Adan.git
cd Adan
time python3 setup.py install --unfused

cd /workspaces/sds
source scripts/install_python_libraries_in_editable_mode.sh

pip install albumentations==1.3.1 --no-deps
pip install qudida --no-deps

pip install segmentation-models-pytorch

pip install scikit-image


pip install jsonlines
pip install pyjson5

pip install onnxruntime-gpu