pip install colorama
python -c "import cv2 ; print('still working after colorama')"

# consider not installing fusedAdan via this:
cd ~
git clone https://github.com/sail-sg/Adan.git
cd Adan
time python3 setup.py install --unfused

python -c "import cv2 ; print('still working after Adan')"


pip install albumentations==1.3.1 --no-deps

python -c "import cv2 ; print('still working after albumentations')"

pip install qudida --no-deps

python -c "import cv2 ; print('still working after qudida')"


pip install segmentation-models-pytorch

python -c "import cv2 ; print('still working after segmentations-models-pytorch')"


pip install scikit-image

python -c "import cv2 ; print('still working after scikit-image')"


pip install jsonlines

python -c "import cv2 ; print('still working after jsonlines')"

pip install pyjson5

python -c "import cv2 ; print('still working after pyjson5')"


pip install onnxruntime-gpu

python -c "import cv2 ; print('still working after onnxruntime-gpu')"

echo 'pip freeze before is:'
pip freeze

# Seems that installing this version of opencv is the problem is the problem:supposedly this will block mmcv from doing bad:
# pip install opencv-python-headless==4.7.0.72

echo "pip freeze before is:"
pip freeze

python -c "import cv2 ; print('still working after opencv-python-headless')"

# pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html

pip install mmcv==2.2.0 --no-deps -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmengine==0.10.6 --no-deps -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install yapf
pip install addict
pip install termcolor
python -c "import mmcv ; print('can import mmcv')"


python -c "import cv2 ; print('import cv2 still working after mmcv manual install')"

pip install awscli

python -c "import cv2 ; print('still working after awscli')"
