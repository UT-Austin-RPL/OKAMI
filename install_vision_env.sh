conda install cudatoolkit=11.7 -c nvidia 
conda install nvidia/label/cuda-11.7.0::cuda-nvcc 
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install nvidia::cuda-cudart-dev
conda install -c conda-forge gcc_linux-64=9.4.0 gxx_linux-64=9.4.0 
conda install mkl==2024.0
conda install git

git clone --recursive https://github.com/princeton-vl/DROID-SLAM third-party/DROID-SLAM
git clone --recursive https://github.com/ViTAE-Transformer/ViTPose third-party/ViTPose

conda install nvidia/label/cuda-11.7.0::cuda-toolkit 

pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -e .
pip install -r requirements.txt

pip install -v -e third-party/ViTPose

cd third-party/DROID-SLAM
python setup.py install

# ---------------------------------------

# hamer
cd ../../third_party
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose

sed -i '14s|.*|ROOT_DIR = os.path.dirname(os.path.abspath(__file__))|' ./vitpose_model.py

# ---------------------------------------

# download hamer data
bash fetch_demo_data.sh

# download slahmr data
cd ../../okami/slahmr_hands
gdown https://drive.google.com/uc?id=1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW
unzip -q slahmr_dependencies.zip
rm slahmr_dependencies.zip

# copy hamer data to slahmr directory
cp ../../third_party/hamer/hamer_demo_data.tar.gz ./
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz

sed -i '5s/".\/_DATA"/os.path.abspath(f"{__file__}\/..\/..\/..\/..\/..\/_DATA")/' ../../third_party/hamer/hamer/configs/__init__.py

# copy smpl models into _DATA
cd ../../
cp configs/smpl_models/mano/MANO_RIGHT.pkl third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl
cp configs/smpl_models/mano/MANO_RIGHT.pkl okami/slahmr_hands/_DATA/data/mano/MANO_RIGHT.pkl
mkdir okami/slahmr_hands/_DATA/body_models/smplx
cp configs/smpl_models/smplx/SMPLX_NEUTRAL.npz okami/slahmr_hands/_DATA/body_models/smplx/SMPLX_NEUTRAL.npz

ln -n -s third_party/hamer/_DATA ./

pip install easydict h5py ruptures scikit-learn ujson open3d
conda install ffmpeg
pip install numpy==1.26.3
pip install pytorch-lightning==1.9.5
pip install torch==1.13.0