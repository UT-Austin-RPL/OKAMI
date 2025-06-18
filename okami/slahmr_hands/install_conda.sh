#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=slahmr_hands

conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME

# install pytorch using pip, update with appropriate cuda drivers if necessary
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
# uncomment if pip installation isn't working
# conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install pytorch scatter using pip, update with appropriate cuda drivers if necessary
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# uncomment if pip installation isn't working
# conda install pytorch-scatter -c pyg -y

# install PHALP
pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

# install HaMeR
git clone --recursive git@github.com:geopavlakos/hamer.git
cd hamer
pip install -e .[all]
cd ..

# install remaining requirements
pip install -r requirements.txt

# install source
pip install -e .

# install ViTPose
pip install -v -e third-party/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
python setup.py install
cd ../..
