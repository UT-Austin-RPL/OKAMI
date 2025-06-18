#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.10 -m venv .slahmr_hands
echo "Activating virtual environment"

source $PWD/.slahmr_hands/bin/activate

# install pytorch
$PWD/.slahmr_hands/bin/pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# torch-scatter
$PWD/.slahmr_hands/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# install PHALP
$PWD/.slahmr_hands/bin/pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

# install HaMeR
git clone --recursive git@github.com:geopavlakos/hamer.git
cd hamer
$PWD/.slahmr_hands/bin/pip install -e .[all]
cd ..

# install source
$PWD/.slahmr_hands/bin/pip install -e .

# install remaining requirements
$PWD/.slahmr_hands/bin/pip install -r requirements.txt

# install ViTPose
$PWD/.slahmr_hands/bin/pip install -v -e third-party/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
$PWD/../../.slahmr_hands/bin/python setup.py install
cd ../..
