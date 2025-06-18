mkdir third_party
cd third_party

# Cutie
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
git checkout ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a
pip install -e .
python cutie/utils/download_models.py
cd ..

# cotracker
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
git checkout 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
cd ..

# dinov2
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout fc49f49d734c767272a4ea0e18ff2ab8e60fc92d
pip install -r requirements.txt
pip install -e .
cd ..

# SAM
git clone https://github.com/facebookresearch/segment-anything
cd segment-anything
git checkout 6fdee8f2727f4506cfbbe553e23b895e27956588
pip install -e .
cd ..

# SAM Checkpoints
mkdir sam_checkpoints
cd sam_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# Grounded-SAM
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
git checkout 856dde20aee659246248e20734ef9ba5214f5e44
python setup.py build
python setup.py install
cd ..

# GR1_retarget
git clone https://github.com/UT-Austin-RPL/OKAMI.git -b retarget
# rename the cloned directory to GR1_retarget
mv OKAMI_release GR1_retarget
cd GR1_retarget
pip install -e .
cd ..

cd ..