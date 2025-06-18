conda install cudatoolkit=11.7 -c nvidia 
conda install nvidia/label/cuda-11.7.0::cuda-nvcc 
conda install nvidia::cuda-cudart-dev 
conda install -c conda-forge gcc_linux-64=9.4.0 gxx_linux-64=9.4.0 
conda install nvidia/label/cuda-11.7.0::cuda-toolkit

# Install the third-party libraries
echo "Installing third-party libraries..."
sh setup_third_party.sh

pip install -r requirements.txt

cd okami/simulation
pip install -e .

cd ../act/detr
pip install -e .

cd ../../..

# output current directory
echo "Current directory: $(pwd)"