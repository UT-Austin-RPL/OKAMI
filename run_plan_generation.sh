eval "$(conda shell.bash hook)"

if [ "$#" -ne 1 ]; then
    echo "Usage: sh run_plan_generation.sh <HDF5_FILE_PATH>"
    exit 1
fi
HDF5_FILE_PATH=$1

conda activate okami
python scripts/pipeline.py --human-demo $HDF5_FILE_PATH

conda activate hamer
python scripts/06_process_hands.py --human-demo $HDF5_FILE_PATH
python scripts/07_human_motion_reconstruction.py --human-demo $HDF5_FILE_PATH

conda activate okami
python scripts/08_generate_plan.py --human-demo $HDF5_FILE_PATH