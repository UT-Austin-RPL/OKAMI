# OKAMI_release

## Installation

```
conda create -n okami python=3.9
conda activate okami
```

Run the following command in root directory to install packages and third parties:
```
sh install_okami_env.sh
```

Set your `OPENAI_API_KEY` in the environment variables.

### Vision modules

You need to first download some smpl models from official websites:

1. Please visit the [MANO website](https://mano.is.tue.mpg.de/) and register to get access to the downloads sections. Download MANO models and put `MANO_RIGHT.pkl` in folder `configs/smpl_models/mano`. Download smplh models and put TODO.
2. Please visit the [SMPLX website](https://smpl-x.is.tue.mpg.de/) and register to get access to the downloads sections. Download SMPLX models and put `SMPLX_NEUTRAL.npz` in folder `configs/smpl_models/smplx`.


Create a new environment for human body reconstruction from videos.
```
conda create -n hamer python=3.10
conda activate hamer
```

Run the following command in root directory to install packages and third parties:
```
sh install_vision_env.sh
```

For common trouble-shooting in this part, you can refer to [this](https://github.com/vye16/slahmr/issues).

<!-- ### Directory structure of `third_party`

The resulting directory structure should look like this:

```
_DATA # for hamer
third_party/
    co-tracker/
    Cutie/
    dinov2/
    GR1_retarget/
    GroundingDINO/
    hamer/
    sam_checkpoints/
    segment-anything/
``` -->

## Stage 1: Reference Manipulation Plan Generation

Record an rgbd human video, and save it as an hdf5 file with the following structure:
```
data (Group)
    attrs:
        data_config: (dict)
            intrinsics: (dict)
                front_camera: (dict)
                    fx
                    fy
                    cx
                    cy
            extrinsics: (dict)
                front_camera: (dict)
                    translation
                    rotation
    human_demo (Group)
        obs (Group)
            agentview_depth (Dataset) : shaped (len, h, w)
            agentview_rgb (Dataset): shaped (len, h, w, c)
```

Put the hdf5 file in folder `datasets/rgbd`.
Example hdf5 files can be downloaded from [here](https://drive.google.com/drive/folders/1pA-fp_fnwdxLCLEfESq-NelgQRAJbgGi?usp=sharing).
```
# move back to the root directory
mkdir -p datasets/rgbd
gdown --folder https://drive.google.com/drive/folders/1pA-fp_fnwdxLCLEfESq-NelgQRAJbgGi?usp=sharing -O datasets/rgbd/
cd datasets/rgbd && find OKAMI\ data -name "*.hdf5" -exec mv {} ./ \; && rm -rf OKAMI\ data && cd ../../
```

Then simply run `sh run_plan_generation.sh HDF5_FILE_PATH`, where `HDF5_FILE_PATH` is the path to the hdf5 file you just saved. Or, you can run the following commands step by step:
```
conda activate okami
python scripts/pipeline.py --human-demo HDF5_FILE_PATH
conda activate hamer
python scripts/06_process_hands.py --human-demo HDF5_FILE_PATH
python scripts/07_human_motion_reconstruction.py --human-demo HDF5_FILE_PATH
conda activate okami
python scripts/08_generate_plan.py --human-demo HDF5_FILE_PATH
```

All results will be saved to the annotation folder `annotations/human_demo/DEMO_NAME/`.

## Stage 2: Object-aware retargeting

Run the following command to simulate the object-aware retargeting process. Currently support two simulation environments: HumanoidPour and HumanoidDrawer. The HumanoidPour environment is used for `salt_demo.hdf5`, and the HumanoidDrawer environment is used for `drawer_demo.hdf5`.
```
python scripts/oar_sim.py --no-vis --num-demo 100 --human-demo HDF5_FILE_PATH --environment SIMULATION_ENVIRONMENT
```
where `SIMULATION_ENVIRONMENT` can be either `HumanoidPour` or `HumanoidDrawer`. The resulting rollout trajectories will be saved in the `annotations/human_demo/DEMO_NAME/rollout/` directory.

To convert the saved pkl data into a robomimic format hdf5 file, use the following command:
```
python scripts/convert_to_hdf5_dataset.py --human-demo HDF5_FILE_PATH
```
The converted dataset will be saved in the `annotations/human_demo/DEMO_NAME/rollout/data.hdf5` file.

## Policy Learning

Training:
```
python scripts/policy_training.py --num_epochs 80002 --human-demo HDF5_FILE_PATH
```

Evaluation in simulation:
```
python scripts/policy_evaluation.py --num_epochs 80002 --ckpt 80000 --human-demo HDF5_FILE_PATH --environment SIMULATION_ENVIRONMENT
```
where `SIMULATION_ENVIRONMENT` can be either `HumanoidPour` or `HumanoidDrawer`. 