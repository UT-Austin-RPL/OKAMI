# generate smplh_traj.pkl and put into annotation folder.
import os
from datetime import datetime
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    args = parser.parse_args()
    
    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])
    
    video_name = args.human_demo.split("/")[-1].split(".")[0] + ".mp4"
    video_path = os.path.join(annotation_path, video_name)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    slahmr_demo_path = os.path.join(current_dir, "../okami/slahmr_hands/demo")
    slahmr_video_path = os.path.join(slahmr_demo_path, 'videos')
    
    os.makedirs(slahmr_video_path, exist_ok=True)
    
    # copy video slahmr directory
    os.system(f"cp {video_path} {slahmr_video_path}")
    
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    date = year + '-' + month + '-' +  day
    
    print("*************Run Slahmr Hands*************")
    commands = [
        "python",
        "okami/slahmr_hands/slahmr/run_opt.py",
        "data=video",
        f"data.seq={video_name.split('.')[0]}",
        f"data.root={slahmr_demo_path}",
        "run_opt=True",
        "run_vis=True",
    ]
    command = " ".join(commands)
    print("command=", command)
    os.system(command)
    
    result_folder = 'outputs/logs/video-val/' + date + f'/{video_name.split(".")[0]}-all-shot-0-0-600'
    assert os.path.exists(result_folder), f"result_folder={result_folder} does not exist"
    
    # find the max epoch in motion chunks result
    max_epoch = -1
    for file in os.listdir(os.path.join(result_folder, 'motion_chunks')):
        if file.endswith('.npz'):
            epoch = int(file.split('_')[-3])
            if epoch > max_epoch:
                max_epoch = epoch
                
    if max_epoch > -1: # use motion chunks result
        max_epoch_str = '{:06d}'.format(max_epoch)
        motion_chunks_file = os.path.join(result_folder, f'motion_chunks/{video_name.split(".")[0]}_{max_epoch_str}_world_results.npz')
        assert os.path.exists(motion_chunks_file), f"motion_chunks_file={motion_chunks_file} does not exist"
        result_file = motion_chunks_file
    else:  # use smooth fit result
        smooth_fit_file = os.path.join(result_folder, f'smooth_fit/{video_name.split(".")[0]}_000060_world_results.npz')
        assert os.path.exists(smooth_fit_file), f"smooth_fit_file={smooth_fit_file} does not exist"
        result_file = smooth_fit_file

    print("result_file=", result_file)
    
    print("*************Post Processing*************")
    commands = [
        "python",
        "scripts/smplh_normalization.py",
        f"-i {result_file}",
        f"-o {os.path.join(annotation_path, 'smplh_traj.pkl')}",
    ]
    command = " ".join(commands)
    os.system(command)
    
    print("result saved in", os.path.join(annotation_path, 'smplh_traj.pkl'))


if __name__ == '__main__':
    main()