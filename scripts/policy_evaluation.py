import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    parser.add_argument("--exptid", type=str, default="0", help="Experiment ID")
    parser.add_argument("--num_epochs", type=int, default=50002, help="Number of epochs")
    parser.add_argument("--ckpt", type=int, default=50000, help="Number of epochs to resume from")
    parser.add_argument("--environment", type=str, default="HumanoidPour", help="Environment to evaluate in")
    args = parser.parse_args()

    annotation_folder = f"annotations/human_demo/{args.human_demo.split('/')[-1].split('.')[0]}"
    dataset_path = os.path.join(annotation_folder, 'rollout/data.hdf5')
    task_name = os.path.basename(annotation_folder).split('_')[0]
    expid = task_name + '_' + args.exptid
    
    print("Task name: ", task_name, "Experiment ID: ", expid)
    
    commands = [
        "python",
        "okami/act/evaluation/sim_evaluation.py",
        "--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --lr 5e-5 --seed 0",
        "--num_epochs",
        str(args.num_epochs),
        "--task-name",
        task_name,
        "--exptid",
        expid,
        "--dataset-path",
        dataset_path,
        "--resume_ckpt",
        str(args.ckpt),
        "--environment",
        args.environment
    ]
    command = " ".join(commands)
    print("command=", command)
    os.system(command)
    
    # copy video into the annotation folder
    past_video_path = os.path.join('okami/act/videos', f"{expid}_eval_{args.ckpt}.mp4")
    video_path = os.path.join(annotation_folder, 'rollout', 'policy_evaluation.mp4')
    os.system(f"cp {past_video_path} {video_path}")
    print("Copied video to ", video_path)

if __name__ == '__main__':
    main()