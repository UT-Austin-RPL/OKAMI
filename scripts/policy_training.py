import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    parser.add_argument("--exptid", type=str, default="0", help="Experiment ID")
    parser.add_argument("--num_epochs", type=int, default=50002, help="Number of epochs")
    args = parser.parse_args()

    annotation_folder = f"annotations/human_demo/{args.human_demo.split('/')[-1].split('.')[0]}"
    dataset_path = os.path.join(annotation_folder, 'rollout/data.hdf5')
    task_name = os.path.basename(annotation_folder).split('_')[0]
    print("Task name: ", task_name)
    
    commands = [
        "python",
        "okami/act/act/imitate_episodes.py",
        "--policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --lr 5e-5 --seed 0",
        "--num_epochs",
        str(args.num_epochs),
        "--task-name",
        task_name,
        "--exptid",
        task_name + '_' + args.exptid,
        "--dataset-path",
        dataset_path
    ]
    command = " ".join(commands)
    print("command=", command)
    os.system(command)

if __name__ == '__main__':
    main()