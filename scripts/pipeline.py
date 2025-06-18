import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='datasets/rgbd/salt_demo.hdf5')
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument('--tap-pen', type=float, default=10, help='Penalty for changepoint detection.')
    args = parser.parse_args()

    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    ret = input('Generate text using vlm? (y/n): ')
    if ret == 'y':
        while True:
            # 1. generate text description
            print("*************Text Description*************")
            commands = [
                "python",
                "scripts/01_generate_descriptions.py",
                "--human_demo",
                args.human_demo,
            ]
            command = " ".join(commands)                                    
            os.system(command)
            
            ret = input('\nRerun vlm text generation? (y/n): ')
            if ret != 'y':
                break

    ret = input('\nRun GAM Annotation? (y/n): ')
    if ret == 'y':
        # 2. gam annotation
        print("*************GAM Annotation*************")
        commands = [
            "python",
            "scripts/02_gam_annotation.py",
            "--human_demo",
            args.human_demo,
        ]
        command = " ".join(commands)
        os.system(command)

    ret = input('\nproceed? (y/n): ')
    if ret != 'y':
        exit(0)

    # 3. cutie segmentation
    print("*************Cutie Segmentation*************")
    commands = [
        "python",
        "scripts/03_cutie_annotation.py",
        "--annotation-folder",
        annotation_path
    ]
    command = " ".join(commands)
    os.system(command)

    # 4. cotracker annotation
    print("*************Cotracker Annotation*************")
    commands = [
        "python",
        "scripts/04_generate_cotracker_annotation.py",
        "--annotation-folder",
        annotation_path,
        "--no-video"
    ]
    if not args.save_video:
        commands.append("--no-video")
    command = " ".join(commands)
    os.system(command)
    
    # 5. tap-based temporal segmentation
    print("*************Tap Annotation*************")
    commands = [
        "python",
        "scripts/05_pt_changepoint_segmentation.py",
        "--annotation-folder",
        annotation_path,
        "--pen",
        str(args.tap_pen),
    ]
    command = " ".join(commands)
    os.system(command)

if __name__ == '__main__':
    main()