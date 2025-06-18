import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    args = parser.parse_args()
    
    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    print("*************Run HaMer*************")
    commands = [
        "python",
        "scripts/06a_hand_analysis.py",
        "--annotation-folder",
        annotation_path,
    ]
    command = " ".join(commands)
    os.system(command)

    print("*****Analyze Hand Object Contact*****")
    commands = [
        "python",
        "scripts/06b_hand_object_contact_calculation.py",
        "--annotation-folder",
        annotation_path,
    ]
    command = " ".join(commands)
    os.system(command)


if __name__ == '__main__':
    main()