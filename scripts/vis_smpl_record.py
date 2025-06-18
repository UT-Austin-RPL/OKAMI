import argparse
import pickle
import time

import cv2
import numpy as np

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
from retarget.utils.constants import name_to_urdf_idx


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded data from smpl")
    parser.add_argument("--input", type=str, help="data to streaming results")
    parser.add_argument(
        "--config", type=str, help="data to streaming results", default="configs/smpl_gr1.yaml"
    )
    # output video file, optional
    parser.add_argument("--output", type=str, help="output video file", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    # s = np.load(args.input).astype(np.float64)  # T 52 4 4

    retargeter = SMPLGR1Retargeter(config, vis=True)

    with open(args.input, "rb") as f:
        s = pickle.load(f)
    # s = np.load(args.input)
    data0 = s[0]
    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")
    if args.output is not None:
        # 1080p
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 20, (1920, 1080))
        time.sleep(5)

    while True:
        result = []
        for data_t in s:
            q = retargeter(data_t)
            result.append(q)
            # break
            if args.output is not None:
                rgba = retargeter.vis.viz.captureImage()[:, :, :3]
                # make the size 1080p, also drop the a channel
                rgba = cv2.resize(rgba, (1920, 1080))
                rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                writer.write(rgba)
            time.sleep(0.05)
        np.save(args.input[:-4] + "_retargeted.npy", np.array(result))
        if args.output is not None:
            writer.release()
            print(f"Saved video to {args.output}")
            break


if __name__ == "__main__":
    main()
