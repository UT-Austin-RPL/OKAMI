import argparse
import pickle
import time

import numpy as np

from retarget.retargeter import TeleopAVPGR1HandRetargeter
from retarget.streamer.avp_record_streamer import AVPRecordStreamer
from retarget.utils.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Record data from SMPL")
    parser.add_argument("--input", type=str, help="data to streaming results")
    args = parser.parse_args()
    config = load_config("configs/smpl_gr1_hand.yaml")
    retargeter = TeleopAVPGR1HandRetargeter(config, side="right", vis=True)

    with open(args.input, "rb") as f:
        s = pickle.load(f)
    data0 = s[0]
    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")

    while True:
        for data_t in s:
            # print(data_t['left'])
            q = retargeter(data_t)
            # print(q)
            time.sleep(0.05)


if __name__ == "__main__":
    main()
