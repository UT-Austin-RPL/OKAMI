import argparse
import datetime
import time

from retarget.retargeter import TeleopAVPGR1HandRetargeter
from retarget.streamer import TeleVisionStreamer
from retarget.utils.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Record data from AVP")
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[720, 1280], help="Video streaming resolution"
    )
    parser.add_argument(
        "--config", type=str, help="config path", default="configs/teleop_tv_gr1_hand.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    retargeter = TeleopAVPGR1HandRetargeter(config, side="left", vis=True)

    s = TeleVisionStreamer(args.resolution)
    sleep_time = 0.05
    _ = input("Ready?")

    data0 = s.get()
    retargeter.calibrate(data0)
    cnt = 0
    print("Calibrated, start streaming...")
    try:
        while True:
            cnt += 1
            start = time.time()
            data_t = s.get()
            _ = retargeter(data_t)
            time_to_sleep = sleep_time - (time.time() - start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
    except KeyboardInterrupt:
        retargeter.save()
        print("Saved data. Can exit...")


if __name__ == "__main__":
    time.sleep(3)
    main()
