import argparse
import time

from retarget.retargeter import TeleopAVPGR1HandRetargeter
from retarget.streamer import AVPStreamer
from retarget.utils.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Record data from AVP")
    parser.add_argument("--ip", type=str, help="AVP IP address")
    args = parser.parse_args()
    config = load_config("configs/teleop_avp_gr1.yaml")
    retargeter = TeleopAVPGR1HandRetargeter(config, side="left", vis=True)
    sleep_time = 0.05

    s = AVPStreamer(args.ip)
    data0 = s.get()
    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")
    while True:
        start = time.time()
        data_t = s.get()
        q = retargeter(data_t)
        # break
        time_to_sleep = sleep_time - (time.time() - start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


if __name__ == "__main__":
    main()
