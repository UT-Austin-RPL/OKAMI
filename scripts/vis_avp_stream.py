import argparse
import datetime
import time

from retarget.retargeter import TeleopAVPGR1Retargeter
from retarget.streamer import AVPStreamer
from retarget.utils.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Record data from AVP")
    parser.add_argument("--ip", type=str, help="AVP IP address")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the data")
    parser.add_argument("--save", action="store_true", help="Save the data")
    parser.add_argument(
        "--config", type=str, help="config path", default="configs/teleop_avp_gr1.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.save or args.save_path is not None:
        if args.save_path is None:
            args.save_path = f"data/avp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        print(f"Will save data to {args.save_path}")
        retargeter = TeleopAVPGR1Retargeter(config, vis=True, save_path=args.save_path)
    else:
        retargeter = TeleopAVPGR1Retargeter(config, vis=True)
    sleep_time = 0.05

    s = AVPStreamer(args.ip)
    data0 = s.get()
    retargeter.calibrate(data0)
    cnt = 0
    print("Calibrated, start streaming...")
    try:
        while True:
            cnt += 1
            start = time.time()
            data_t = s.get()
            q, error_dict = retargeter(data_t)
            if cnt % 20 == 0:
                print(f"Error: {error_dict}")
            # break
            time_to_sleep = sleep_time - (time.time() - start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
    except KeyboardInterrupt:
        retargeter.save()
        print("Saved data. Can exit...")


if __name__ == "__main__":
    time.sleep(3)
    main()
