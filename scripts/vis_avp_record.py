import argparse
import time

from retarget.retargeter import TeleopAVPGR1Retargeter
from retarget.streamer.avp_record_streamer import AVPRecordStreamer
from retarget.utils.configs import load_config


def main():
    parser = argparse.ArgumentParser(description="Record data from AVP")
    parser.add_argument("--input", type=str, help="data to streaming results")
    args = parser.parse_args()
    config = load_config("configs/teleop_avp_gr1.yaml")
    retargeter = TeleopAVPGR1Retargeter(config, vis=True)

    s = AVPRecordStreamer(path=args.input)
    data0 = s.get()
    retargeter.calibrate(data0)
    print("Calibrated, start streaming...")

    while True:
        try:
            data_t = s.get()
        except IndexError:
            s.reset()
            print("reset")
            data_t = s.get()
        q, error_dict = retargeter(data_t)
        # break
        time.sleep(0.05)


if __name__ == "__main__":
    main()
