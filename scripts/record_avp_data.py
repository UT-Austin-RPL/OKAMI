import argparse
import pickle
import time
from datetime import datetime

from avp_stream import VisionProStreamer


def main():
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Record data from AVP")
    parser.add_argument("--ip", type=str, help="IP of Apple Vision Pro")
    parser.add_argument("--num", type=int, default=100, help="Number of frames to record")
    parser.add_argument("--freq", type=int, default=20, help="frequency of recording")
    parser.add_argument(
        "--output",
        type=str,
        default=f"data/{cur_time}_avp_record",
        help="output file name",
    )
    args = parser.parse_args()

    s = VisionProStreamer(ip=args.ip, record=False)
    interval = 1.0 / args.freq
    data = []
    for _ in range(args.num):
        r = s.latest
        t = time.time()
        r["time"] = t
        data.append(r)
        time.sleep(interval)

    output_path = f"{args.output}_{args.freq}Hz_{args.num}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print("Done")
    exit(0)


if __name__ == "__main__":
    time.sleep(3)
    main()
