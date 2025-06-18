import pickle
import time

from avp_stream import VisionProStreamer


class AVPStreamer:
    def __init__(self, ip):
        self.streamer = VisionProStreamer(ip=ip, record=False)
        self.record = []

    def get(self):
        data = self.streamer.latest
        data["time"] = time.time()
        self.record.append(data)
        return self.streamer.latest

    def save_record(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.record, f)
