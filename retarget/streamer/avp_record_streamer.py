import pickle


class AVPRecordStreamer:
    def __init__(self, path, repeat=False):
        with open(path, "rb") as f:
            self.record = pickle.load(f)
        self.idx = 0
        self.repeat = repeat

    def get(self):
        if self.idx >= len(self.record):
            if self.repeat:
                self.reset()
            else:
                raise IndexError("End of record")
        data = self.record[self.idx]
        self.idx += 1
        return data

    def reset(self):
        self.idx = 0
