import torch
import numpy as np

class Segment():
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def numpy(self):
        return np.array([self.start_idx, self.end_idx])
    
    def to_torch_tensor(self):
        return torch.tensor(self.numpy())
    
    @property
    def seg_len(self):
        return self.end_idx - self.start_idx + 1
    
    def in_duration(self, idx):
        return idx >= self.start_idx and idx <= self.end_idx
    
    def __repr__(self) -> str:
        repr_str = f"Segment(start_idx={self.start_idx}, end_idx={self.end_idx})"
        return repr_str

class TemporalSegments():
    def __init__(self, name=""):
        self.name = name
        self.segments = []
        self.index = 0

    def add_segment(self, start_idx, end_idx):
        self.segments.append(Segment(start_idx, end_idx))

    def check_data_integrity(self):
        assert(self.segments[0].start_idx == 0), "The first segment should start at 0"
        for i in range(1, len(self.segments)):
            assert(self.segments[i].start_idx == self.segments[i-1].end_idx), "The segments should be contiguous"
        for i in range(len(self.segments)):
            assert(self.segments[i].start_idx < self.segments[i].end_idx), "The segments should have positive length"

    @property
    def num_segments(self):
        return len(self.segments)
    
    @property
    def seq_len(self):
        return self.segments[-1].end_idx - self.segments[0].start_idx + 1
    
    def numpy(self):
        return np.stack([segment.numpy() for segment in self.segments])
    
    def to_torch_tensor(self):
        return torch.tensor(self.numpy())
    
    def __getitem__(self, idx):
        assert(idx < len(self.segments)), "Index out of range"
        return self.segments[idx]
    
    def __iter__(self):
        # Return the iterator object (in this case, the instance itself)
        return self

    def __next__(self):
        if self.index < len(self.segments):
            result = self.segments[self.index]
            self.index += 1
            return result
        else:
            # Reset the index for next iteration and stop iteration
            self.index = 0
            raise StopIteration
        
    def __repr__(self) -> str:
        repr_str = f"TemporalSegments(name={self.name}, num_segments={self.num_segments}, seq_len={self.seq_len})"
        return repr_str
        
    def save_to_file(self, file_path):
        torch.save(self.to_torch_tensor(), file_path)

    def load(self, data):
        self.segments = []
        self.segments = [Segment(*seg) for seg in data]
        self.check_data_integrity()

    def load_from_file(self, file_path):
        load_data = torch.load(file_path)
        self.load(load_data)