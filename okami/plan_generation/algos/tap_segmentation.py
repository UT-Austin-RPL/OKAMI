import argparse
import numpy as np
import os
import ruptures as rpt
import matplotlib.pyplot as plt

from easydict import EasyDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from okami.plan_generation.utils.misc_utils import *
from okami.plan_generation.algos.temporal_segments import TemporalSegments

DEFAULT_CONFIG = EasyDict({
    "clustering": {
        "max_K": 10,
    },
    "temporal_segmentation": {
        "min_size": 2,
        "kernel": "rbf",
        "pen": 10,
    }
})

class TAPSegmentation:
    def __init__(self, cfg=DEFAULT_CONFIG):
        self.clusters = []
        self.temporal_segments = TemporalSegments()
        self.cfg = cfg
        self.optimal_K = None

    def set_pen(self, pen):
        self.cfg.temporal_segmentation.pen = pen

    def segmentation(self, annotation_path):
        # Load data from annotation path
        tap_results = get_tracked_points_annotation(annotation_path)
        config_dict = get_annotation_info(annotation_path)

        # Process data
        pixel_trajs = tap_results["pred_tracks"].permute(0, 2, 1, 3)[0].detach().cpu().numpy()
        visiblity_trajs = tap_results["pred_visibility"].permute(0, 2, 1)[0].detach().cpu().numpy()
        
        # Compute trajectory statistics
        p = pixel_trajs[:, 1:, :] - pixel_trajs[:, 0:-1, :]
        N, T, _ = p.shape
        p_reshaped = p.reshape(N, T*2)
        p_standardized = StandardScaler().fit_transform(p_reshaped)

        # Find optimal K using silhouette value
        silhouette_scores = []
        K = range(2, self.cfg.clustering.max_K)  # Start from 2 as silhouette score is not defined for k=1
        for k in K:
            km = KMeans(n_clusters=k)
            clusters = km.fit_predict(p_standardized)
            silhouette_scores.append(silhouette_score(p_standardized, clusters))
        
        optimal_k = K[silhouette_scores.index(max(silhouette_scores))]

        # Apply K-means clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k)
        clusters = kmeans.fit_predict(p_standardized)

        # Changepoint detection
        v = np.linalg.norm(p, axis=2)
        signal = [v[clusters == cluster_idx].mean(axis=0) for cluster_idx in range(optimal_k)]
        if len(signal[0].shape) == 1:
            signal = [np.expand_dims(s, axis=-1) for s in signal]
        signal = np.concatenate(signal, axis=-1)
        
        algo_c = rpt.KernelCPD(kernel=self.cfg.temporal_segmentation.kernel, 
                               min_size=self.cfg.temporal_segmentation.min_size).fit(signal)
        result = algo_c.predict(pen=self.cfg.temporal_segmentation.pen)
        print(optimal_k)
        print(clusters.reshape(-1, 40))
        print(result)

        self.clusters = clusters
        self.optimal_K = optimal_k

        result = [0] + result
        if result[-1] != T:
            result.append(T)

        for i in range(len(result) - 1):
            self.temporal_segments.add_segment(result[i], result[i + 1])
        
    def generate_segment_videos(self, annotation_path):
        # Load data from annotation path
        video_seq = get_video_seq_from_annotation(annotation_path)

        initial_images = []
        for segment in self.temporal_segments.segments:
            initial_images.append(video_seq[segment.start_idx])
        video_writer = VideoWriter(annotation_path, video_name=f"tap_segmentation.mp4", fps=30)
        for i in range(len(video_seq)):
            image_frame = []
            for segment in self.temporal_segments.segments:
                if segment.in_duration(i):
                    image_frame.append(video_seq[i])
                elif i >= segment.end_idx:
                    image_frame.append(video_seq[segment.end_idx])
                else:
                    image_frame.append(video_seq[segment.start_idx])
            image_frame = np.concatenate(image_frame, axis=1)
            video_writer.append_image(image_frame)
        video_writer.save(bgr=False)

        # self.save_segments(os.path.join(annotation_path, "temporal_segments.pt"))
        # self.save_clustering(os.path.join(annotation_path, "clustering.pt"))

    def save_clustering(self, file_path):
        torch.save(self.clusters, file_path)

    def load_clustering(self, file_path):
        self.clusters = torch.load(file_path) # weights_only=False

    def save_segments(self, file_path):
        self.temporal_segments.save_to_file(file_path)

    def load_segments(self, file_path):
        self.temporal_segments.load_from_file(file_path)

    def save(self, annotation_path):
        self.save_segments(os.path.join(annotation_path, "tap_temporal_segments.pt"))
        self.save_clustering(os.path.join(annotation_path, "clustering.pt"))

    def load(self, annotation_path):
        self.load_segments(os.path.join(annotation_path, "tap_temporal_segments.pt"))
        self.load_clustering(os.path.join(annotation_path, "clustering.pt"))