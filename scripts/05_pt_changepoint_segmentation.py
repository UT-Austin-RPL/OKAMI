import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

from easydict import EasyDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import init_path
from okami.plan_generation.utils.misc_utils import *
from okami.plan_generation.algos.tap_segmentation import TAPSegmentation


def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.', default='annotations/human_demo/salt_demo')
    parser.add_argument('--pen', type=float, default=10, help='Penalty for changepoint detection.')
    args = parser.parse_args()

    tap_segmentation = TAPSegmentation()
    tap_segmentation.set_pen(args.pen)

    with open(os.path.join(args.annotation_folder, "pt_segmentation_info.json"), "w") as f:
        json.dump(tap_segmentation.cfg, f)

    tap_segmentation.segmentation(args.annotation_folder)
    tap_segmentation.save(args.annotation_folder)
    tap_segmentation.generate_segment_videos(args.annotation_folder)

if __name__ == "__main__":
    main()