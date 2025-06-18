# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import argparse
import imageio.v3 as iio
import numpy as np

from tqdm import trange

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerOnlinePredictor

import init_path
from okami.plan_generation.utils.misc_utils import get_annotation_info
# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation-path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
    )

    args = parser.parse_args()

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    load_points = torch.load(f"{args.annotation_path}/points.pt")

    queries_y = torch.from_numpy(load_points[:, 1]).float().to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(-1)
    queries_x = torch.from_numpy(load_points[:, 0]).float().to(DEFAULT_DEVICE).unsqueeze(0).unsqueeze(-1)
    queries = torch.cat([torch.zeros_like(queries_x), queries_x, queries_y], dim=-1)


    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, queries=None):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )

    # Iterating over video frames, processing one window at a time:
    
    annotation_info = get_annotation_info(args.annotation_path)
    assert(annotation_info["mode"] in ["video", "demo", "human_demo"]), "target file must be a video"
    video_path = annotation_info["video_file"]
    print(annotation_info)


    video_frames = read_video_from_path(video_path)
    print(video_frames.shape)

    if annotation_info["mode"] != "human_demo":
        video_frames = video_frames[::5, ...]

    is_first_step = True
    for i in trange(len(video_frames)):
        frame = video_frames[i]
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries=queries,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
    )

    print("Tracks are computed")
    tracked_point_file = f"{args.annotation_path}/tracked_points.pt"
    pred_results = {
        "pred_tracks": pred_tracks,
        "pred_visibility": pred_visibility,
    }
    torch.save(pred_results, tracked_point_file)

    print("Data is ready. ")

    if args.save_video:
        print("Rendering video ...")
        # save a video with predicted tracks
        seq_name = "cotracker"
        video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
        vis = Visualizer(save_dir=f"./{args.annotation_path}/{seq_name}", 
                        pad_value=120, 
                        linewidth=3,
                        tracks_leave_trace=30)
        vis.visualize(video, 
                    pred_tracks, 
                    pred_visibility, 
                    query_frame=args.grid_query_frame)


    # pred_tracks: [B, T, N, 2]
    # pred_visibility: [B, T, N]


if __name__ == "__main__":
    main()