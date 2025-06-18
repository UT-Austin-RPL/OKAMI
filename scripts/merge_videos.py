import argparse

import cv2


def combine_videos(video_path1, video_path2, output_path, target_height):
    # Open the two video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Check if videos opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return

    # Get properties for scaling
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(cap1.get(cv2.CAP_PROP_FPS)), int(cap2.get(cv2.CAP_PROP_FPS)))

    # Calculate new widths while preserving aspect ratio
    new_width1 = int(width1 * (target_height / height1))
    new_width2 = int(width2 * (target_height / height2))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width1 + new_width2, target_height))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            # Resize frames to maintain aspect ratio
            resized_frame1 = cv2.resize(frame1, (new_width1, target_height))
            resized_frame2 = cv2.resize(frame2, (new_width2, target_height))

            # Concatenate images horizontally (left and right)
            combined_frame = cv2.hconcat([resized_frame1, resized_frame2])
            out.write(combined_frame)
        else:
            break

    # Release everything when done
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete and the output file is saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two videos side by side")
    parser.add_argument("--video1", type=str, help="Path to the first video file")
    parser.add_argument("--video2", type=str, help="Path to the second video file")
    parser.add_argument("--output", type=str, help="Path to save the combined video file")
    parser.add_argument("--height", type=int, help="Output video height in pixels", default=480)
    args = parser.parse_args()

    combine_videos(args.video1, args.video2, args.output, args.height)
