import time
from threading import Event, Thread

import cv2
import numpy as np
import pyrealsense2 as rs

from retarget.streamer.TeleVision.TeleVision import TeleVision


class TeleVisionStreamer:
    def __init__(self, resolution=(720, 1280), frame_rate=30):
        # Configuration for the RealSense camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, resolution[1], resolution[0], rs.format.bgr8, frame_rate
        )

        # Start the RealSense camera
        self.pipeline.start(self.config)

        # Initialize TeleVision app
        self.tv = TeleVision(resolution)

        # Thread control
        self.stop_event = Event()
        self.stream_thread = Thread(target=self.stream_loop)
        self.stream_thread.start()

    def modify_image_and_stream(self):
        # Wait for a coherent pair of frames: color frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        # Convert images to numpy arrays and process
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Modify and update the image on TeleVision
        self.tv.modify_shared_image(np.vstack((rgb_image, rgb_image)))

    def get(self):
        # Fetch the latest hand pose data from TeleVision
        return {
            "left_hand": self.tv.left_hand.copy(),
            "right_hand": self.tv.right_hand.copy(),
            "head": self.tv.head_matrix.copy(),
            "left_landmarks": self.tv.left_landmarks.copy(),
            "right_landmarks": self.tv.right_landmarks.copy(),
        }

    def stream_loop(self):
        while not self.stop_event.is_set():
            self.modify_image_and_stream()

    def stop(self):
        # Signal the thread to stop and wait for it to finish
        print("Closing!")
        self.stop_event.set()
        self.stream_thread.join()
        self.pipeline.stop()


if __name__ == "__main__":
    # Example usage:
    streamer = TeleVisionStreamer()
    while True:
        try:
            print(streamer.get())
            time.sleep(1)
        except KeyboardInterrupt:
            break
    streamer.stop()
