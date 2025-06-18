import io
from typing import Any
import imageio

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from PIL import Image

class TAP3dTraj:
    def __init__(self, x, y, z, name, visibility=None):
        self.x = x
        self.y = y
        self.z = z
        self.name = name
        if visibility is None:
            self.visibility = [True] * len(x)
        else:
            self.visibility = visibility

    def get_traj(self):
        return self.x, self.y, self.z
    
    def get_visibility(self):
        return self.visibility
    
    def __len__(self):
        return len(self.x)
    
def draw_3d_traj(traj_list):
    fig = go.Figure()
    for traj in traj_list:
        x, y, z = traj.get_traj()
        collected_points = []
        visibility = traj.get_visibility()
        for i in range(len(x)):
            if not visibility[i]:
                collected_points.append([x[i], y[i], z[i]])
        fig = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))

        if len(collected_points) > 0:
            collected_points = np.array(collected_points)
            fig = fig.add_trace(go.Scatter3d(x=collected_points[:, 0], y=collected_points[:, 1], z=collected_points[:, 2], 
                                             mode='markers', marker=dict(size=2, color='red')))

    fig.update_layout(
        title="3D Trajectory",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        showlegend=False
    )

    # Show the plot
    fig.show()

def draw_3d_traj_video(traj_list,
                       video_name='3d_trajectory_video.mp4'):
    # Prepare video writer using imageio
    writer = imageio.get_writer(video_name, fps=20)

    # Loop over each frame in the trajectory
    for frame_idx in range(len(traj_list[0])):
        fig = go.Figure()
        # Plot the trajectory up to the current frame
        for traj in traj_list:
            x, y, z = traj.get_traj()
            visibility = traj.get_visibility()
            collected_points = []

            # Process each point in the trajectory
            for i in range(len(x)):
                if not visibility[i]:
                    collected_points.append([x[i], y[i], z[i]])

            # Select the segment of the trajectory to display
            end_idx = min(frame_idx, len(x))
            start_idx = max(0, end_idx - 10)
            fig.add_trace(go.Scatter3d(x=x[start_idx:end_idx], y=y[start_idx:end_idx], z=z[start_idx:end_idx], 
                                       mode='lines+markers',
                                       marker=dict(size=4, color='blue'),
                                       line=dict(width=2)))

            # Add invisible points
            if len(collected_points) > 0:
                collected_points = np.array(collected_points)
                fig.add_trace(go.Scatter3d(x=collected_points[:, 0], y=collected_points[:, 1], z=collected_points[:, 2], 
                                           mode='markers', marker=dict(size=2, color='red')))

        # Update layout
        fig.update_layout(
            title="3D Trajectory",
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                xaxis=dict(range=[0.5, 0.7]),
                yaxis=dict(range=[-0.2, 0.2]),
                zaxis=dict(range=[-0.1, 0.4])
            ),
            showlegend=False
        )

        # Convert Plotly figure to an image
        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)

        # Write frame to video
        writer.append_data(img)

        # Close the current plot
        # plt.close()

    # Close the writer
    writer.close()
    print(f"Video saved to {video_name}")


def convert_to_tap3dtraj(segments):
    tap3d_segments = []
    for i in range(len(segments)):
        segment = segments[i]
        tap3d_segment = TAP3dTraj(segment[:, 0], segment[:, 1], segment[:, 2], f'traj-{i}')
        tap3d_segments.append(tap3d_segment)
    return tap3d_segments