import plotly.express as px
import plotly.graph_objects as go

def plotly_draw_image(image, 
                      width=300,
                      height=300,
                      offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )
    if not offline:
        fig.show()
    else:
        return fig
    
def plotly_draw_seg_image(image, 
                          mask,
                          width=300,
                          height=300,
                          offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        mask (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)

    fig.data[0].customdata = mask
    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}<br>Mask ID: %{customdata}'

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )

    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_3d_pcd(pcd_points, pcd_colors=None, addition_points=None, marker_size=3, equal_axis=True, title="", offline=False, no_background=False, default_rgb_str="(255,0,0)",additional_point_draw_lines=False, uniform_color=False):

    if pcd_colors is None:
        color_str = [f'rgb{default_rgb_str}' for _ in range(pcd_points.shape[0])]
    else:
        color_str = ['rgb('+str(r)+','+str(g)+','+str(b)+')' for r,g,b in pcd_colors]

    # Extract x, y, and z columns from the point cloud
    x_vals = pcd_points[:, 0]
    y_vals = pcd_points[:, 1]
    z_vals = pcd_points[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(size=3, color=color_str, opacity=0.8)
    )
    data = [rgbd_scatter]
    if addition_points is not None:
        assert(addition_points.shape[-1] == 3)
        # check if addition_points are three dimensional
        if len(addition_points.shape) == 2:
            addition_points = [addition_points]
        for points in addition_points:
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            if additional_point_draw_lines:
                mode = "lines+markers"
            else:
                mode = "markers"
            marker_dict = dict(size=marker_size,
                                opacity=0.8)
            
            if uniform_color:
                marker_dict["color"] = f'rgb{default_rgb_str}'
            rgbd_scatter2 = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode=mode,
                marker=marker_dict,
                )
            data.append(rgbd_scatter2)

    if equal_axis:
        scene_dict = dict(   
            aspectmode='data',  
        )
    else:
        scene_dict = dict()
    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        # axes range
        scene=scene_dict,
        title=dict(text=title, automargin=True)
    )

    fig = go.Figure(data=data, layout=layout)

    if no_background:
        fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            yaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            zaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(l=0, r=0, b=0, t=0),  # No margins
        showlegend=False,
    )


    if not offline:
        fig.show()
    else:
        return fig