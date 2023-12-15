import numpy as np
from scipy.ndimage import distance_transform_edt
import plotly.graph_objects as go
import sys
sys.path.append("build")
import morse_complex as mc

def distance_transform(image):
    distance_f = np.copy(image)
    distance_f = distance_transform_edt(distance_f)
    distance_b = np.copy(1-image)
    distance_b = distance_transform_edt(distance_b)
    signed_distance = -distance_f + distance_b

    return signed_distance

def minimum_distance_in(image):
    min_distances = []
    values = np.sort(np.unique(image))
    min_distance = np.min(np.abs(values[1:]-values[:-1]))

    return min_distance

def plot_morse_skeleton(MC, image=np.zeros((0)), threshold=0, plot_below=True, plot_above=False, plot_critical=True):
    if image.shape != (0):
        skeleton = np.copy(image)
    
    if plot_below:
        skeleton_pixels_below = MC.get_morse_skeleton_below()
        for pixel in skeleton_pixels_below:
            skeleton[pixel[0], pixel[1], pixel[2]] = 100000

    if plot_above:
        skeleton_pixels_above = MC.get_morse_skeleton_above()
        for pixel in skeleton_pixels_above:
            skeleton[pixel[0], pixel[1], pixel[2]] = 100002
    
    if plot_critical:
        critical_cubes = MC.get_critical_cells()
        critical_voxels_below = []
        critical_voxels_above = []
        for dim in range(4):
            for c in critical_cubes[dim]:
                if c.birth < threshold:
                    critical_voxels_below.append(c.get_voxels())
                else:
                    critical_voxels_above.append(c.get_voxels())
        if plot_below:
            for c in critical_voxels_below:
                for pixel in c:
                    skeleton[pixel[0], pixel[1], pixel[2]] = 100001
        if plot_above:
            for c in critical_voxels_above:
                for pixel in c:
                    skeleton[pixel[0], pixel[1], pixel[2]] = 100003


    x_0, y_0, z_0 = np.where(skeleton < threshold)
    x_2, y_2, z_2 = np.where(skeleton == 100000)
    x_3, y_3, z_3 = np.where(skeleton == 100001)
    x_4, y_4, z_4 = np.where(skeleton == 100002)
    x_5, y_5, z_5 = np.where(skeleton == 100003)

    fig = go.Figure()
    if image.shape != (0):
        fig.add_trace(go.Scatter3d(x=x_0, y=y_0, z=z_0, opacity=0.2, mode='markers', marker=dict(size=2, color='blue'), name='voxels below'))
    if plot_below:
        fig.add_trace(go.Scatter3d(x=x_2, y=y_2, z=z_2, mode='markers', marker=dict(size=2, color='orange'), name='skeleton below'))
        if plot_critical:
            fig.add_trace(go.Scatter3d(x=x_3, y=y_3, z=z_3, mode='markers', marker=dict(size=2, color='red'), name='critical below'))
    if plot_above:
        fig.add_trace(go.Scatter3d(x=x_4, y=y_4, z=z_4, mode='markers', marker=dict(size=2, color='green'), name='skeleton above'))
        if plot_critical:
            fig.add_trace(go.Scatter3d(x=x_5, y=y_5, z=z_5, mode='markers', marker=dict(size=2, color='yellow'), name='critical above'))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='manual'))
    fig.show()