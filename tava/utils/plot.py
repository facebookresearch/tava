# Copyright (c) Meta Platforms, Inc. and affiliates.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from tava.utils.bone import get_end_points
from tava.utils.camera import project_points_to_image_plane
from tava.utils.structures import Bones, Cameras


def _frontback(T):
    """
    Sort front and back facing triangles
    Parameters:
    -----------
    T : (n,3) array
       Triangles to sort
    Returns:
    --------
    front and back facing triangles as (n1,3) and (n2,3) arrays (n1+n2=n)
    """
    Z = (
        (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 1, 1] + T[:, 0, 1])
        + (T[:, 2, 0] - T[:, 1, 0]) * (T[:, 2, 1] + T[:, 1, 1])
        + (T[:, 0, 0] - T[:, 2, 0]) * (T[:, 0, 1] + T[:, 2, 1])
    )
    return Z >= 0, Z < 0


class PyPlotVisualizer:
    """Visualizer based on matplotlib.pyplot

    Useful to rasterize a 3D scene into 2D images.
    """

    def __init__(self, cameras: Cameras, dpi=50):
        assert cameras.intrins.dim() == 2, "Only support a single camera"

        self.cameras = cameras
        self.fig = plt.figure(
            figsize=(cameras.width / dpi, cameras.height / dpi), dpi=dpi
        )
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes(
            [0, 0, 1, 1],
            xlim=[0, self.cameras.width],
            ylim=[self.cameras.height, 0],
            aspect=1,
        )
        self.ax.axis("off")

    def plot_bones(self, bones: Bones, color="red"):
        heads, tails = get_end_points(bones)
        heads2d, _ = project_points_to_image_plane(self.cameras, heads)
        tails2d, _ = project_points_to_image_plane(self.cameras, tails)
        heads2d = heads2d.view(-1, 2).cpu().numpy()
        tails2d = tails2d.view(-1, 2).cpu().numpy()
        # plot the bones
        for (x1, y1), (x2, y2) in zip(heads2d, tails2d):
            plt.plot(
                np.stack([x1, x2]),
                np.stack([y1, y2]),
                marker="o",
                color=color,
                linewidth=0.006 * self.cameras.width,
                markersize=0.01 * self.cameras.height,
            )

    def plot_mesh(
        self,
        verts,
        faces,
        facecolors="white",
        edgecolors="black",
        linewidths=0.5,
        mode="front",
    ):
        assert mode in ["front", "back"]

        verts2d, depth = project_points_to_image_plane(self.cameras, verts)
        verts2d, depth = verts2d.cpu().numpy(), depth.cpu().numpy()

        T = verts2d[faces]
        Z = -depth[faces].mean(axis=1)

        # Back face culling
        if mode == "front":
            front, back = _frontback(T)
            T, Z = T[front], Z[front]
            if len(facecolors) == len(faces):
                facecolors = facecolors[front]
            if len(edgecolors) == len(faces):
                edgecolors = edgecolors[front]

        # Front face culling
        elif mode == "back":
            front, back = _frontback(T)
            T, Z = T[back], Z[back]
            if len(facecolors) == len(faces):
                facecolors = facecolors[back]
            if len(edgecolors) == len(faces):
                edgecolors = edgecolors[back]

        # Separate 2d triangles from zbuffer
        triangles = T[:, :, :2]
        antialiased = linewidths > 0

        # Sort triangles according to z buffer
        I = np.argsort(Z)
        triangles = triangles[I, :]
        if len(facecolors) == len(I):
            facecolors = facecolors[I, :]
        if len(edgecolors) == len(I):
            edgecolors = edgecolors[I, :]

        collection = PolyCollection([], closed=True)
        collection.set_verts(triangles)
        collection.set_linewidths(linewidths)
        collection.set_facecolors(facecolors)
        collection.set_edgecolors(edgecolors)
        collection.set_antialiased(antialiased)
        self.ax.add_collection(collection)

    def draw(self, show=True):
        # draw & save to image.
        self.canvas.draw()
        s, _ = self.canvas.print_to_buffer()
        image = np.frombuffer(s, np.uint8).reshape(
            (self.cameras.height, self.cameras.width, 4)
        )
        if show:
            plt.show()
        plt.close(self.fig)
        return image
