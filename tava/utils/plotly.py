#!/usr/bin/env python3
#
# File   : scene.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 04/15/2021
#
# Distributed under terms of the MIT license.
# 
# Example Usuage:
# 
# 

import dataclasses
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# install by "pip install plotly"
import plotly.graph_objects as go


class PerspectiveCamera(NamedTuple):
    # All properties are in (W, H) order.
    focals: Tuple[float, float]
    image_size: Tuple[int, int]
    # Assume OpenGL camera space.
    c2w: np.ndarray  # (4, 4)
    principal_points: Optional[Tuple[float, float]] = None
    colors: Optional[Union[List, np.ndarray]] = None  # (3,)


class AxisArgs(NamedTuple):
    showgrid: bool = True
    zeroline: bool = True
    showline: bool = True
    ticks: str = "outside"
    showticklabels: bool = True
    backgroundcolor: str = "rgb(230, 230, 250)"
    showaxeslabels: bool = False


class Lighting(NamedTuple):
    ambient: float = 0.8
    diffuse: float = 1.0
    fresnel: float = 0.0
    specular: float = 0.0
    roughness: float = 0.5
    facenormalsepsilon: float = 1e-6
    vertexnormalsepsilon: float = 1e-12


@dataclasses.dataclass
class PointCloud(object):
    points: np.ndarray  # (N, 3)
    colors: Optional[Union[List, np.ndarray]] = None  # (3,) or (N, 3)


@dataclasses.dataclass
class Segment(PointCloud):
    parents: Optional[np.ndarray] = None  # (J,)
    end_points: Optional[np.ndarray] = None  # (J, 3)


@dataclasses.dataclass
class Trimesh(object):
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (F, 3)
    vertex_colors: Optional[Union[List, np.ndarray]] = None  # (3,) or (N, 3)


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * np.array([-2, 1.5, -4])
    up1 = 0.5 * np.array([0, 1.5, -4])
    up2 = 0.5 * np.array([0, 2, -4])
    b = 0.5 * np.array([2, 1.5, -4])
    c = 0.5 * np.array([-2, -1.5, -4])
    d = 0.5 * np.array([2, -1.5, -4])
    C = np.zeros(3)
    F = np.array([0, 0, -3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x for x in camera_points]) * scale
    return lines


def get_plane_pts(
    image_size, focals, principal_points, camera_scale=0.3, scale_factor=1 / 4
):
    w, h = image_size
    fx, fy = focals
    if principal_points is None:
        principal_points = w / 2, h / 2
    cx, cy = principal_points
    Z = -2 * camera_scale
    oW, oH = w / fx * abs(Z), h / fy * abs(Z)

    X0, Y0, X1, Y1 = (
        -cx / w * oW,
        cy / h * oH,
        oW - cx / w * oW,
        -(oH - cy / h * oH),
    )
    wsteps, hsteps = int(w * scale_factor), int(h * scale_factor)
    Ys, Xs = np.meshgrid(
        np.linspace(Y0, Y1, num=hsteps),
        np.linspace(X0, X1, num=wsteps),
        indexing="ij",
    )
    Zs = np.ones_like(Xs) * Z
    plane_pts = np.stack([Xs, Ys, Zs], axis=-1)
    return plane_pts


def get_plane_pts_from_camera(camera, camera_scale=1, scale_factor=1 / 4):
    focals = camera.focals
    principal_points = camera.principal_points
    image_size = camera.image_size
    if principal_points is None:
        principal_points = (image_size[0] / 2, image_size[1] / 2)
    Z = -(focals[0] + focals[1]) / 2 * camera_scale
    X0, Y0, X1, Y1 = (
        -principal_points[0] * camera_scale,
        principal_points[1] * camera_scale,
        (image_size[0] - principal_points[0]) * camera_scale,
        -(image_size[1] - principal_points[1]) * camera_scale,
    )

    # scale image to plane such that it can go outside of the x0x1 range.
    ratio = min(image_size[0] / (X1 - X0), image_size[1] / (Y0 - Y1))
    X0, Y0, X1, Y1 = [s / ratio for s in [X0, Y0, X1, Y1]]
    wsteps, hsteps = (
        int(image_size[0] * scale_factor),
        int(image_size[1] * scale_factor),
    )
    Ys, Xs = np.meshgrid(
        np.linspace(Y0, Y1, num=hsteps),
        np.linspace(X0, X1, num=wsteps),
        indexing="ij",
    )
    Zs = np.ones_like(Xs) * Z
    plane_pts = np.stack([Xs, Ys, Zs], axis=-1)
    return plane_pts


def c2w_to_eye_at_up(c2w):
    # in opengl format.
    eye_at_up_c = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    eye_at_up_w = (c2w[None, :3, :3] @ eye_at_up_c[..., None])[..., 0] + c2w[
        None, :3, -1
    ]
    eye, at, up_plus_eye = (eye_at_up_w[0], eye_at_up_w[1], eye_at_up_w[2])
    up = up_plus_eye - eye
    return eye, at, up


SceneStruct = Union[
    PerspectiveCamera,
    PointCloud,
    Segment,
    Trimesh,
]


def plot_scene(
    plots: Union[SceneStruct, Dict[str, Dict[str, Union[SceneStruct, Any]]]],
    *,
    axis_args: AxisArgs = AxisArgs(),
    lighting: Lighting = Lighting(),
    mesh_opacity: float = 1,
    camera_scale: float = 0.3,
    show_camera_wireframe: bool = True,
    image_opacity: float = 1,
    image_scale_size: float = 1 / 4,
    marker_size: int = 2,
    segment_size: int = 1,
    point_opacity: float = 1,
    show_point: bool = True,
    segment_opacity: float = 1,
    show_segment: float = True,
    use_cone_segment: bool = False,
    up: str = "+z",
    eye: Sequence[float] = (1.25, -1, 1.25),
    height: Optional[int] = None,
    width: Optional[int] = None,
    uirevision: bool = True,  # Make camera persistent across updates.
    viewpoint_c2w: Optional[np.ndarray] = None,
    **kwargs,
):
    fig = go.Figure()
    axis_args = axis_args._asdict()
    lighting = kwargs.get("lighting", Lighting())._asdict()

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args}
    y_settings = {**axis_args}
    z_settings = {**axis_args}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get("xaxis", {}))
    y_settings.update(**kwargs.get("yaxis", {}))
    z_settings.update(**kwargs.get("zaxis", {}))

    # in opengl format.
    viewpoint_camera_dict = {
        "up": {
            k: (1 if up.startswith("+") else -1) * (1 if up.endswith(k) else 0)
            for k in "xyz"
        },
        "eye": {k: v for k, v in zip("xyz", eye)},
    }
    viewpoint_eye_at_up = None
    if viewpoint_c2w is not None:
        viewpoint_eye_at_up = c2w_to_eye_at_up(viewpoint_c2w)

    for trace_name, struct_dict in plots.items():
        if not isinstance(struct_dict, dict):
            struct_dict = {"struct": struct_dict}
        struct = struct_dict["struct"]
        if isinstance(struct, (Trimesh)):
            struct_dict.setdefault("lighting", lighting)
            struct_dict.setdefault("mesh_opacity", mesh_opacity)
            _add_mesh_trace(fig, **struct_dict, trace_name=trace_name)
        elif isinstance(struct, PerspectiveCamera):
            struct_dict.setdefault("camera_scale", camera_scale)
            struct_dict.setdefault(
                "show_camera_wireframe", show_camera_wireframe
            )
            struct_dict.setdefault("image_scale_size", image_scale_size)
            struct_dict.setdefault("image_opacity", image_opacity)
            struct_dict.setdefault("marker_size", marker_size)
            _add_camera_trace(fig, **struct_dict, trace_name=trace_name)
        elif isinstance(struct, (PointCloud, Segment)) or (
            isinstance(struct, List)
            and isinstance(struct[0], (PointCloud, Segment))
        ):
            struct_dict.setdefault("marker_size", marker_size)
            struct_dict.setdefault("segment_size", segment_size)
            struct_dict.setdefault("point_opacity", point_opacity)
            struct_dict.setdefault("segment_opacity", segment_opacity)
            struct_dict.setdefault("show_point", show_point)
            struct_dict.setdefault("show_segment", show_segment)
            struct_dict.setdefault("use_cone_segment", use_cone_segment)
            _add_pointcloud_trace(fig, **struct_dict, trace_name=trace_name)
        else:
            raise ValueError(
                (
                    "struct {} is not a PerspectiveCamera, Pointcloud, "
                    "Segment, Trimesh, trimesh.Trimesh, or "
                    "trimesh.PointCloud."
                ).format(struct)
            )

    layout = fig["layout"]["scene"]
    xaxis = layout["xaxis"]
    yaxis = layout["yaxis"]
    zaxis = layout["zaxis"]

    # Update the axes with our above default and provided settings.
    xaxis.update(**x_settings)
    yaxis.update(**y_settings)
    zaxis.update(**z_settings)

    # cubify the view space.
    x_range = xaxis["range"]
    y_range = yaxis["range"]
    z_range = zaxis["range"]
    ranges = np.array([x_range, y_range, z_range])
    center = ranges.mean(1)
    max_len = (ranges[:, 1] - ranges[:, 0]).max() * 1.1
    ranges = np.stack(
        [center - max_len / 2, center + max_len / 2], axis=0
    ).T.tolist()
    xaxis["range"] = ranges[0]
    yaxis["range"] = ranges[1]
    zaxis["range"] = ranges[2]

    # update camera viewpoint if provided
    if viewpoint_eye_at_up:
        eye, at, up = viewpoint_eye_at_up
        eye_x, eye_y, eye_z = eye.tolist()
        at_x, at_y, at_z = at.tolist()
        up_x, up_y, up_z = up.tolist()

        # scale camera eye to plotly [-1, 1] ranges
        eye_x = _scale_camera_to_bounds(eye_x, x_range, True)
        eye_y = _scale_camera_to_bounds(eye_y, y_range, True)
        eye_z = _scale_camera_to_bounds(eye_z, z_range, True)

        at_x = _scale_camera_to_bounds(at_x, x_range, True)
        at_y = _scale_camera_to_bounds(at_y, y_range, True)
        at_z = _scale_camera_to_bounds(at_z, z_range, True)

        up_x = _scale_camera_to_bounds(up_x, x_range, False)
        up_y = _scale_camera_to_bounds(up_y, y_range, False)
        up_z = _scale_camera_to_bounds(up_z, z_range, False)

        viewpoint_camera_dict["eye"] = {"x": eye_x, "y": eye_y, "z": eye_z}
        viewpoint_camera_dict["center"] = {"x": at_x, "y": at_y, "z": at_z}
        viewpoint_camera_dict["up"] = {"x": up_x, "y": up_y, "z": up_z}

    layout.update(
        {
            "xaxis": xaxis,
            "yaxis": yaxis,
            "zaxis": zaxis,
            "aspectmode": "cube",
            "camera": viewpoint_camera_dict,
            "dragmode": "turntable",
        }
    )

    fig.update_layout(width=width, height=height, uirevision=uirevision)
    return fig


def _add_mesh_trace(
    fig: go.Figure,
    struct: Trimesh,
    trace_name: str,
    lighting: Dict = Lighting()._asdict(),
    mesh_opacity: float = 1,
    transl: Union[List[float], np.ndarray] = [0, 0, 0],
):
    verts = struct.vertices + transl
    faces = struct.faces
    # If struct has vertex colors defined as texture, use vertex colors
    # for figure, otherwise use plotly's default colors.
    verts_rgb = None
    if isinstance(struct, Trimesh) and struct.vertex_colors is not None:
        verts_rgb = np.asarray(struct.vertex_colors)
    if verts_rgb is not None and verts_rgb.ndim == 1:
        verts_rgb = verts_rgb[None].repeat(len(verts), axis=0)

    # Reposition the unused vertices to be "inside" the object
    # (i.e. they won't be visible in the plot).
    verts_used = np.zeros(verts.shape[0], dtype=np.bool)
    verts_used[np.unique(faces)] = True
    verts_center = verts[verts_used].mean(0)
    verts[~verts_used] = verts_center

    fig.add_trace(
        go.Mesh3d(  # pyre-ignore[16]
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            vertexcolor=verts_rgb,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            lighting=lighting,
            name=trace_name,
            opacity=mesh_opacity,
            showlegend=True,
        )
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene"
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    max_expand = (verts.max(0) - verts.min(0)).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)


def _add_camera_trace(
    fig: go.Figure,
    struct: PerspectiveCamera,
    trace_name: str,
    camera_scale: float,
    show_camera_wireframe: bool = True,
    image: Optional[np.ndarray] = None,
    image_scale_size: float = 1 / 4,
    image_opacity: float = 1,
    marker_size: int = 1,
    transl: Union[List[float], np.ndarray] = [0, 0, 0],
):
    """
    Adds a trace rendering a PerspectiveCamera object to the passed in figure,
    with a given name and in a specific subplot.
    Args:
        fig: plotly figure to add the trace within.
        camera: the PerspectiveCamera object to render.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of sublpots per row.
        camera_scale: the size of the wireframe used to render the
            PerspectiveCamera object.
    """
    cam_wires = get_camera_wireframe(camera_scale)
    c2w = struct.c2w
    focals = struct.focals
    principal_points = struct.principal_points
    cam_colors = (
        np.array(struct.colors, dtype=np.uint8)
        if struct.colors is not None
        else None
    )
    cam_wires_trans = (
        (c2w[:3, :3] @ cam_wires[..., None])[..., 0]
        + c2w[None, :3, -1]
        + transl
    )
    x, y, z = cam_wires_trans.T

    if show_camera_wireframe:
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore [16]
                x=x,
                y=y,
                z=z,
                marker={
                    "color": cam_colors[None].repeat(x.shape[0], axis=0)
                    if cam_colors is not None
                    else None,
                    "size": 1,
                },
                line={
                    "color": cam_colors[None].repeat(x.shape[0], axis=0)
                    if cam_colors is not None
                    else None,
                    "width": 1,
                },
                mode="lines+markers",
                name=trace_name,
                showlegend=image is None,
                legendgroup=trace_name,
            )
        )
    if image is not None:
        H, W = image.shape[:2]
        if show_camera_wireframe:
            plane_pts = get_plane_pts(
                (W, H),
                focals,
                principal_points,
                camera_scale=camera_scale * 1.1,
                scale_factor=image_scale_size,
            )
        else:
            plane_pts = get_plane_pts_from_camera(
                struct, camera_scale=camera_scale, scale_factor=image_scale_size
            )
        h, w = plane_pts.shape[:2]
        plane_pts_trans = (
            (
                (c2w[:3, :3] @ (plane_pts.reshape(-1, 3)[..., None]))[..., 0]
                + c2w[None, :3, -1]
                + transl
            )
        ).reshape(h, w, 3)
        images_sample = cv2.resize(
            image, None, fx=image_scale_size, fy=image_scale_size
        )
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore[16]
                x=plane_pts_trans[..., 0].reshape(-1),
                y=plane_pts_trans[..., 1].reshape(-1),
                z=plane_pts_trans[..., 2].reshape(-1),
                marker={
                    "color": images_sample.reshape(-1, 3),
                    "size": marker_size,
                },
                mode="markers",
                name=trace_name,
                opacity=image_opacity,
                legendgroup=trace_name,
            )
        )

    # Access the current subplot's scene configuration
    plot_scene = "scene"
    current_layout = fig["layout"][plot_scene]

    flattened_wires = cam_wires_trans
    if not show_camera_wireframe:
        flattened_wires = plane_pts_trans.reshape(-1, 3)
    points_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0) - flattened_wires.min(0)).max()
    _update_axes_bounds(points_center, max_expand, current_layout)


def _add_pointcloud_trace(
    fig: go.Figure,
    struct: Union[PointCloud, Segment],
    trace_name: str,
    marker_size: int,
    segment_size: Optional[int] = None,
    point_opacity: float = 1,
    segment_opacity: float = 1,
    show_point: bool = True,
    show_segment: bool = True,
    use_cone_segment: bool = False,
    transl: Union[List[float], np.ndarray] = [0, 0, 0],
):
    """
    Adds a trace rendering a segment object to the passed in figure, with
    a given name and in a specific subplot.
    Args:
        fig: plotly figure to add the trace within.
        struct: Segment object to render.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        marker_size: the size of the rendered points
    """
    if isinstance(struct, (PointCloud, Segment)):
        points = struct.points
        colors = struct.colors
        parents = getattr(struct, "parents", None)
        end_points = getattr(struct, "end_points", None)
    else:
        points = np.asarray(struct.vertices)
        parents = None
        colors = (
            np.asarray(struct.colors[:, :3])
            if struct.colors.size != 0
            else None
        )
        end_points = None
    points = points + transl
    if end_points is not None:
        end_points = end_points + transl

    if colors is not None:
        if isinstance(colors, List):
            colors = np.array(colors, dtype=np.uint8)[None].repeat(
                points.shape[0], axis=0
            )
        if colors.shape[1] == 3:
            template = "rgb(%d, %d, %d)"
            colors = [template % (r, g, b) for r, g, b in colors]
        else:
            raise NotImplementedError("Only support RGB segments right now.")

    if not show_segment or (parents is None and end_points is None):
        # just show the point cloud.
        fig.add_trace(
            go.Scatter3d(  # pyre-ignore[16]
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                opacity=point_opacity,
                marker={"color": colors, "size": marker_size},
                mode="markers",
                name=trace_name,
            )
        )
    else:
        if segment_size is None:
            segment_size = int(np.ceil(marker_size * 0.5))
        if show_point:
            fig.add_trace(
                go.Scatter3d(  # pyre-ignore[16]
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    opacity=point_opacity,
                    marker={"color": colors, "size": marker_size},
                    mode="markers",
                    name=trace_name,
                )
            )
        if use_cone_segment:
            if parents is not None:
                end_points = np.concatenate(
                    [points[:1], points[parents]], axis=0
                )
            fig.add_trace(
                go.Cone(  # pyre-ignore[16]
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    u=end_points[:, 0] - points[:, 0],
                    v=end_points[:, 1] - points[:, 1],
                    w=end_points[:, 2] - points[:, 2],
                    sizemode="absolute",
                    sizeref=segment_size,
                    showscale=False,
                    opacity=segment_opacity,
                    name=None if show_point else trace_name,
                    showlegend=not show_point,
                    legendgroup=trace_name,
                )
            )
        else:
            x, y, z, color = [], [], [], []
            if parents is not None:
                for p, j in zip(parents[1:], np.arange(parents.shape[0])[1:]):
                    # color bones with parent's color
                    x.append(points[p, 0])
                    y.append(points[p, 1])
                    z.append(points[p, 2])
                    color.append(colors[p])
                    x.append(points[j, 0])
                    y.append(points[j, 1])
                    z.append(points[j, 2])
                    color.append(colors[p])
                    x.append(None)
                    y.append(None)
                    z.append(None)
                    color.append(colors[p])
            else:
                for point, end_point, c in zip(points, end_points, colors):
                    x.append(point[0])
                    y.append(point[1])
                    z.append(point[2])
                    color.append(c)
                    x.append(end_point[0])
                    y.append(end_point[1])
                    z.append(end_point[2])
                    color.append(c)
                    x.append(None)
                    y.append(None)
                    z.append(None)
                    color.append(c)
            fig.add_trace(
                go.Scatter3d(  # pyre-ignore[16]
                    x=x,
                    y=y,
                    z=z,
                    line={"color": color, "width": segment_size},
                    mode="lines",
                    opacity=segment_opacity,
                    name=None if show_point else trace_name,
                    showlegend=not show_point,
                    legendgroup=trace_name,
                )
            )

    # Access the current subplot's scene configuration
    plot_scene = "scene"
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    points_center = points.mean(0)
    max_expand = (points.max(0) - points.min(0)).max()
    _update_axes_bounds(points_center, max_expand, current_layout)


def _update_axes_bounds(
    verts_center: np.array,
    max_expand: float,
    current_layout: go.Scene,  # pyre-ignore[11]
):
    """
    Takes in the vertices' center point and max spread, and the current plotly
    figure layout and updates the layout to have bounds that include all traces
    for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices'
            center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the
            referenced trace.
    """
    verts_min = verts_center - max_expand
    verts_max = verts_center + max_expand
    bounds = np.stack([verts_min, verts_max], axis=-1)

    # Ensure that within a subplot, the bounds capture all traces
    old_xrange, old_yrange, old_zrange = (
        current_layout["xaxis"]["range"],
        current_layout["yaxis"]["range"],
        current_layout["zaxis"]["range"],
    )
    x_range, y_range, z_range = bounds
    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])

    xaxis = {"range": x_range}
    yaxis = {"range": y_range}
    zaxis = {"range": z_range}
    current_layout.update({"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis})


def _scale_camera_to_bounds(
    coordinate: float, axis_bounds: Tuple[float, float], is_position: bool
):
    """
    We set our plotly plot's axes' bounding box to [-1,1]x[-1,1]x[-1,1]. As
    such, the plotly camera location has to be scaled accordingly to have its
    world coordinates correspond to its relative plotted coordinates for
    viewing the plotly plot.
    This function does the scaling and offset to transform the coordinates.
    Args:
        coordinate: the float value to be transformed
        axis_bounds: the bounds of the plotly plot for the axis which
            the coordinate argument refers to
        is_position: If true, the float value is the coordinate of a position,
            and so must be moved in to [-1,1]. Otherwise it is a component of a
            direction, and so needs only to be scaled.
    """
    scale = (axis_bounds[1] - axis_bounds[0]) / 2
    if not is_position:
        return coordinate / scale
    offset = (axis_bounds[1] / scale) - 1
    return coordinate / scale - offset
