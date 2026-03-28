import numpy as np
import open3d as o3d
import sys
import os
from dataclasses import dataclass

@dataclass
class MapPoint:
    xyz: np.ndarray    # (3,) world position
    desc: np.ndarray   # descriptor vector
    color: np.ndarray  # (3,) RGB in [0, 1]

def create_point(position, radius=0.5, color=[0, 0, 1]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(position)          # fixed typo: postion -> position
    return sphere

# pose: 4x4 world-frame [R|t] homogeneous transform
def create_camera_frame(pose, size=10, color=[1, 1, 0]):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(np.linalg.inv(pose))               # fixed: T -> pose
    return frame

def visualize_world(poses, map_points=None, window_name="world visualization"):

    geoms = []
    for mp in additional_axes():
        pt = create_point(mp.xyz, radius = 0.05,color=mp.color.tolist())
        geoms.append(pt)
    
    for pose in poses:
        print("from o3d")
        print(pose)
        cam = create_camera_frame(pose, size = 1)
        geoms.append(cam)

    points = []    
    pcd = o3d.geometry.PointCloud()

    for mp in map_points:
        points.append(mp.xyz)

    points = np.array(points)
    pcd.points = o3d.utility.Vector3dVector(points)
    geoms.append(pcd)
    
    print(f"{len(points)} points and {len(poses)} poses")
    o3d.visualization.draw_geometries(geoms, window_name=window_name)




def additional_axes():
    map_points = []
        # Points along +X axis — red
    desc = 12
    desc = np.array(desc).astype(np.float32)
    for x in np.linspace(0, 5, 10):
        map_points.append(MapPoint(
            xyz=np.array([x, 0.0, 0.0]),
            color=np.array([1.0, 0.0, 0.0]),
            desc = desc
        ))

    # Points along +Y axis — green
    for y in np.linspace(0.2, 5, 10):
        map_points.append(MapPoint(
            xyz=np.array([0.0, y, 0.0]),
            color=np.array([0.0, 1.0, 0.0]),
            desc = desc
        ))

    # Points along +Z axis — blue
    for z in np.linspace(0.2, 5, 10):
        map_points.append(MapPoint(
            xyz=np.array([0.0, 0.0, z]),
            color=np.array([0.0, 0.0, 1.0]),
            desc = desc
        ))
    
    return map_points