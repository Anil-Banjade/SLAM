#visualization.py
from __future__ import annotations 
from dataclasses import dataclass 
from typing import Optional 

import numpy as np 

try:
    import rerun as rr 
    import rerun.blueprint as rrb
    _HAS_RERUN=True 

except Exception:
    rr=None 
    rrb=None 
    _HAS_RERUN=False 

@dataclass 
class VizConfig:
        enabled:bool=True
        recording_id:str="nerfslam"

def init_viz(conf:VizConfig)->None:
        if not conf.enabled or not _HAS_RERUN:
            return 
        rr.init("nerfslam", recording_id=conf.recording_id, spawn=True)
        rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.send_blueprint(get_blueprint())

def get_blueprint():
        if not _HAS_RERUN:
            return None 
        blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D", origin="/world", contents=["$origin/**"]),
        # rrb.Vertical(
        #     rrb.Horizontal(
        #         rrb.Spatial2DView(name="render", origin="/world/render"),
        #         rrb.Spatial2DView(name="gt", origin="/world/gt"),
        #     ),
        #     rrb.Horizontal(
        #         rrb.TimeSeriesView(name="loss", origin="/world/loss"),
        #         rrb.TimeSeriesView(name="psnr", origin="/world/psnr"),
        #     ),
        # ),
        # column_shares=[7, 3],
        )
        return rrb.Blueprint(blueprint, collapse_panels=True)

def _mat3_to_quat_xyzw(R:np.ndarray)->np.ndarray:
        m=R
        t=np.trace(m)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        return np.array([x,y,z,w],dtype=np.float32)

def log_pose(name:str, pose_c2w:np.ndarray)->None:
    if not _HAS_RERUN:
        return
    
    R=pose_c2w[:3,:3]
    t=pose_c2w[:3,3]
    q=_mat3_to_quat_xyzw(R)
    q=q / np.linalg.norm(q)
    rr.log(
            name, 
            rr.Transform3D(
                translation=t,
                rotation=rr.datatypes.Quaternion(xyzw=q),
                from_parent=True,
                ),
        )

def log_camera_pinhole(name, K, width, height):
        if not _HAS_RERUN:
            return 
        rr.log(
            name, 
            rr.Pinhole(
                resolution=[width, height],
                focal_length=[float(K[0,0]), float(K[1,1])],
                principal_point=[float(K[0,2]), float(K[1,2])],
                ),
        )

def log_images(render_rgb_u8: Optional[np.ndarray], gt_rgb_u8: Optional[np.ndarray]) -> None:
    if not _HAS_RERUN:
        return
    if render_rgb_u8 is not None:
        rr.log("/world/render", rr.Image(render_rgb_u8).compress(jpeg_quality=90))
    if gt_rgb_u8 is not None:
        rr.log("/world/gt", rr.Image(gt_rgb_u8).compress(jpeg_quality=90))

def log_scalars(loss: Optional[float] = None, psnr: Optional[float] = None) -> None:
    if not _HAS_RERUN:
        return
    if loss is not None:
        rr.log("/world/loss", rr.Scalar(float(loss)))
    if psnr is not None:
        rr.log("/world/psnr", rr.Scalar(float(psnr)))

def log_points(name, pts, colors=None):
        if not _HAS_RERUN:
            return 
        if pts is None or pts.size==0:
            return 
        rr.log(
            name, 
            rr.Points3D(pts.astype(np.float32),
                        colors=colors
                )
            ) 


def log_trajectory(name, traj):
    if not _HAS_RERUN:
        return
    if len(traj) < 2:
        return

    rr.log(
        name,
        rr.LineStrips3D(
            [traj.astype(np.float32)],
            colors=[[255, 0, 0]]
        )
    )

    rr.log(
        name+"_points",
        rr.Points3D(
            traj.astype(np.float32),
            colors=[0,0,255],
            radii=0.02
        )
    )
    
            


def log_camera_frustum(name, pose_c2w, K, width, height, scale=0.2):
    if not _HAS_RERUN:
        return

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    # compute frustum corners in camera frame
    z = scale

    corners = np.array([
        [(0-cx)/fx*z, (0-cy)/fy*z, z],
        [(width-cx)/fx*z, (0-cy)/fy*z, z],
        [(width-cx)/fx*z, (height-cy)/fy*z, z],
        [(0-cx)/fx*z, (height-cy)/fy*z, z],
    ])

    cam_center = np.zeros((1,3))

    pts = np.vstack([cam_center, corners])

    
    R = pose_c2w[:3,:3]
    t = pose_c2w[:3,3]

    pts_world = (R @ pts.T).T + t

    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]

    rr.log(
        name,
        rr.LineStrips3D(
            [pts_world[[i,j]] for i,j in lines],
            colors=[0,255,0]
        )
    )