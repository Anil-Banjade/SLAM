from __future__ import annotations 
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
from data import VideoFrame
from messages import BackendMessage, FrontendMessage

from frame import Frame
from pnp_tracker import EpipolarAndPnP, PnPConfig
from visualization import (
    VizConfig,
    init_viz,
    log_camera_pinhole,
    log_images,
    log_pose,
    log_points,
    log_scalars,
    log_trajectory,
    log_camera_frustum,
)

def rotation_angle_deg(R:np.ndarray):
    tr=np.trace(R[:3, :3])
    tr=np.clip((tr-1.0)/2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))

@dataclass 
class FrontendConfig:
    video: Path
    width: int = 1280
    height: int = 720
    focal: float = 233.0
    max_frames: int | None = 300

    kf_translation_thresh:float=0.15
    kf_rotation_thresh_deg:float=10.0
    send_every_n_frames:int=1

    enable_rerun:bool=True 
    run_name:str="nerfslam"


class Frontend(mp.Process):
    def __init__( 
        self, 
        conf:FrontendConfig, 
        to_backend:mp.Queue, 
        from_backend:mp.Queue, 
        frontend_done:Event, 
        backend_done:Event, 
        global_pause:Optional[Event]=None, 
    ):
        super().__init__()
        self.conf=conf
        self.to_backend=to_backend
        self.from_backend=from_backend
        self.frontend_done=frontend_done 
        self.backend_done=backend_done
        self.global_pause=global_pause

        self.W=conf.width 
        self.H=conf.height 

        self.K_raw = np.array(
            [[conf.focal, 0, self.W / 2], [0, conf.focal, self.H / 2], [0, 0, 1]],
            dtype=np.float32,
        )

        self.dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        self.K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K_raw, self.dist_coeffs, (self.W, self.H), alpha=0, newImgSize=(self.W, self.H)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw, self.dist_coeffs, None, self.K, (self.W, self.H), cv2.CV_16SC2
        )
        
        self.frames:list[Frame]=[]
        self.frame_idx=0
        self.tracker=EpipolarAndPnP(self.K.astype(np.float32), PnPConfig())
        self.last_keyframe_pose:Optional[np.ndarray]=None 
        self.trajectory:list[np.ndarray]=[]

    def _undistort(self,img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def _should_add_keyframe(self, pose_c2w:np.ndarray)->bool:
        if self.last_keyframe_pose is None:
            return True
        
        dt=pose_c2w[:3, 3]-self.last_keyframe_pose[:3, 3]
        trans=float(np.linalg.norm(dt))
        dR=pose_c2w[:3, :3] @ self.last_keyframe_pose[:3, :3].T
        rot_deg=rotation_angle_deg(dR)
        return (trans > self.conf.kf_translation_thresh) or (rot_deg>self.conf.kf_rotation_thresh_deg)

    def _make_videoframe(self, img_bgr, pose_c2w)->VideoFrame:
        rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb=rgb.astype(np.float32)/255.0
        return VideoFrame(rgb, pose_c2w.astype(np.float32), self.K.astype(np.float32), int(self.frame_idx))
    
    def _track_one(self, img_bgr):
        img_bgr=cv2.resize(img_bgr, (self.W, self.H))
        img_bgr=self._undistort(img_bgr)
        
        dummy_map=type("Map", (), {"frames":self.frames})
        _f=Frame(dummy_map, img_bgr, self.K)

        pose, _initialized, _method=self.tracker.track(self.frames)

        if len(self.tracker.map_points)>0:
            pts=np.array([p.xyz for p in self.tracker.map_points], dtype=np.float32)
            cols=np.array([p.color for p in self.tracker.map_points], dtype=np.uint8)
        else:
            pts=np.zeros((0,3), dtype=np.float32)
            cols=None 
        is_kf=self._should_add_keyframe(pose)
        if is_kf:
            self.last_keyframe_pose=pose.copy()

        return pose, pts, cols, img_bgr, is_kf 

    def _preview_matplotlib(self, render_rgb_u8: np.ndarray | None, gt_rgb_u8: np.ndarray | None) -> None:
        """
        In notebook / Colab, figures created in a multiprocessing child process
        don't show inline. Instead, save each preview to disk so you can open
        them from the file browser.
        """
        if render_rgb_u8 is None and gt_rgb_u8 is None:
            return

        plt.figure(figsize=(10, 4))
        n_cols = 0
        if render_rgb_u8 is not None:
            n_cols += 1
        if gt_rgb_u8 is not None:
            n_cols += 1
        col_idx = 1

        if render_rgb_u8 is not None:
            plt.subplot(1, n_cols, col_idx)
            plt.imshow(render_rgb_u8)
            plt.title("NeRF render")
            plt.axis("off")
            col_idx += 1

        if gt_rgb_u8 is not None:
            plt.subplot(1, n_cols, col_idx)
            plt.imshow(gt_rgb_u8)
            plt.title("Ground truth")
            plt.axis("off")

        plt.tight_layout()
        # Save to disk instead of relying on inline display from a child process
        import os, time
        out_dir = os.path.join("nerfslam_outputs", "previews")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        out_path = os.path.join(out_dir, f"preview_{ts}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[NeRF] saved preview to {out_path}")

    def _handle_backend_message(self, msg):
        match msg:
            case (BackendMessage.SYNC, payload):
                loss, psnr = payload.get("loss"), payload.get("psnr")
                total_steps = payload.get("total_steps")
                warmup_pct = payload.get("warmup_pct")
                n_keyframes = payload.get("n_keyframes", 0)
                # Progress line: step and phase (warmup % or keyframe phase)
                if total_steps is not None:
                    if warmup_pct is not None:
                        print(f"[NeRF] training step {total_steps} — warmup {warmup_pct}%")
                    else:
                        print(f"[NeRF] training step {total_steps} — keyframe phase (keyframes={n_keyframes})")
                if loss is not None or psnr is not None:
                    parts = []
                    if loss is not None:
                        parts.append(f"loss={loss:.6f}")
                    if psnr is not None:
                        parts.append(f"psnr={psnr:.2f}")
                    print("       ", " ".join(parts))
                # Rerun visualization if enabled, otherwise fallback to matplotlib preview
                if self.conf.enable_rerun:
                    log_scalars(loss, psnr)
                    log_images(payload.get("render_rgb_u8"), payload.get("gt_rgb_u8"))
                else:
                    self._preview_matplotlib(
                        payload.get("render_rgb_u8"),
                        payload.get("gt_rgb_u8"),
                    )
            case (BackendMessage.CHECKPOINT, path_str):
                print(f"Backend checkpoint : {path_str}")
            case (BackendMessage.COMPLETED, _):
                print("Backend complete")
            case _:
                pass 

    def run(self):
        init_viz(VizConfig(enabled=self.conf.enable_rerun, recording_id=self.conf.run_name))
        log_camera_pinhole("/world/cam", self.K, self.W, self.H)

        cap = cv2.VideoCapture(str(self.conf.video))
        if not cap.isOpened():
            raise RuntimeError(f"Couldn't open video: {self.conf.video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or not np.isfinite(fps) or fps < 1.0 or fps > 240.0:
            fps = 30.0
        dt = 1.0 / float(fps)
        self.tracker.conf.kalman_dt = dt
        self.tracker.kf.set_dt(dt)

        self.to_backend.put((FrontendMessage.REQUEST_INIT, {"K":self.K, "W":self.W, "H":self.H}))

        try:
            while True:
                if self.conf.max_frames is not None and self.frame_idx >= int(self.conf.max_frames):
                    break 

                if not self.from_backend.empty():
                    self._handle_backend_message(self.from_backend.get())

                if self.global_pause is not None and self.global_pause.is_set():
                    self.global_pause.wait()

                ret, img=cap.read()

                if not ret:
                    break 

                pose, pts, cols, img_u, is_kf=self._track_one(img)

                 # SLAM logs
                cam_position = pose[:3, 3]
                self.trajectory.append(cam_position.copy())
                log_pose("/world/camera", pose)
                log_camera_frustum(f"/world/cameras/{self.frame_idx}", pose, self.K, self.W, self.H)
                log_trajectory("/world/car_path", np.array(self.trajectory))
                log_points("/world/map_points", pts, cols)

                # nerf training messages from frontend to backend 
                vf=self._make_videoframe(img_u, pose)
                if is_kf:
                    self.to_backend.put((FrontendMessage.ADD_KEYFRAME, vf))

                if (self.frame_idx%self.conf.send_every_n_frames)==0:
                    self.to_backend.put((FrontendMessage.ADD_FRAME, vf))

                self.frame_idx+=1

        finally:
            cap.release()

        self.to_backend.put((FrontendMessage.END, None))
        self.backend_done.wait()
        self.frontend_done.set()

















