#pnp_tracker.py
from __future__ import annotations 
from dataclasses import dataclass 
from typing import Optional, List, Tuple 

import cv2 
import numpy as np 
from frame import Frame, match_frames, IRt 

@dataclass 
class MapPoint:
    xyz: np.ndarray
    desc: np.ndarray 
    
    color:np.ndarray

class PnPConfig:
    ratio_test = 0.75
    max_map_points_for_pnp = 3000
    pnp_reproj_error_px = 4.0
    pnp_iterations = 200
    pnp_confidence = 0.999
    min_pnp_matches = 30
    min_pnp_inliers = 20
    min_depth = 0.0
    min_homog_w = 5e-3
    
    n_init_frames=8

def triangulate(pose1_c2w, pose2_c2w, pts1, pts2, K):
    P1 = K @ np.linalg.inv(pose1_c2w)[:3]
    P2 = K @ np.linalg.inv(pose2_c2w)[:3]
    pts1 = pts1.T.astype(np.float32)
    pts2 = pts2.T.astype(np.float32)
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    return pts4d.T

class EpipolarAndPnP:
    def __init__(self, K, conf=None):
        self.K = K.astype(np.float32)
        self.conf = conf or PnPConfig()
        self.map_points = []
        

    def _add_points_from_two_view(self, f_new, f_prev, idx_new, idx_prev):
        pts4d = triangulate(f_prev.pose, f_new.pose, f_prev.kps_px[idx_prev], f_new.kps_px[idx_new], self.K)
        
        
        if not np.all(np.isfinite(pts4d)):
            return
        w_vals = pts4d[:, 3]
        safe = np.abs(w_vals) > self.conf.min_homog_w
        if not np.any(safe):
            return
        
        pts4d = pts4d[safe]
        idx_new  = idx_new[safe]     
        idx_prev = idx_prev[safe]
        
        
        pts4d /= pts4d[:, 3:]
        pts3d = pts4d[:, :3]
        
        w2c_new = np.linalg.inv(f_new.pose)
        pts_cam = (w2c_new[:3, :3] @ pts3d.T).T + w2c_new[:3, 3]
        
        good = (
            np.all(np.isfinite(pts3d), axis=1) &       
            (pts_cam[:, 2] > 0) &
            (np.abs(pts3d[:, 0]) < 1000) &
            (np.abs(pts3d[:, 1]) < 1000) &
            (np.abs(pts3d[:, 2]) < 1000)
        )
        if not np.any(good):
            return
        
        for i in np.where(good)[0]:
            xyz = pts3d[i].astype(np.float32)
            desc = f_new.des[idx_new[i]].copy()
            
            px = f_new.kps_px[idx_new[i]].astype(int)
            x,y = px

            if 0 <= x < f_new.img.shape[1] and 0 <= y < f_new.img.shape[0]:
                color = f_new.img[y,x]
            else:
                color = np.array([255,255,255])
            self.map_points.append(MapPoint(xyz=xyz, desc=desc, color=color))
            if len(self.map_points) > 50000:
                self.map_points = self.map_points[-30000:]

    def _pnp(self, f, guess_pose):
        if f.des is None or len(self.map_points) < self.conf.min_pnp_matches:
            return False, None, 0

        if len(self.map_points) > self.conf.max_map_points_for_pnp:
            ids = np.random.choice(len(self.map_points), self.conf.max_map_points_for_pnp, replace=False)
            mp_subset = [self.map_points[i] for i in ids]
        else:
            mp_subset = self.map_points 

        mp_descs = np.stack([p.desc for p in mp_subset], axis=0)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(f.des, mp_descs, k=2)

        obj_pts = []
        img_pts = []

        for m, n in matches:
            if m.distance < self.conf.ratio_test * n.distance:
                obj_pts.append(mp_subset[m.trainIdx].xyz)
                img_pts.append(f.kps_px[m.queryIdx])

        if len(obj_pts) < self.conf.min_pnp_matches:
            return False, None, 0 
        
        obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
        img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)

        dist = np.zeros((4, 1), dtype=np.float32)

        rvec = None
        tvec = None 
        use_guess = False 

        if guess_pose is not None:
            R0 = guess_pose[:3, :3].astype(np.float32)
            t0 = guess_pose[:3, 3].astype(np.float32)
            rvec, _ = cv2.Rodrigues(R0)
            tvec = t0.reshape(3, 1)
            use_guess = True 

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts, 
            imagePoints=img_pts,
            cameraMatrix=self.K, 
            distCoeffs=dist,
            rvec=rvec, 
            tvec=tvec, 
            useExtrinsicGuess=use_guess, 
            iterationsCount=self.conf.pnp_iterations, 
            reprojectionError=self.conf.pnp_reproj_error_px, 
            confidence=self.conf.pnp_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE, 
        )

        if (not ok) or (inliers is None) or (len(inliers) < self.conf.min_pnp_inliers):
            return False, None, 0
        
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R.astype(np.float64)
        pose[:3, 3] = tvec.reshape(-1).astype(np.float64)

        pose = np.linalg.inv(pose)

        return True, pose, int(len(inliers))
    
    
    
    def _recover_scale_from_map(self, f_prev, f_new, Rt_unit):
    
        if len(self.map_points) < 10:
            return Rt_unit

        w2c_prev = np.linalg.inv(f_prev.pose)
        mp_xyz = np.array([p.xyz for p in self.map_points], dtype=np.float32)
        pts_cam = (w2c_prev[:3, :3] @ mp_xyz.T).T + w2c_prev[:3, 3]
        
        visible = pts_cam[:, 2] > 0.1
        if np.sum(visible) < 5:
            return Rt_unit

        pts_cam_vis = pts_cam[visible]

        median_depth = np.median(pts_cam_vis[:, 2])
        if median_depth <= 0:
            return Rt_unit

        target_baseline = np.clip(median_depth * 0.02, 0.01, 2.0)

        Rt_scaled = Rt_unit.copy()
        Rt_scaled[:3, 3] *= target_baseline
        return Rt_scaled

    def _renormalize_pose(self,pose: np.ndarray) -> np.ndarray:
        R = pose[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        pose[:3, :3] = U @ Vt          
        return pose


    def _is_pose_valid(self,pose: np.ndarray) -> bool:
        if not np.all(np.isfinite(pose)):
            return False
        if np.linalg.norm(pose[:3, 3]) > 1e4:   
            return False
        return True

    def track(self, frames):
        f = frames[-1]
        if f.id == 0:
            f.pose = IRt.copy()
            return f.pose, False, "init-epi"

        f_prev = frames[-2]
        
        
        still_initializing = (
        f.id < self.conf.n_init_frames or
        len(self.map_points) < self.conf.min_pnp_matches
        )

        if still_initializing:
            idx_new, idx_prev, Rt = match_frames(f, f_prev)
            candidate = f_prev.pose @ Rt
            candidate = self._renormalize_pose(candidate)          
            f.pose = candidate if self._is_pose_valid(candidate) else f_prev.pose.copy()
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "init-epi"
        
        guess_w2c = np.linalg.inv(f_prev.pose)
        ok, pose, _ninliers = self._pnp(f, guess_pose=guess_w2c)
        
        if ok and pose is not None and self._is_pose_valid(pose):
            f.pose = pose 
            idx_new, idx_prev, _Rt = match_frames(f, f_prev)
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "pnp"

        idx_new, idx_prev, Rt = match_frames(f, f_prev)
        Rt_scaled = self._recover_scale_from_map(f_prev, f, Rt)
        candidate = f_prev.pose @ Rt_scaled
        candidate = self._renormalize_pose(candidate)              
        f.pose = candidate if self._is_pose_valid(candidate) else f_prev.pose.copy()
        self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
        return f.pose, True, "epi-fallback"
