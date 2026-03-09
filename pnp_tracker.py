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

def triangulate(pose1_c2w, pose2_c2w, pts1, pts2):
    # cv2.triangulatePoints strictly requires World-to-Camera matrices
    P1 = np.linalg.inv(pose1_c2w)[:3]
    P2 = np.linalg.inv(pose2_c2w)[:3]
    pts1 = pts1.T
    pts2 = pts2.T
    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    return pts4d.T

class EpipolarAndPnP:
    def __init__(self, K, conf=None):
        self.K = K.astype(np.float32)
        self.conf = conf or PnPConfig()
        self.map_points = []

    def _add_points_from_two_view(self, f_new, f_prev, idx_new, idx_prev):
        pts4d = triangulate(f_new.pose, f_prev.pose, f_new.pts[idx_new], f_prev.pts[idx_prev])
        pts4d /= pts4d[:, 3:]
        pts3d = pts4d[:, :3]
        
        # Transform points back to the new camera frame to check true depth
        w2c_new = np.linalg.inv(f_new.pose)
        pts_cam = (w2c_new[:3, :3] @ pts3d.T).T + w2c_new[:3, 3]
        
        good = (
            (np.abs(pts4d[:, 3]) > self.conf.min_homog_w) &
            (pts_cam[:, 2] > 0) &  # Check Camera Z, NOT World Z
            (np.abs(pts3d[:, 0]) < 1000) &
            (np.abs(pts3d[:, 1]) < 1000) &
            (np.abs(pts3d[:, 2]) < 1000)
        )
        if not np.any(good):
            return
        
        for i in np.where(good)[0]:
            xyz = pts3d[i].astype(np.float32)
            desc = f_new.des[idx_new[i]].copy()
            self.map_points.append(MapPoint(xyz=xyz, desc=desc))

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
            # guess_pose is W2C, which is correct for cv2.solvePnPRansac
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

        if (not ok) or (inliers is None) or (len(inliers) < self.conf.min_pnp_matches):
            return False, None, 0
        
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.astype(np.float32)
        pose[:3, 3] = tvec.reshape(-1).astype(np.float32)

        # Invert to return a Camera-to-World pose
        pose = np.linalg.inv(pose)

        return True, pose, int(len(inliers))

    def track(self, frames):
        f = frames[-1]
        if f.id == 0:
            f.pose = IRt.copy()
            return f.pose, False, "init-epi"

        f_prev = frames[-2]

        if f.id == 1 or len(self.map_points) < self.conf.min_pnp_matches:
            idx_new, idx_prev, Rt = match_frames(f, f_prev)
            # f.pose must be Camera-to-World
            f.pose = f_prev.pose @ np.linalg.inv(Rt)
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "init-epi"
        
        # PnP needs a World-to-Camera guess
        guess_w2c = np.linalg.inv(f_prev.pose)
        ok, pose, _ninliers = self._pnp(f, guess_pose=guess_w2c)
        
        if ok and pose is not None:
            f.pose = pose # _pnp correctly outputs C2W
            idx_new, idx_prev, _Rt = match_frames(f, f_prev)
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "pnp"

        idx_new, idx_prev, Rt = match_frames(f, f_prev)
        # Fallback must also maintain the C2W standard
        f.pose = f_prev.pose @ np.linalg.inv(Rt)
        self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
        return f.pose, True, "epi-fallback"