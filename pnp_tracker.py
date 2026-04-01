#pnp_tracker.py
from __future__ import annotations 
from dataclasses import dataclass 
from typing import Optional, Tuple 

import cv2 
import numpy as np 
from frame import Frame, match_frames, IRt 
from pose_kalman import PoseVelocityKalman, anchor_translation_to_measurement

@dataclass 
class MapPoint:
    xyz: np.ndarray
    desc: np.ndarray 
    
    color:np.ndarray

class PnPConfig:
    ratio_test = 0.7
    max_map_points_for_pnp = 3000
    pnp_reproj_error_px = 3.5
    pnp_iterations = 200
    pnp_confidence = 0.999
    min_pnp_matches = 30
    min_pnp_inliers = 22
    min_depth = 0.0
    min_homog_w = 5e-3
    
    n_init_frames=8

    # Constant-velocity Kalman on camera center (world); improves PnP guess + smooths path
    use_pose_kalman: bool = True
    kalman_dt: float = 1.0 / 30.0
    kalman_sigma_accel: float = 2.0
    # Lower = trust PnP/epipolar translation more (stay closer to feature geometry).
    kalman_sigma_meas: float = 0.035
    kalman_mahalanobis_gate_sq: float = 12.0

    # After KF update, limit how far the smoothed center may deviate from the raw visual t.
    # ratio scales the cap by frame-to-frame motion; floor avoids a zero cap when static.
    kalman_meas_anchor_ratio: float = 0.55
    kalman_meas_anchor_floor: float = 0.04
    # Epipolar fallback is noisier — allow a looser anchor (multiplier on ratio).
    kalman_meas_anchor_epi_scale: float = 2.0

    # Levenberg–Marquardt refinement on inliers after RANSAC
    pnp_refine_lm: bool = True

    # When PnP passes RANSAC but disagrees strongly with the KF prediction, prefer fallback
    use_innovation_gate: bool = True

    # Map / triangulation: skip matches with almost no motion (weak parallax → bad depth).
    min_pixel_disparity_triangulate: float = 2.5
    # Drop new 3D points that do not reproject cleanly in both keyframes (pixels).
    max_triangulation_reproj_px: float = 8.0

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
        c = self.conf
        self.kf = PoseVelocityKalman(
            dt=c.kalman_dt,
            sigma_accel=c.kalman_sigma_accel,
            sigma_meas=c.kalman_sigma_meas,
            mahalanobis_gate_sq=c.kalman_mahalanobis_gate_sq,
        )

    def _c2w_from_predicted_center(self, R_prev_c2w: np.ndarray, t_world: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_prev_c2w.astype(np.float64)
        T[:3, 3] = t_world.reshape(3)
        return T

    def _maybe_refine_pnp(
        self,
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        inliers: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.conf.pnp_refine_lm or inliers is None or len(inliers) < 4:
            return rvec, tvec
        ii = inliers.ravel()
        obj_in = obj_pts[ii].astype(np.float64)
        img_in = img_pts[ii].astype(np.float64)
        try:
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, self.K, dist, rvec, tvec)
        except cv2.error:
            pass
        return rvec, tvec

    def _mask_disparity_ok(self, f_prev, f_new, idx_prev, idx_new) -> np.ndarray:
        dp = f_new.kps_px[idx_new].astype(np.float64) - f_prev.kps_px[idx_prev].astype(np.float64)
        return np.linalg.norm(dp, axis=1) >= self.conf.min_pixel_disparity_triangulate

    def _reprojection_mask(
        self,
        Xw: np.ndarray,
        f_prev,
        f_new,
        idx_prev: np.ndarray,
        idx_new: np.ndarray,
    ) -> np.ndarray:
        """True where 3D points reproject within max_triangulation_reproj_px in both views."""
        if len(Xw) == 0:
            return np.zeros(0, dtype=bool)
        K = self.K.astype(np.float64)
        T1 = np.linalg.inv(f_prev.pose.astype(np.float64))
        T2 = np.linalg.inv(f_new.pose.astype(np.float64))
        N = Xw.shape[0]
        Xh = np.concatenate([Xw.astype(np.float64), np.ones((N, 1), dtype=np.float64)], axis=1)
        Xc1 = (T1 @ Xh.T).T[:, :3]
        Xc2 = (T2 @ Xh.T).T[:, :3]
        z1, z2 = Xc1[:, 2], Xc2[:, 2]
        front = (z1 > 1e-6) & (z2 > 1e-6)
        u1 = (K[0, 0] * Xc1[:, 0] / z1 + K[0, 2])
        v1 = (K[1, 1] * Xc1[:, 1] / z1 + K[1, 2])
        u2 = (K[0, 0] * Xc2[:, 0] / z2 + K[0, 2])
        v2 = (K[1, 1] * Xc2[:, 1] / z2 + K[1, 2])
        obs1 = f_prev.kps_px[idx_prev].astype(np.float64)
        obs2 = f_new.kps_px[idx_new].astype(np.float64)
        e1 = np.hypot(u1 - obs1[:, 0], v1 - obs1[:, 1])
        e2 = np.hypot(u2 - obs2[:, 0], v2 - obs2[:, 1])
        ok = front & (e1 <= self.conf.max_triangulation_reproj_px) & (e2 <= self.conf.max_triangulation_reproj_px)
        return ok

    def _add_points_from_two_view(self, f_new, f_prev, idx_new, idx_prev):
        disp_ok = self._mask_disparity_ok(f_prev, f_new, idx_prev, idx_new)
        if not np.any(disp_ok):
            return
        idx_new = idx_new[disp_ok]
        idx_prev = idx_prev[disp_ok]

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

        reproj_ok = self._reprojection_mask(pts3d, f_prev, f_new, idx_prev, idx_new)
        if not np.any(reproj_ok):
            return
        pts3d = pts3d[reproj_ok]
        idx_new = idx_new[reproj_ok]
        idx_prev = idx_prev[reproj_ok]
        
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
            if len(self.map_points) > 150000:
                #self.map_points = self.map_points[-100000:]
                self.map_points = self.map_points[-100000:]


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

        rvec, tvec = self._maybe_refine_pnp(obj_pts, img_pts, inliers, rvec, tvec, dist)
        
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

    def _fuse_kalman_translation(self, pose_meas: np.ndarray, f_prev: Frame, relaxed: bool = False) -> None:
        """Assumes predict() already ran for this frame. Updates KF and overwrites translation in pose_meas in-place."""
        t_prev = f_prev.pose[:3, 3].astype(np.float64)
        t_meas = pose_meas[:3, 3].astype(np.float64)
        z = t_meas.copy()
        if relaxed:
            self.kf.update_relaxed(z, meas_noise_scale=4.0)
        else:
            self.kf.update(z)
        t_filt = self.kf.x[:3].copy()
        r = self.conf.kalman_meas_anchor_ratio
        if r > 0.0:
            scale = self.conf.kalman_meas_anchor_epi_scale if relaxed else 1.0
            t_filt = anchor_translation_to_measurement(
                t_filt,
                t_meas,
                t_prev,
                ratio=r * scale,
                floor=self.conf.kalman_meas_anchor_floor,
            )
        self.kf.x[:3] = t_filt
        pose_meas[:3, 3] = t_filt

    def track(self, frames):
        f = frames[-1]
        if f.id == 0:
            f.pose = IRt.copy()
            if self.conf.use_pose_kalman:
                self.kf.init_first(f.pose)
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
            if self.conf.use_pose_kalman:
                self.kf.bootstrap(f_prev.pose, f.pose)
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "init-epi"

        if self.conf.use_pose_kalman:
            self.kf.enter_pnp_mode()
            self.kf.seed_from_prev_if_needed(f_prev.pose)
            if self.kf.ready_for_prediction:
                t_pred = self.kf.predict()
                T_guess = self._c2w_from_predicted_center(f_prev.pose[:3, :3], t_pred)
                guess_w2c = np.linalg.inv(T_guess)
            else:
                guess_w2c = np.linalg.inv(f_prev.pose)
        else:
            guess_w2c = np.linalg.inv(f_prev.pose)

        ok, pose_pnp, _ninliers = self._pnp(f, guess_pose=guess_w2c)

        use_pnp = (
            ok
            and pose_pnp is not None
            and self._is_pose_valid(pose_pnp)
            and (
                not self.conf.use_pose_kalman
                or not self.conf.use_innovation_gate
                or self.kf.gate_measurement(pose_pnp[:3, 3])
            )
        )

        if use_pnp:
            f.pose = pose_pnp.copy()
            if self.conf.use_pose_kalman:
                self._fuse_kalman_translation(f.pose, f_prev, relaxed=False)
            idx_new, idx_prev, _Rt = match_frames(f, f_prev)
            self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
            return f.pose, True, "pnp"

        idx_new, idx_prev, Rt = match_frames(f, f_prev)
        Rt_scaled = self._recover_scale_from_map(f_prev, f, Rt)
        candidate = f_prev.pose @ Rt_scaled
        candidate = self._renormalize_pose(candidate)              
        f.pose = candidate if self._is_pose_valid(candidate) else f_prev.pose.copy()
        if self.conf.use_pose_kalman:
            self._fuse_kalman_translation(f.pose, f_prev, relaxed=True)
        self._add_points_from_two_view(f, f_prev, idx_new, idx_prev)
        return f.pose, True, "epi-fallback"
