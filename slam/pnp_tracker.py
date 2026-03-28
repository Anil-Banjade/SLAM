'''
Pose estimation
Logic of rejection of keyframes should be here too. when we return the pose, if we return reject, the keyframe must not be stored in the Map.
'''

import cv2
import numpy as np
import config
import triangulate
from frame import match_frames
from triangulate import Triangulate
from map import MapPoint
from frame import match_frame_to_map, match_frames
from optimization.bundle_adjustment import Ceres_solver
from utils import o3d_vis

class EpipolarAndPnP:
    def __init__(self, K, mapp, conf = None):
        self.K = K
        self.conf = conf
        self.still_initializing = True #especially use full when homography wins
        # self.obs = Observations() already defined in Map class
        self.mapp = mapp
        self.triangulate = Triangulate(self.K, self.mapp)
        self.optimizer = Ceres_solver(self.K)
    
    def track(self, frames):
        f = frames[-1]
        if f.id == 1:
            f.pose = np.eye(4)
            return f.pose, True, "init-epi"
        f_prev = frames[-2]

        if self.still_initializing:
            idx_prev, idx_new, Rt = match_frames(f_prev, f)

            if idx_prev is None:
                return None, False, "fundamental"
            
            # candidate = f_prev.pose @ Rt
            candidate = Rt @ f_prev.pose
            f.pose = candidate #pose in world coordiante frame
            self.triangulate.add_points_from_two_view(f_prev, f , idx_prev, idx_new)
            self.still_initializing = False

            return f.pose, True, "init-epi"
        
        else:
            pose, success, method = self.track_pnp(f_prev, f)
            if success:
                f.pose = pose
                poses_opt, mps_opt = self.optimizer.start(self.mapp.sliding_window_frames)
                print("DISPLAYING results from optimizer after pnp pose estimation")
                # o3d_vis.visualize_world(poses_opt, mps_opt)
                print(f"Before optimization: \n{f.pose} \n After optimization: \n{poses_opt[-1]} \n")
                self.triangulate.add_points_from_pnp(f_prev, f)
            else: 
                T_velocity = f_prev.pose @ invert_pose(f[-3].pose)
                f.pose = T_velocity @ f_prev # implement good fallback with velocity
            return f.pose, success, method 
    
    def invert_pose(self,T):
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv
    
    def track_pnp(self, f_prev, f):
        MIN_POINTS = 6
        P3P_THRESHOLD = 20

        PNP_ACCEPTANCE = 20
        dist_coeffs = np.zeros(4) # pts are already distorted at frame.extract_features
        use_guess = False
        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.zeros((3, 1), dtype=np.float64)
        kp_idx, map_points = match_frame_to_map(f, self.mapp)

        if config.args.show_tests: 
            idx1, idx2, Rt = match_frames(f_prev, f)
            # if config.args.use_tests:
            #     use_guess = True
            #     rvec_init, _ = cv2.Rodrigues(Rt[:3, :3])
            #     tvec_init = Rt[:3, 3].reshape(3,1)

        if len(map_points) < MIN_POINTS:
            return None, False, "pnp-failed-insufficient-map"



        obj_pts = []
        img_pts = []

        for kp_id, mp in zip(kp_idx, map_points):
            obj_pts.append(mp.xyz)
            img_pts.append(f.kpx_px[kp_id])
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        

        if len(obj_pts) < P3P_THRESHOLD:
            # P3P works on exactly 3 points — cv2 handles the minimal solver
            # internally with RANSAC-style hypothesize-and-verify
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.K, dist_coeffs,
                rvec=rvec_init, tvec=tvec_init,
                flags=cv2.SOLVEPNP_P3P,
                useExtrinsicGuess=use_guess,
                iterationsCount=10000,
                reprojectionError=8,
                confidence=0.99
            )
            method = "pnp-p3p"
        else:
            # ITERATIVE (Levenberg-Marquardt) is more accurate with many points
            # RANSAC wrapper handles outliers
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.K, dist_coeffs,
                rvec=rvec_init, tvec=tvec_init,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=use_guess,
                iterationsCount=10000,
                reprojectionError=8,
                confidence=0.99
            )
            method = "pnp-iterative"
        
        if not success or inliers is None or len(inliers) < MIN_POINTS:
            if success:
                print("low inliers from PnP: " ,len(inliers))
            print(f"Low inliers from {len(obj_pts)} matches found in PnP")
            return None, False, f"{method}-failed"

        if config.args.show_tests:
            print(f"\npnp_tracker: {f.id} ") 
            print(f"inliers: {len(inliers.ravel())}")
            
        inliers = inliers.ravel()
        kp_idx = np.array(kp_idx)
        map_points = np.array(map_points)[inliers]
        kp_idx = kp_idx[inliers]
        for kp_id, mp in zip(kp_idx, map_points):
            f.add_observation(kp_id, mp)
        
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.ravel()

        #REprojection errror here

        return pose, True, method