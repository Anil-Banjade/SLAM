'''
Pose estimation
Logic of rejection of keyframes should be here too. when we return the pose, if we return reject, the keyframe must not be stored in the Map.
'''

import cv2
import numpy as np
import config
import triangulate
from slam import Point
from frame import match_frames

class EpipolarAndPnP:
    def __init__(self, K, mapp, conf = None):
        self.K = K
        self.conf = conf
        self.still_initializing = True #especially use full when homography wins
        # self.obs = Observations() already defined in Map class
        self.mapp = mapp
    
    def track(self, frames):
        f = frames[-1]
        if f.id == 0:
            f.pose = np.eye(4)
        f_prev = frames[-2]

        if self.still_initializing:
            idx_prev, idx_new, Rt = match_frames(f_prev, f)

            if idx_prev is None:
                return None, False, "fundamental"
            
            candidate = f_prev.pose @ Rt
            candidate = Rt @ f_prev.pose
            f.pose = candidate #pose in world coordiante frame
            self.triangulate.add_points_from_two_view(f_prev, f , idx_prev, idx_new)
            self.still_initializing = False

            return f.pose, True, "init-epi"
        
        else:
            pose, success, method = self.track_pnp(f_prev, f)
            if success:
                f.pose = pose
                self.triangulate.add_points_from_pnp(f_prev, f)
            return f.pose, success, method 
    
    def get_neighbors():
        '''logic yet to be defined that retrieves from covisibility or neighbors graph'''
        idxs = []
        l = len(self.mapp.frames)
        for i in range(3):
            idxs.append(l-i)

    def track_pnp(self, f_prev, f):
        MIN_POINTS = 6
        P3P_THRESHOLD = 30

        if len(self.mapp.points) < MIN_POINTS:
            return None, False, "pnp-failed-insufficeint-map"
        
        map_descs = []
        idx_used = [] #pixel idxs for current frame that will be used for 2d-3d corrspondance
        idxs = get_neighbors()

        for idx in idxs:
            for pts_3d in self.mapp.frames[idx]:
                for desc in pts_3d.descs:
                    map_descs.append(desc)

        map_descs = np.array(map_descs) 

        bf_knn = cv2.BFMatcher(cv2.NORM_HAMMING)

        obj_pts = []
        img_pts = []
        idx_used = []
        idx_map_points = []
        use_guess = False
        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.zeros((3, 1), dtype=np.float64)

        if config.args.lowe:
            matches = bf_knn.knnMatch(map_descs, f.desc, k=2)
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    obj_pts.append(self.map_points[m.queryIdx].xyz)
                    idx_map_points.append(m.queryIdx)
                    img_pts.append(f.kps_px[m.trainIdx])
                    idx_used.append(m.trainIdx)
            use_guess = False
        else:
            if config.args.quick_test:
                idx1, idx2, pose_guess = match_frames(f_prev, f)
                R_guess = pose_guess[:3, :3]
                t_guess = pose_guess[:3,  3]
                rvec_init, _ = cv2.Rodrigues(R_guess)
                tvec_init    = t_guess.reshape(3, 1).astype(np.float64)
                use_guess    = True
            bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches  = bf_cross.match(map_descs, f.desc)
            matches  = sorted(matches, key=lambda x: x.distance)
            for m in matches:
                obj_pts.append(self.map_points[m.queryIdx].xyz)
                idx_map_points.append(m.queryIdx)
                img_pts.append(f.kps_px[m.trainIdx])
                idx_used.append(m.trainIdx)

        #duplciate mapping removal with best kept


        if len(obj_pts) < MIN_POINTS:
            print(f"low matches found in pnp: {len(obj_pts)}")
            return None, False, "pnp-failed-insufficient-matches"
        
        obj_pts = np.array(obj_pts, dtype=np.float32)   # (M, 3)
        img_pts = np.array(img_pts, dtype=np.float32)   # (M, 2)
        
        #solve pnp
        dist_coeffs = np.zeros(4)  #assuming undistorted
        
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
        
        idx_used = np.array(idx_used)
        f.idx_used = set(idx_used[inliers.ravel()])

                mask = np.ones(len(idx_map_points))
        mask[inliers] = True
        rmse_before = reproject_error_3d(self.K , get_transform_matrix(rvec, tvec), obj_pts[inliers], img_pts[inliers])
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts[inliers.ravel()],
            img_pts[inliers.ravel()],
            self.K, dist_coeffs,
            rvec, tvec
        )
        rmse_after = reproject_error_3d(self.K, get_transform_matrix(rvec, tvec), obj_pts[inliers], img_pts[inliers])
        
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3]  = tvec.ravel()

        if config.args.show_tests:
            print("--------------------current focus------------------------------------")
            print(f"{len(inliers)} observed from {len(obj_pts)}")
            print(f"{rmse_before} and {rmse_after} reprojection error from pnp")
            print("--------------------current focus------------------------------------")


        
        return pose, True, method