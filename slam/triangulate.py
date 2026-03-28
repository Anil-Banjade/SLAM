'''
adding map_points
, should i keep map_point removal logic here too?
'''
import cv2
import numpy as np
import config
from map import MapPoint
from utils import o3d_vis, plot_utils
# from slam import MapPoint

class Triangulate:
    def __init__(self, K, mapp):
        self.K = K
        self.mapp = mapp
    
    def add_points_from_two_view(self, f_prev, f, idx_prev, idx_new):
        p1 = f_prev.kpx_px[idx_prev].T.astype(np.float32)  # (2, N)
        p2 = f.kpx_px[idx_new].T.astype(np.float32)        # (2, N)

        #following world_to_camera convention
        P1 = self.K @ f_prev.pose[:3]  # (3, 4)
        P2 = self.K @ f.pose[:3]       # (3, 4)

        pts4d = cv2.triangulatePoints(P1, P2, p1, p2).T   # (N, 4)
        pts4d /= pts4d[:, 3:]
        pts3d = pts4d[:, :3]  # world coords

        # Cheirality: world-to-cam means pose @ [X;1] gives cam-space point
        pts_h = np.hstack([pts3d, np.ones((len(pts3d), 1))])   # (N, 4)
        pts_cam1 = (f_prev.pose[:3] @ pts_h.T).T   # (N, 3) in cam1 space
        pts_cam2 = (f.pose[:3] @ pts_h.T).T         # (N, 3) in cam2 space

        # Reprojection error for quality filtering
        def reproj_err_per_point(K, pose_3x4, pts3d_w, kpx_px):
            """Returns per-point reprojection error (N,)"""
            pts_h = np.hstack([pts3d_w, np.ones((len(pts3d_w), 1))])
            proj = (K @ pose_3x4 @ pts_h.T).T   # (N, 3)
            proj /= proj[:, 2:]
            diff = proj[:, :2] - kpx_px
            return np.linalg.norm(diff, axis=1)

        err1 = reproj_err_per_point(self.K, f_prev.pose[:3], pts3d, f_prev.kpx_px[idx_prev])
        err2 = reproj_err_per_point(self.K, f.pose[:3],      pts3d, f.kpx_px[idx_new])

        MAX_REPROJ_ERR = 2.0   # tighten this to 1.5 if map still noisy

        good = (
            np.all(np.isfinite(pts3d), axis=1) &
            (pts_cam1[:, 2] > 0) &           # in front of cam1
            (pts_cam2[:, 2] > 0) &           # in front of cam2
            (err1 < MAX_REPROJ_ERR) &        # low reproj error in frame 1
            (err2 < MAX_REPROJ_ERR) &        # low reproj error in frame 2
            (np.abs(pts3d[:, 0]) < 100) &    # tighter bounds — 1000 is too loose
            (np.abs(pts3d[:, 1]) < 100) &
            (pts3d[:, 2] < 100) &
            (pts3d[:, 2] > 0)                # explicit positive depth in world
        )

        if not np.any(good):
            print(f"\n\n[add_points_from_two_view] No good points after filtering for frames {f_prev.id} and {f.id}.")
            # print(f"(mean err1={err1.mean():.2f}, err2={err2.mean():.2f} px)\n\n")
            return

        good_idx = np.where(good)[0]
        good_3d       = pts3d[good_idx]
        idx_prev_good = idx_prev[good_idx]
        idx_new_good  = idx_new[good_idx]

        if config.args.use_tests:
            mps = []
            kp1 = []
            kp2 = []
        color = np.array([0,1,0])
        for id_prev, id_new, xyz in zip(idx_prev_good, idx_new_good, good_3d):
            mp = MapPoint(xyz, color)
            if config.args.use_tests:
                mps.append(mp)
                kp1.append(f_prev.kpx_px[id_prev])
                kp2.append(f.kpx_px[id_new])

            f_prev.add_observation(id_prev, mp)
            f.add_observation(id_new, mp)
            self.mapp.add_map_point(mp)
        if config.args.use_tests:
            poses = []
            poses.append(f_prev.pose)
            poses.append(f.pose)

            print("Testing points triangualated: ")
            o3d_vis.visualize_world(poses, mps) 
            plot_utils.draw_matches(f_prev.img, f.img, kp1, kp2)


                

        if config.args.show_tests:
            print("\nTriangulation frame {f.id} \n")
            print(f"idx_good: {len(idx_prev_good)} good_3d: {len(good_3d)}")
            print("\n\n")
        # breakpoint() 

    def add_points_from_pnp(self, f_prev, f):
        bad = [] #represents bad keypoints to use for triangulatin which is kp that has already been used for triangulation
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(f_prev.desc, f.desc, k=2)

        idx_prev = []
        idx_new = []
        p1 = []
        p2 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                idx_prev.append(m.queryIdx)
                idx_new.append(m.trainIdx)

                p1.append(f_prev.kpx_px[m.queryIdx])
                p2.append(f.kpx_px[m.trainIdx])
        
        for i, (id_prev, id_new) in enumerate(zip(idx_prev, idx_new)):
            if id_prev in f_prev.observations.keys():
                bad.append(i)
            if id_new in f.observations.keys():
                bad.append(i)
        print(f"len idx_prev: {len(idx_prev)}")
        mask = np.ones(len(idx_new), dtype=bool)
        mask[bad] = False
        idx_prev = np.array(idx_prev)
        idx_new  = np.array(idx_new)

        idx_prev = idx_prev[mask] 
        idx_new = idx_new[mask]
        # breakpoint()

        self.add_points_from_two_view(f_prev, f, idx_prev, idx_new)

