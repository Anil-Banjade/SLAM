'''
Input: Frames
Output: Parallax and keyframe candidates
'''

import cv2
import numpy as np
import config
from utils.plot_utils import draw_matches, draw_orb_keypoints
'''
this file is becoming more of a utils, important utils
'''

def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)

def extract_features(img, K, dist_coeff=config.dist_coeffs):
    orb = cv2.ORB_create(nfeatures=3000)
    kps, descs = orb.detectAndCompute(img, None)
    
    pts = np.array([(kp.pt[0], kp.pt[1]) for kp in kps], dtype=np.float32)
    
    # Undistort the pixel coordinates before storing them
    pts_undistorted = cv2.undistortPoints(
        pts.reshape(-1, 1, 2), K, dist_coeff, P=K
    )
    pts_undistorted = pts_undistorted.reshape(-1, 2)
    
    return pts_undistorted, descs


def match_frames(f1, f2):                           
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    idx1, idx2 = [], []
    P1 = []
    P2 = []

    if config.args.lowe:
        matches = bf.knnMatch(f1.desc, f2.desc, k=2)

        #lowe's ratio test for filtering bad matches
        ret = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                p1 = f1.kpx_px[m.queryIdx]
                p2 = f2.kpx_px[m.trainIdx]
                P1.append(p1)
                P2.append(p2)
                ret.append((p1, p2))
        assert len(ret) >= 8
        ret = np.array(ret)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(f1.desc, f2.desc)
        matches = sorted(matches, key=lambda x: x.distance)

        ret = []
        for m in matches:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            p1 = f1.kpx_px[m.queryIdx]
            p2 = f2.kpx_px[m.trainIdx]

            P1.append(p1)
            P2.append(p2)
            ret.append((p1, p2))

        assert len(ret) >= 8
        ret = np.array(ret)

    idx1=np.array(idx1)
    idx2=np.array(idx2)
    #do we need to project to essential space? ou i could check properties of essential matrix
    # iff Tx is skew-symmetric and R is orthogonal then E has SVD with higher sigma equal and one sigma zero ( sig, sig, 0) format
    P1 = np.array(P1, dtype=np.float32)
    P2 = np.array(P2, dtype=np.float32)
    
    # Fundamental matrix
    F, maskF = cv2.findFundamentalMat(P1, P2, cv2.FM_RANSAC, 2.0)

    H, maskH = cv2.findHomography(P1, P2, cv2.RANSAC, 1.0)

    f_inliers = maskF.sum()
    h_inliers = maskH.sum()
    if f_inliers < 1.2* h_inliers:
        print(f"Homography wins: H:{h_inliers} vs F:{f_inliers}")
        return None, None, None
    
    maskF = maskF.ravel()
    
    P1 = P1[maskF ==1]
    P2 = P2[maskF ==1]
    idx1 = idx1[maskF ==1]
    idx2 = idx2[maskF == 1]
    
    E, mask_e = cv2.findEssentialMat(P1, P2, config.K, method = cv2.RANSAC, prob = 0.999, threshold = 2.0)
    # E_valid = is_essential_matrix(E)

    mask_e = mask_e.ravel()
    P1 = P1[mask_e == 1]
    P2 = P2[mask_e == 1]
    idx1 = idx1[mask_e == 1]
    idx2 = idx2[mask_e == 1]

    if config.args.show_tests:
        print(f"\nMATCH_FRAMES: {f1.id} and {f2.id}")
        print("Match filter with  lowe's: ", config.args.lowe)
        print("Inliers seen in F and H: ", f_inliers, h_inliers)
        print('Total inliers obtained through F and E: ', len(P1), len(P2))
        print('\n\n')

        draw_img = draw_matches(f1.img, f2.img, P1, P2)
    
    _, R, t, mask = cv2.recoverPose(E, P1, P2, config.K)
    mask = mask.ravel()

    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3,3]  = t.ravel()
    # valid_rot = is_valid_rotation_from_pose(Rt)

    return idx1, idx2, Rt


'''
well a covisiblity graph is expected. Although we do not know whether to implement it implicitly via Frame class storing neighbors and MapPoint storing who has seen it,
we could deduce it from MapPoint too or
keep a separate explicit graph.
'''

def get_neighbors(f, mapp):
    frames = []
    if f is not mapp.frames[-1]:
        print("Error supplied frame is not current frame. No graph available. Process requires current or latest frame \n")
    frames.append(mapp.frames[-2])
    frames.append(mapp.frames[-3])
    return frames
    
def match_frame_to_map(f, mapp):
    neighbor_frames = get_neighbors(f, mapp)
    
    #creating flat list of available mps and their descriptors: one mp may have two or more descs
    candidate_descs = []
    candidate_mps = []

    #use it? to avoid listing same mp? but since we are using observations from frames and retrieveing desc of mp from kp of each frame, even if already seen we get diff descriptor which is what we want
    # seen_mp_ids = set()

    for frame in neighbor_frames:
        for kp_idx, mp in frame.observations.items():
            desc = frame.desc[kp_idx]
            candidate_descs.append(desc)
            candidate_mps.append(mp)

    if not candidate_descs:
        print("\n Error in process of retrieveing neighboring frame's observations. \n")
        return [], []

    candidate_descs_arr = np.array(candidate_descs, dtype=np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    #Lowe is off but could be on? is it ok? yeah

    # matches = bf.knnMatch(f.desc, candidate_descs_arr, k=2)
    matches = bf.knnMatch(f.desc, candidate_descs_arr, k=2)

    #duplicate mapping removal: one mp should be associated with only

    kp_indices = []
    map_points = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            if m.distance < 40:
                kp_indices.append(m.queryIdx)
                map_points.append(candidate_mps[m.trainIdx])

    # for match in matches:
    #     m = match[0]
    #     if m.distance > 15
    #     kp_indices.append(m.queryIdx)
    #     map_points.append(candidate_mps[m.trainIdx])
    if config.args.show_tests:
        print(f"\nTotal MapPoints: {len(mapp._map_points)}")
        print(f"frame {f.id} 2d-3d Correspondance: {len(map_points)} from {len(candidate_mps)} \n")

    # draw_orb_keypoints(f.img, f.kpx_px[kp_indices]) 
    return kp_indices, map_points