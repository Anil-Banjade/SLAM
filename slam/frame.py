'''
Input: Frames
Output: Parallax and keyframe candidates
'''

import cv2
import numpy as np
import config
from utils.plot_utils import draw_matches
'''
this file is becoming more of a utils, important utils
'''

def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)

def extract_features(img, K, dist_coeff=config.dist_coeffs):
    orb = cv2.ORB_create(nfeatures=3000)
    kps, des = orb.detectAndCompute(img, None)
    
    pts = np.array([(kp.pt[0], kp.pt[1]) for kp in kps], dtype=np.float32)
    
    # Undistort the pixel coordinates before storing them
    pts_undistorted = cv2.undistortPoints(
        pts.reshape(-1, 1, 2), K, dist_coeffs, P=K
    )
    pts_undistorted = pts_undistorted.reshape(-1, 2)
    
    return pts_undistorted, des


def match_frames(f1, f2):                           
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    idx1, idx2 = [], []
    P1 = []
    P2 = []

    if config.args.lowe:
        matches = bf.knnMatch(f1.des, f2.des, k=2)

        #lowe's ratio test for filtering bad matches
        ret = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                p1 = f1.kps_px[m.queryIdx]
                p2 = f2.kps_px[m.trainIdx]
                P1.append(p1)
                P2.append(p2)
                ret.append((p1, p2))
        assert len(ret) >= 8
        ret = np.array(ret)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(f1.des, f2.des)
        matches = sorted(matches, key=lambda x: x.distance)

        ret = []
        for m in matches:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            p1 = f1.kps_px[m.queryIdx]
            p2 = f2.kps_px[m.trainIdx]

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
    E_valid = is_essential_matrix(E)

    mask_e = mask_e.ravel()
    P1 = P1[mask_e == 1]
    P2 = P2[mask_e == 1]
    idx1 = idx1[mask_e == 1]
    idx2 = idx2[mask_e == 1]

    if config.args.show_tests:
        print("\nMATCH_FRAMES:")
        print("Match filter with  lowe's: ", config.args.lowe)
        print("Inliers seen in F and H: ", f_inliers, h_inliers)
        print('Total inliers obtained through F and E: ', len(P1), len(P2))

        draw_img = draw_matches(f1.img, f2.img, P1, P2)
        plt.figure()
        plt.imshow(draw_img)
        plt.title(f"Feature Matching: {f1.id} and {f2.id}")
    
    _, R, t, mask = cv2.recoverPose(E, P1, P2, config.K)
    mask = mask.ravel()

    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3,3]  = t.ravel()
    valid_rot = is_valid_rotation_from_pose(Rt)

    return idx1, idx2, Rt