import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys 
import os 

sys.path.append(os.path.abspath(".."))

from frame import Frame, match_frames, IRt

def create_synthetic_pair(W, H, K):
    img1 = np.zeros((H, W), dtype=np.uint8)
    for _ in range(500):
        x, y = np.random.randint(0, W), np.random.randint(0, H)
        size = np.random.randint(1, 4)
        color = np.random.randint(150, 255)
        cv2.circle(img1, (x, y), size, color, -1)

    img1 = cv2.GaussianBlur(img1, (3, 3), 0)

    theta = np.radians(5)
    R_gt = np.array([[np.cos(theta), 0, np.sin(theta)], 
                     [0, 1, 0], 
                     [-np.sin(theta), 0, np.cos(theta)]])
    t_gt = np.array([[0.5, 0, 0.1]]).T
    Kinv = np.linalg.inv(K)
    H_mat = K @ (R_gt + t_gt @ np.array([[0, 0, 1]])) @ Kinv
    
    img2 = cv2.warpPerspective(img1, H_mat, (W, H))
    return img1, img2, R_gt, t_gt



W, H, F = 640, 360, 500
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]], dtype=float)

imgL, imgR, R_gt, t_gt = create_synthetic_pair(W, H, K)

class MockMap:
    def __init__(self): self.frames = []

mapp_test = MockMap()

f1 = Frame(mapp_test, cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR), K)
f2 = Frame(mapp_test, cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR), K)

print(f"Features found: Frame1={len(f1.pts)}, Frame2={len(f2.pts)}")

try:
    idx1, idx2, Rt_estimated = match_frames(f1, f2)
    f2.pose = np.dot(Rt_estimated, f1.pose)

    print(f"SUCCESS: Matched {len(idx1)} inliers")
    print("Estimated Translation Direction:\n", Rt_estimated[:3, 3])
 
    print("Ground Truth Direction:\n", (t_gt / np.linalg.norm(t_gt)).flatten())
    print("Estimated Rotation:\n", Rt_estimated[:3, :3])
    print("Ground Truth Rotation:\n", R_gt)

except Exception as e:
    print(f"ERROR: {e}")


def visualize_matches(f1, f2, idx1, idx2, img_l, img_r):
    kp1 = [cv2.KeyPoint(p[0]*F + W/2, p[1]*F + H/2, 1) for p in f1.pts]
    kp2 = [cv2.KeyPoint(p[0]*F + W/2, p[1]*F + H/2, 1) for p in f2.pts]
    d_matches = [cv2.DMatch(i, j, 0) for i, j in zip(idx1, idx2)]
    
    out_img = cv2.drawMatches(img_l, kp1, img_r, kp2, d_matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(out_img)
    plt.title(f"Inlier Matches Found: {len(d_matches)}")
    plt.show()

visualize_matches(f1,f2,idx1,idx2, imgL, imgR)
