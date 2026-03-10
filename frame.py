#frame.py
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform 


def add_ones(x):
    return np.concatenate([x,np.ones((x.shape[0],1))],axis=1)

IRt=np.eye(4) 

# this is pose estimation
def extractRt(E, pts1, pts2):
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,_,Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1

    if np.linalg.det(Vt) < 0:
        Vt *= -1

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:,2]

    poses = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]

    best_pose = None
    best_count = 0

    for R,t in poses:

        P1 = np.hstack((np.eye(3), np.zeros((3,1))))
        P2 = np.hstack((R, t.reshape(3,1)))

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts4d /= pts4d[3]

        z1 = pts4d[2]
        z2 = (R @ pts4d[:3] + t.reshape(3,1))[2]

        valid = np.sum((z1 > 0) & (z2 > 0))

        if valid > best_count:
            best_count = valid
            best_pose = (R,t)

    R,t = best_pose

    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3,3]  = t

    return Rt


def extract(img):
    orb = cv2.ORB_create()
     # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1],size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv,pts):
    return np.dot(Kinv,add_ones(pts).T).T[:,0:2]


def denormalize(K,pt):
    ret=np.dot(K,np.array([pt[0],pt[1],1.0]))
    ret/=ret[2]
    return int(round(ret[0])),int(round(ret[1]))

#def match(f1, f2):
def match_frames(f1,f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ratio test
  ret = []
  idx1,idx2=[],[] 

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

      p1 = f1.pts[m.queryIdx]
      p2 = f2.pts[m.trainIdx]
      ret.append((p1, p2))
  assert len(ret) >= 8
  ret = np.array(ret)
  idx1=np.array(idx1)
  idx2=np.array(idx2)

  # fit matrix
  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                          EssentialMatrixTransform,
                          #FundamentalMatrixTransform,
                          min_samples=8,
                          #residual_threshold=1,
                          residual_threshold=0.005,
                          max_trials=200)
  #print(sum(inliers), len(inliers))
  

  # ignore outliers
  ret = ret[inliers]
  
  Rt = extractRt(model.params, ret[:,0], ret[:,1])
#   Rt = extractRt(model.params, ret[:,0], ret[:,1])


  # return
  #return ret, Rt
  return idx1[inliers],idx2[inliers],Rt


class Frame(object):
  def __init__(self, mapp, img, K):
    self.K = K
    self.Kinv = np.linalg.inv(self.K)
    self.pose=IRt
    self.img=img
    
    pts, self.des = extract(img)
    self.kps_px=pts
    self.pts = normalize(self.Kinv, pts)
     
    self.id=len(mapp.frames)
    mapp.frames.append(self)


