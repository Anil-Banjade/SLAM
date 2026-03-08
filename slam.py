import cv2
import time 
from display import Display 
import numpy as np
import pyg2o
from frame import Frame, denormalize, match_frames, IRt


from display3d import Display3d

disp3d=Display3d()


W=1280//2
H=720//2
F=255
K=np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))


class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []

  def display(self):
    for f in self.frames:
      #print(f.id)
      #print(f.pose)
      print()
disp=Display(W,H)

mapp=Map()
class Point(object):
  def __init__(self,mapp, loc):
    self.xyz = loc
    self.frames = []
    self.idxs = []

    self.id=len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxs.append(idx)

def triangulate(pose1, pose2, pts1, pts2):
  return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frame(img):
    img=cv2.resize(img,(W,H))
    frame=Frame(mapp,img,K)
    if frame.id==0:
        return 
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)


    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)


    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    disp.point(img)
    if frame.id % 100 == 0:  # Print every 10th frame
        mapp.display()
        print(f"Frame {frame.id}: {len(mapp.points)} map points, {len(mapp.frames)} keyframes")
    #mapp.display()


    #gpt
    all_points = np.array([p.xyz[:3] for p in mapp.points])
    all_poses  = np.array([f.pose for f in mapp.frames])
    #gpt
    
    disp3d.update_map(all_points,all_poses)
    
if __name__=="__main__":
    cap=cv2.VideoCapture("test_nyc.mp4")
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret==True:
            process_frame(frame)
        else:
            break

