import cv2
import time 
from display import Display 
import numpy as np
import pyg2o
from frame import Frame, denormalize, match_frames, IRt

from display3d import Display3d
from pnp_tracker import EpipolarAndPnP


disp3d=Display3d()


W=1280
H=720
F=233.0
K=np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))

tracker=EpipolarAndPnP(K)

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

    pose, tracked, mode = tracker.track(mapp.frames)
    frame.pose = pose

    # visualize matches (optional)
    if frame.id > 0:

        f1 = mapp.frames[-1]
        f2 = mapp.frames[-2]

        idx1, idx2, _ = match_frames(f1,f2)

        for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
            u1,v1 = denormalize(K,pt1)
            u2,v2 = denormalize(K,pt2)

            cv2.circle(img,(u1,v1),3,(0,255,0))
            cv2.line(img,(u1,v1),(u2,v2),(255,0,0))

    disp.point(img)

    all_points = np.array([p.xyz for p in tracker.map_points])
    all_poses  = np.array([f.pose for f in mapp.frames])

    disp3d.update_map(all_points, all_poses)

if __name__=="__main__":
    cap=cv2.VideoCapture("test_nyc.mp4")
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret==True:
            process_frame(frame)
        else:
            break

