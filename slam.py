import cv2
import time 
from display import Display 
import numpy as np
#from extractor import FeatureExtractor
import pyg2o
from frame import Frame, denormalize, match_frames, IRt




# gpt
from display3d import Display3d

##gpt
disp3d=Display3d()




W=1280//2
H=720//2
#disp=Display(W,H)
F=255
#orb=cv2.ORB_create()
K=np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))

#fe=FeatureExtractor(K)

class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []

  def display(self):
    for f in self.frames:
      print(f.id)
      print(f.pose)
      print()


disp=Display(W,H)


mapp=Map()

class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

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





#frames=[]

def process_frame(img):
    img=cv2.resize(img,(W,H))
    #kp,desc=orb.detectAndCompute(img,None)
    #for p in kp:
     #   u,v=map(lambda x:int(round(x)),p.pt)
      #  cv2.circle(img,(u,v),color=(0,255,0),radius=3)
    

    #matches,pose=fe.extract(img)
    #if img is None:
    #    return
    #print('%d matches' %(len(matches)))
    #print(pose)

    #for pt1,pt2 in matches:
    #    u1,v1=fe.denormalize(pt1)
    #    u2,v2=fe.denormalize(pt2)


#    frame=Frame(img,K)
#    frames.append(frame)
#    if len(frames)<=1:
    
    frame=Frame(mapp,img,K)
    if frame.id==0:
        return 
    #ret, Rt=match(frames[-1], frames[-2])



    #idx1, idx2, Rt = match_frames(frames[-1], frames[-2])
    #frames[-1].pose = np.dot(Rt, frames[-2].pose)
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)


  # homogeneous 3-D coords
    #pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].pts[idx1], frames[-2].pts[idx2])
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

  # reject pts without enough "parallax" (this right?)
  # reject points behind the camera
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)


    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
#        pt = Point(p)
#        pt.add_observation(frames[-1], idx1[i])
#        pt.add_observation(frames[-2], idx2[i])

        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

       # for pt1, pt2 in ret:
       #u1,v1=denormalize(K,pt1)
       # u2,v2=denormalize(K,pt2)
       # cv2.circle(img,(u1,v1),color=(0,255,0),radius=3)
       # cv2.line(img,(u1,v1),(u2,v2),color=(255,0,0))
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

