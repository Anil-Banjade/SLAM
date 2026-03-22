#frontend.py
from __future__ import annotations 
from dataclasses import dataclass 
from pathlib import Path 
from typing import List, Optional, Tuple 

import cv2
import numpy as np 

from frame import Frame 
from pnp_tracker import EpipolarAndPnP, PnPConfig
import argparse 

from display import Display 
from frame import match_frames, denormalize


from visualization import ( 
    VizConfig, 
    init_viz,
    log_camera_pinhole,
    log_pose, 
    log_points,
    log_trajectory,
    log_camera_frustum,

)

@dataclass
class FrontendConfig:
    video:Path
    width:int=1280
    height:int=720
    focal:float=233.0
    
    
    # width:int=2*2014
    # height:int=2*944
    # focal:float=2*3141 

class Frontend:
    def __init__(self, conf):
        self.conf=conf
        self.W=conf.width
        self.H=conf.height
        
        self.K_raw=np.array(
            [[conf.focal, 0, self.W / 2], [0, conf.focal, self.H / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        self.dist_coeffs = np.array(
            [-0.3, 0.1, 0.0, 0.0, 0.0],   # k1, k2, p1, p2, k3
            dtype=np.float64
        )
        self.K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K_raw, self.dist_coeffs,
            (self.W, self.H), alpha=0,
            newImgSize=(self.W, self.H)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw, self.dist_coeffs, None,
            self.K, (self.W, self.H), cv2.CV_16SC2
        )
        
        self.frames:List[Frame]=[]
        self.frame_idx=0
        self.tracker=EpipolarAndPnP(self.K.astype(np.float32), PnPConfig())
        
        self.trajectory=[]
        #self.disp = Display(self.W, self.H)
    
    def _undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    
    
    def _draw_matches(self, img: np.ndarray) -> np.ndarray:
        
        if len(self.frames) < 2:
            return img

        f1 = self.frames[-1]   
        f2 = self.frames[-2]   

        try:
            idx1, idx2, _ = match_frames(f1, f2)
        except Exception:
            return img

        vis = img.copy()

        for i, (i1, i2) in enumerate(zip(idx1, idx2)):
            u1, v1 = denormalize(self.K, f1.pts[i1])
            u2, v2 = denormalize(self.K, f2.pts[i2])
            
            cv2.line(vis, (u2, v2), (u1, v1), (255, 100, 0), 1)
            
            cv2.circle(vis, (u1, v1), 3, (0, 255, 0), -1)

        # Stats overlay
        n = len(idx1)
        mode = getattr(self.frames[-1], '_mode', 'pnp')
        map_pts = len(self.tracker.map_points)

        cv2.putText(vis, f"Matches: {n}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"MapPts:  {map_pts}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"Frame:   {self.frame_idx}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis

    def _track_one(self, img_bgr):
        img_bgr=cv2.resize(img_bgr, (self.W, self.H))
        img_bgr = self._undistort(img_bgr)
        
        dummy_map=type("Map",(), {"frames":self.frames})
        _f=Frame(dummy_map, img_bgr, self.K)

        pose, _initialized, _method=self.tracker.track(self.frames)

        if len(self.tracker.map_points)>0:
            pts=np.array([p.xyz for p in self.tracker.map_points], dtype=np.float32)
            cols=np.array([p.color for p in self.tracker.map_points], dtype=np.uint8)
        else:
            pts=np.zeros((0,3), dtype=np.float32)
            cols=None
            
        #vis = self._draw_matches(img_bgr)
        #self.disp.point(vis)  
        return pose, pts, cols 
    
    
    

    def run(self, max_frames=None, pose_out=None):
        init_viz(VizConfig(enabled=True, recording_id="slam-only"))
        log_camera_pinhole("/world/cam", self.K, self.W, self.H)

        cap=cv2.VideoCapture(str(self.conf.video))
        if not cap.isOpened():
            raise RuntimeError(f"Couldn't open video: {self.conf.video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or not np.isfinite(fps) or fps < 1.0 or fps > 240.0:
            fps = 30.0
        dt = 1.0 / float(fps)
        self.tracker.conf.kalman_dt = dt
        self.tracker.kf.set_dt(dt)

        try:
            while True:
                if max_frames is not None and self.frame_idx >=max_frames:
                    break 
                ret, img=cap.read()
                if not ret:
                    break
                pose, pts, cols=self._track_one(img)
                
                
                cam_position = pose[:3,3]
                self.trajectory.append(cam_position.copy())
                
                log_pose(f"/world/camera", pose)
                log_camera_frustum(f"/world/cameras/{self.frame_idx}",pose,self.K,self.W,self.H)

                log_trajectory("/world/car_path", np.array(self.trajectory))
                
                log_points("/world/map_points", pts, cols)
                
                self.frame_idx+=1
        finally:
            cap.release()

        if pose_out is not None:
            poses=np.array([fr.pose for fr in self.frames], dtype=np.float32)
            np.save(pose_out, poses)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test_nyc.mp4")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--pose_out", type=str, default=None)
    args=parser.parse_args()

    conf=FrontendConfig(video=Path(args.video))
    fe=Frontend(conf)
    fe.run(max_frames=args.max_frames, pose_out=Path(args.pose_out) if args.pose_out else None)

if __name__=="__main__":
    main()



                
                

