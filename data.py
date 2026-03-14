import cv2
from pathlib import Path 
from typing import Iterator, Tuple, List 
import numpy as np 

class VideoFrame:
    def __init__(self, image, pose_c2w, intrinsics, index):
        assert image.ndim==3 and image.shape[2] in (3,4)
        assert pose_c2w.shape==(4,4)
        assert intrinsics.shape == (3, 3)

        # Always store RGB in [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0  
        if image.shape[2] == 4:
            image = image[..., :3]

        self.rgb=image
        self.pose_c2w=pose_c2w.astype(np.float32)
        self.K = intrinsics.astype(np.float32)
        self.index = index

class TrackedSequence:
    def __init__(self, frames:List[VideoFrame]):
        self.frames=frames 
    def __len__(self):
        return len(self.frames)
    def __getitem__(self,idx:int)->VideoFrame:
        return self.frames[idx]

def iterate_video(video_path:Path, resize_hw:Tuple[int, int]|None=None)-> Iterator[np.ndarray]:
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not load video: {video_path}")
    try:
        while True:
            ret, frame=cap.read()
            if not ret:
                break
            if resize_hw is not None:
                h,w=resize_hw 
                frame=cv2.resize(frame, (w,h))
            yield frame 
    finally:
        cap.release()





