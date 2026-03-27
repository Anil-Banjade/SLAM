'''
abstract view of whole slam framework
'''

import cv2
import argparse
import sys
from frame import extract

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.observations = []

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.pose = np.eye(4)
        self.img = img

        self.kps_px , self.desc = extract_features(img)
        self.id = len(mapp.frames)
        mapp.frames.append(self)

        self.idx_used = [] #refers to idx of kp_px already used to form Map Points
        self.points_3d = [] #Map point corresponding to Kp_px indexed by idx_used

class Point(object):
    def __init__(self, mapp, loc):
        self.xyz = loc
        self.descs = [] #store desc of frames that observes it, remember to update it when filtering KeyFrames in Map
        self.frames = [] #frames that has observed this Point
        self.idxs = [] #refers to frame index in mapp or kp_px index in corresponding frame in self.frames
        
        self.id = len(mapp.points)
        mapp.point.append(self)
    
    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

def process_frame(img, K, mapp, tracker, optimzer = None):
    frame = Frame(mapp, img, K)
    pose, tracked_status, mode_used = tracker.track(mapp.frames)

    print(f"With {mode} tracking {tracked}")
    print("--------------------------------------------")
    frame.pose = pose

def main(K):
    mapp = Map()
    tracker = EpipolarAndPnP(K, mapp)
    #optimizer...

    parser = argparse.ArgumentParser(description="SLAM modes")

    parser.add_argument("--source", type=str, help="path to frame source")
    parser.add_argument("--use_images", type=bool, default=False, help="Flag to indicate source contains image frames not video")
    parser.add_argument("--use_tests", type=bool, default=False, help="Flag to indicate all tests written to be conducted")
    parser.add_argument("--fps_divider", type=int, default=-1, help="Factor to divide original fps and decrease fps")
    parser.add_argument("--show_tests"), type=bool, default=False, help="Flag to indicate testing logic included"

    args = parser.parse_args()
    config.args = args
    print("Config Arguments: ")
    print(config.args)

    if args.images:
    # Process images from a folder
        image_folder = args.source
        if not os.path.isdir(image_folder):
            print(f"Error: {image_folder} is not a valid directory")
        else:
            # Simple loop over images
            image_files = sorted([
                f for f in os.listdir(image_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            for img_file in image_files:
                img_path = os.path.join(image_folder, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    process_frame(img)
            print("All images processed.")
            print("Poses before local BA") #we would need to start with 6 frame sliding window
                                            #it's okay for now since all processes are abstractly used
            for frame in mapp.frames:
                print(frame.pose)
            print(len(tracker.map_points))
            print("INITIATING LOCAL BUNDLE ADJUSTMENT................")
            P, map_pts = optimzer.start(tracker.obs)
            print("after local BA")
            for p in P:
                print(p)
            visualize_world2(P, map_pts)

    else:
        print(f"Generating map with {args.source} frames")

        frame_id = 0
        processed = 0
        cap=cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frame rate of video: ", fps)
        step = max(int(fps/args.fps),1)
        print("step is ",step)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Total frames:", total_frames)
        while(cap.isOpened()):
            if args.num_frames is not None and processed >= args.num_frames:
                print("exiting slam. Processed 2 frames.")
                break
            ret,frame = cap.read()
            if ret == True:

                if frame_id % step == 0:
                    process_frame(frame)
                    # cv2.imwrite(f"images/{frame_id}.jpg", frame)
                    # print(f"{frame_id} written to ../images")
                    processed +=1
                frame_id += 1
            else:
                print("exiting slam.....")
                break
    

    # visualization
    poses = []
    for frame in mapp.frames:
        print(frame.pose)
        poses.append(frame.pose)

    visualize_world(poses, tracker.map_points)


if __name__ == "__main__":
    K = config.K
    main(K)