'''
abstract view of whole slam framework
'''

import cv2
import numpy as np
import os
import config
from pnp_tracker import EpipolarAndPnP
import argparse
import sys

from map import Map, MapPoint, Frame
from utils import o3d_vis

def process_frame(img, K, mapp, tracker, optimzer = None):
    frame = Frame(mapp, img, K)
    pose, tracked_status, mode_used = tracker.track(mapp.frames)

    print(f"With {mode_used} tracking {tracked_status}")
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
    parser.add_argument("--show_tests", type=bool, default=False, help="Flag to indicate testing logic included")
    parser.add_argument("--lowe", type=bool, default=False, help="to use knn + lowe's")

    args = parser.parse_args()
    config.args = args
    print("Config Arguments: ")
    print(config.args)

    if args.use_images:
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
                    process_frame(img, K, mapp, tracker)
            print("All images processed.")
            print("Poses before local BA") #we would need to start with 6 frame sliding window
                                            #it's okay for now since all processes are abstractly used
            for frame in mapp.frames:
                print(frame.pose)
            # print("INITIATING LOCAL BUNDLE ADJUSTMENT................")
            # P, map_pts = optimzer.start(tracker.obs)
            # print("after local BA")
            # for p in P:
            #     print(p)
            # visualize_world2(P, map_pts)

    else:
        print(f"Generating map with {args.source} frames")

        frame_id = 0
        processed = 0
        cap=cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frame rate of video: ", fps)
        step = max(int(fps/args.fps_divider),1)
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
                    process_frame(frame, K, mapp)
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
        # print(frame.pose)
        poses.append(frame.pose)

    map_points = mapp.map_points
    o3d_vis.visualize_world(poses, map_points)


if __name__ == "__main__":
    K = config.K
    main(K)