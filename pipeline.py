from __future__ import annotations
import argparse 
from pathlib import Path 

import torch 
import torch.multiprocessing as mp 


from backend import Backend, BackendConfig
from nerf_frontend import Frontend, FrontendConfig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test_nyc.mp4")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="nerfslam_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal", type=float, default=233.0)
    parser.add_argument("--no_rerun", action="store_true")
    parser.add_argument("--send_every_n_frames", type=int, default=1)
    parser.add_argument("--kf_translation_thresh", type=float, default=0.15)
    parser.add_argument("--kf_rotation_thresh_deg", type=float, default=10.0)



    args = parser.parse_args()
    
    video_path=Path(args.video)
    output_dir=Path(args.output_dir)

    to_backend=mp.Queue()
    to_frontend=mp.Queue()

    frontend_done=mp.Event()
    backend_done=mp.Event()
    global_pause=mp.Event()

    fe=Frontend(
        FrontendConfig(
            video=video_path, 
            width=args.width,
            height=args.height, 
            focal=args.focal, 
            max_frames=args.max_frames,
            kf_translation_thresh=args.kf_translation_thresh,
            kf_rotation_thresh_deg=args.kf_rotation_thresh_deg,
            send_every_n_frames=args.send_every_n_frames,
            enable_rerun=not args.no_rerun,
            run_name="nerfslam",

        ),
        to_backend=to_backend,
        from_backend=to_frontend,
        frontend_done=frontend_done,
        backend_done=backend_done,
        global_pause=global_pause,
    )

    be = Backend(
        BackendConfig(output_dir=output_dir, device=args.device),
        from_frontend=to_backend,
        to_frontend=to_frontend,
        backend_done=backend_done,
        global_pause=global_pause,
    )

    be.start()
    fe.start()

    fe.join() 
    be.join()

if __name__ =="__main__":
    mp.set_start_method("spawn", force=True)
    main()





