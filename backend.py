from __future__ import annotations 

from dataclasses import dataclass 
from pathlib import Path 
from threading import Event 
from typing import Optional 

import numpy as np 
import torch 
import torch.multiprocessing as mp 
from torch import nn 

from lie import se3_exp 
from data import VideoFrame 
from messages import BackendMessage, FrontendMessage
from nerf_model import( 
    NeRF,
    NeRFRenderConfig,
    get_rays,
    positional_encoding,
    render_image,
    render_volume_density,
    sample_points,
)

def psnr_from_mse(mse:float)->float:
    return float(-10.0 * np.log10(max(1e-12, mse)))


@dataclass
class BackendConfig:
    output_dir:Path=Path("nerfslam_outputs")
    device:str="cuda" if torch.cuda.is_available() else "cpu"

    warmup_steps:int=200
    steps_per_keyframe:int=200
    steps_per_sync:int=50
    batch_size:int=8192
    n_samples:int=64
    lr:float=5e-4 
    enable_pose_refinement:bool=True 
    pose_lr:float=2e-3 
    pose_reg_weight=2e-3
    preview_every_steps:int=1000 

class Backend(mp.Process):
    def __init__(
        self, 
        conf:BackendConfig, 
        from_frontend:mp.Queue, 
        to_frontend:mp.Queue, 
        backend_done:Event, 
        global_pause:Optional[Event]=None, 
    ):
        super().__init__()
        self.conf=conf
        self.from_frontend=from_frontend 
        self.to_frontend=to_frontend
        self.to_backend_done=backend_done 
        self.global_pause=global_pause 

        self.frames:list[VideoFrame]=[]
        self.keyframes:list[VideoFrame]=[]
        
        self.base_poses_c2w:dict[int, torch.Tensor]={}
        self.pose_deltas_se3:dict[int, torch.nn.Parameter]={}
        self.pose_param_group_added=False 

        self.model:Optional[NeRF]=None 
        self.optimizer:Optional[torch.optim.Optimizer]=None
        self.loss_fn=nn.MSELoss()

        self.render_conf:Optional[NeRFRenderConfig]=None 
        self.total_steps=0

    def __init_from_payload(self, payload:dict):
        K=payload["K"]
        W=int(payload["W"])
        H=int(payload["H"])
        focal=float(K[0][0])

        self.render_conf=NeRFRenderConfig(H=H, W=W, focal=focal, N_samples=self.conf.n_samples, device=self.conf.device)

        input_dim=3+2*3*10 

        self.model=NeRF(input_dim=input_dim).to(self.conf.device)

        self.optimizer=torch.optim.Adam(
                [{"params":self.model.parameters(), "lr":self.conf.lr}],
                lr=self.conf.lr, 
        )
        self.conf.output_dir.mkdir(parents=True, exist_ok=True)

    def _sample_training_frame(self)->VideoFrame:
        if len(self.keyframes) > 0:
            return self.keyframes[np.random.randint(len(self.keyframes))]
        return self.frames[np.random.randint(len(self.frames))]

    def _register_pose_if_needed(self, f:VideoFrame)-> None:
        device=torch.device(self.conf.device)
        idx=int(f.index)

        if idx not in self.base_poses_c2w:
            self.base_poses_c2w[idx]=torch.from_numpy(f.pose_c2w).to(device).float()

        if not self.conf.enable_pose_refinement:
            return 
        
        if idx not in self.pose_deltas_se3:
            requires_grad=idx!=0 
            delta=torch.zeros(6, device=device, dtype=torch.float32, requires_grad=requires_grad)
            self.pose_deltas_se3[idx] = torch.nn.Parameter(delta, requires_grad=requires_grad)

            assert self.optimizer is not None 
            self.optimizer.add_param_group({"params": [self.pose_deltas_se3[idx]], "lr":self.conf.pose_lr})

    def _refined_pose_c2w(self, f:VideoFrame)->torch.Tensor:
        idx=int(f.index)
        base=self.base_poses_c2w[idx]
        if not self.conf.enable_pose_refinement:
            return base 
        delta=self.pose_deltas_se3[idx]
        T=se3_exp(delta)

        return base@T

    
    def _train_steps(self)->tuple[float, float]:
        assert self.model is not None 
        assert self.optimizer is not None 
        assert self.render_conf is not None 

        device=torch.device(self.conf.device)
        f=self._sample_training_frame()
        self._register_pose_if_needed(f)
        
        target=torch.from_numpy(f.rgb).to(device).float()
        pose=self._refined_pose_c2w(f)

        rays_o, rays_d=get_rays(self.render_conf.H, self.render_conf.W, self.render_conf.focal, pose, device=device)

        coords=torch.stack(
            torch.meshgrid(
                torch.arange(self.render_conf.H, device=device), 
                torch.arange(self.render_conf.W, device=device), 
                indexing="ij",
                )
        ).reshape(2,-1).T

        select_inds=torch.randint(0, coords.shape[0], (self.conf.batch_size,), device=device)
        select_coords=coords[select_inds].long()

        rays_o_s=rays_o[select_coords[:,0], select_coords[:, 1]]
        rays_d_s = rays_d[select_coords[:, 0], select_coords[:, 1]]
        target_s = target[select_coords[:, 0], select_coords[:, 1]] 

        pts,t_vals=sample_points(rays_o_s, rays_d_s, self.conf.n_samples, near=2.0, far=6.0)

        embedded=positional_encoding(pts, num_freqs=10)
        raw=self.model(embedded)
        pred_rgb=render_volume_density(raw, t_vals, rays_d_s)

        loss=self.loss_fn(pred_rgb, target_s)

        if self.conf.enable_pose_refinement and (int(f.index) in self.pose_deltas_se3):
            loss=loss+self.conf.pose_reg_weight * self.pose_deltas_se3[int(f.index)].square().mean()

        self.optimizer.zero_grad()
        loss.backward()

        mse=float(loss.item())
        return mse, psnr_from_mse(mse)


    @torch.no_grad()
    def _preview(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        assert self.render_conf is not None
        # pick a keyframe if available
        f = self.keyframes[-1] if len(self.keyframes) > 0 else self.frames[-1]
        self._register_pose_if_needed(f)
        pose = self._refined_pose_c2w(f).detach().cpu().numpy()
        rgb = render_image(self.model, pose, self.render_conf)
        render_u8 = (rgb * 255.0).astype(np.uint8)
        gt_u8 = (np.clip(f.rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return render_u8, gt_u8


    def _checkpoint(self):
        assert self.model is not None 
        ckpt_path=self.conf.output_dir/f"nerf_{self.total_steps:08d}.pt"
        payload={"model":self.model.state_dict()}

        if self.conf.enable_pose_refinement:
            payload["base_poses_c2w"]={k:v.detach().cpu() for k, v in self.base_poses_c2w.items()}
            payload["pose_deltas_se3"]={k:v.detach().cpu() for k, v in self.pose_deltas_se3.items()}

        torch.save(payload, ckpt_path)
        self.to_frontend.put((BackendMessage.CHECKPOINT, str(ckpt_path)))
        

    def run(self):
        done=False 
        while not done:
            if self.global_pause is not None and self.global_pause.is_set():
                self.global_pause.wait()

            msg=self.from_frontend.get()
            match msg:
                case (FrontendMessage.REQUEST_INIT, payload):
                    self.__init_from_payload(payload)
                    print("[Backend] NeRF model initialized, ready for training.")

                case (FrontendMessage.ADD_FRAME, frame):
                    assert isinstance(frame, VideoFrame)
                    self.frames.append(frame)
                    self._register_pose_if_needed(frame)

                    if self.model is not None and len(self.frames)>=2 and self.total_steps<self.conf.warmup_steps:
                        for _ in range(self.conf.steps_per_sync):
                            mse, psnr=self._train_steps()
                            self.total_steps+=1
                        if (self.total_steps % self.conf.preview_every_steps)==0:
                            render_u8, gt_u8=self._preview()
                            warmup_pct = round(100 * self.total_steps / self.conf.warmup_steps) if self.total_steps < self.conf.warmup_steps else None
                            self.to_frontend.put(
                                (
                                    BackendMessage.SYNC,
                                    {
                                        "loss": mse, "psnr": psnr,
                                        "render_rgb_u8": render_u8, "gt_rgb_u8": gt_u8,
                                        "total_steps": self.total_steps,
                                        "warmup_pct": warmup_pct,
                                        "n_keyframes": len(self.keyframes),
                                    },
                                )
                            )

                case (FrontendMessage.ADD_KEYFRAME, frame):
                    assert isinstance(frame, VideoFrame)
                    self.keyframes.append(frame)
                    self.frames.append(frame)
                    self._register_pose_if_needed(frame)
                    if self.model is not None and len(self.frames) >= 2:
                        for _ in range(self.conf.steps_per_keyframe):
                            mse, psnr = self._train_steps()
                            self.total_steps += 1
                        render_u8, gt_u8 = self._preview()
                        warmup_pct = round(100 * self.total_steps / self.conf.warmup_steps) if self.total_steps < self.conf.warmup_steps else None
                        self.to_frontend.put(
                            (
                                BackendMessage.SYNC,
                                {
                                    "loss": mse, "psnr": psnr,
                                    "render_rgb_u8": render_u8, "gt_rgb_u8": gt_u8,
                                    "total_steps": self.total_steps,
                                    "warmup_pct": warmup_pct,
                                    "n_keyframes": len(self.keyframes),
                                },
                            )
                        )
                        if (self.total_steps % (self.conf.preview_every_steps * 5)) == 0:
                            self._checkpoint()

                case (FrontendMessage.END, _):
                    done=True
                case _:
                    pass 



        if self.model is not None:
            final_path=self.conf.output_dir/"nerf_final.pt"
            payload = {"model": self.model.state_dict()}
            if self.conf.enable_pose_refinement:
                payload["base_poses_c2w"] = {k: v.detach().cpu() for k, v in self.base_poses_c2w.items()}
                payload["pose_deltas_se3"] = {k: v.detach().cpu() for k, v in self.pose_deltas_se3.items()}
            torch.save(payload, final_path)

            self.to_frontend.put((BackendMessage.CHECKPOINT, str(final_path)))
        self.to_frontend.put((BackendMessage.COMPLETED, None))
        self.to_backend_done.set()





























        
    
