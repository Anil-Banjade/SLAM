from __future__ import annotations 

import torch 

def _skew(w):
    wx,wy,wz=w.unbind(dim=-1)
    O=torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([0, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )

def so3_exp(w, eps):
    theta=torch.linalg.norm(w, dim=-1, keepdim=True)
    W=_skew(w)
    theta2=theta*theta
    A=torch.where(theta>eps, torch.sin(theta)/theta, 1.0-theta2/6.0)
    B=torch.where(theta>eps, (1.0-torch.cos(theta))/theta2, 0.5-theta2/24.0)

    I=torch.eye(3, device=w.device, dtype=w.dtype).expand(W.shape)
    R=I+A[..., None] * W + B[..., None]*(W@W)

    return R

def se3_exp(xi, eps):
    w=xi[...,:3]
    v=xi[...,3:]

    theta=torch.linalg.norm(w, dim=-1, keepdim=True)
    W=_skew(w)
    R=so3_exp(w, eps=eps)

    theta2=theta*theta
    theta3=theta2*theta

    A = torch.where(theta > eps, (1.0 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24.0)
    B = torch.where(theta > eps, (theta - torch.sin(theta)) / theta3, (1.0 / 6.0) - theta2 / 120.0)

    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(W.shape)
    V = I + A[..., None] * W + B[..., None] * (W @ W)

    t = (V @ v[..., None]).squeeze(-1)

    T = torch.eye(4, device=xi.device, dtype=xi.dtype).expand((*xi.shape[:-1], 4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


