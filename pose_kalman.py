# pose_kalman.py — constant-velocity Kalman filter on camera center (world frame)
from __future__ import annotations

import numpy as np


class PoseVelocityKalman:
    """
    Linear Kalman filter: state x = [C_x, C_y, C_z, v_x, v_y, v_z]^T
    where C is the camera center in world coordinates (same as pose_c2w[:3, 3]).

    Model: constant velocity — C_{k+1} = C_k + dt * v_k, v_{k+1} = v_k + noise.

    Rotation is not in the state; it comes from the visual front-end (PnP / epipolar).
    The filter smooths translation (reduces trajectory overshoot / outlier jumps) and
    supplies a predicted camera center for the next frame's PnP initial guess.
    """

    def __init__(
        self,
        dt: float,
        sigma_accel: float = 2.0,
        sigma_meas: float = 0.035,
        mahalanobis_gate_sq: float = 12.0,
    ):
        self.dt = float(dt)
        self.sigma_accel = float(sigma_accel)
        self.sigma_meas = float(sigma_meas)
        self.mahalanobis_gate_sq = float(mahalanobis_gate_sq)

        dt = self.dt
        self.F = np.eye(6, dtype=np.float64)
        self.F[:3, 3:6] = np.eye(3) * dt

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[:3, :3] = np.eye(3)

        # Continuous white noise acceleration on velocity; discretized Q (block 3x3 structure)
        q = self.sigma_accel**2
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        q11 = 0.25 * dt4 * q
        q12 = 0.5 * dt3 * q
        q22 = dt2 * q
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.Q[:3, :3] = np.eye(3) * q11
        self.Q[:3, 3:6] = np.eye(3) * q12
        self.Q[3:6, :3] = np.eye(3) * q12
        self.Q[3:6, 3:6] = np.eye(3) * q22

        self.R = np.eye(3, dtype=np.float64) * (self.sigma_meas**2)

        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self._bootstrap_frames = 0
        self._pnp_mode = False

    def set_dt(self, dt: float) -> None:
        """Recompute F and Q when frame interval changes (e.g. from video FPS)."""
        self.dt = float(dt)
        dt = self.dt
        self.F = np.eye(6, dtype=np.float64)
        self.F[:3, 3:6] = np.eye(3) * dt
        q = self.sigma_accel**2
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        q11 = 0.25 * dt4 * q
        q12 = 0.5 * dt3 * q
        q22 = dt2 * q
        self.Q = np.zeros((6, 6), dtype=np.float64)
        self.Q[:3, :3] = np.eye(3) * q11
        self.Q[:3, 3:6] = np.eye(3) * q12
        self.Q[3:6, :3] = np.eye(3) * q12
        self.Q[3:6, 3:6] = np.eye(3) * q22

    @property
    def ready_for_prediction(self) -> bool:
        return self._pnp_mode and self._bootstrap_frames > 0

    def init_first(self, pose_c2w: np.ndarray) -> None:
        self.x[:3] = pose_c2w[:3, 3].astype(np.float64)
        self.x[3:6] = 0.0
        self.P = np.eye(6, dtype=np.float64) * 10.0
        self._bootstrap_frames = 0
        self._pnp_mode = False

    def bootstrap(self, pose_prev_c2w: np.ndarray, pose_curr_c2w: np.ndarray) -> None:
        """During monocular bootstrap: estimate velocity from finite differences."""
        t_prev = pose_prev_c2w[:3, 3].astype(np.float64)
        t_curr = pose_curr_c2w[:3, 3].astype(np.float64)
        self.x[:3] = t_curr
        if self._bootstrap_frames > 0:
            self.x[3:6] = (t_curr - t_prev) / max(self.dt, 1e-9)
        else:
            self.x[3:6] = 0.0
        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 0.5
        self.P[3:6, 3:6] *= 2.0
        self._bootstrap_frames += 1

    def enter_pnp_mode(self) -> None:
        self._pnp_mode = True

    def seed_from_prev_if_needed(self, pose_prev_c2w: np.ndarray) -> None:
        """If bootstrap never ran (unusual configs), align state with the previous camera center."""
        if self._bootstrap_frames > 0:
            return
        self.x[:3] = pose_prev_c2w[:3, 3].astype(np.float64)
        self.x[3:6] = 0.0
        self._bootstrap_frames = 1

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def predicted_position(self) -> np.ndarray:
        """Use before predict() if you need z_pred = H @ x without advancing state."""
        return (self.H @ self.x).reshape(3)

    def innovation_stats(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        z_pred = (self.H @ self.x).reshape(3)
        innov = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        try:
            d2 = float(innov.T @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            d2 = float("inf")
        return innov, d2

    def gate_measurement(self, z: np.ndarray) -> bool:
        _, d2 = self.innovation_stats(z)
        return np.isfinite(d2) and d2 <= self.mahalanobis_gate_sq

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        z_pred = self.H @ self.x
        innov = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    def update_relaxed(self, z: np.ndarray, meas_noise_scale: float = 3.0) -> None:
        """One update step with larger measurement noise (weaker trust), e.g. for epipolar fallback."""
        R_save = self.R.copy()
        self.R = self.R * (float(meas_noise_scale) ** 2)
        try:
            self.update(z)
        finally:
            self.R = R_save

    def fused_pose_c2w(self, pose_meas_c2w: np.ndarray) -> np.ndarray:
        """Build c2w: filtered translation, measured rotation."""
        out = np.asarray(pose_meas_c2w, dtype=np.float64).copy()
        out[:3, 3] = self.x[:3]
        return out


def anchor_translation_to_measurement(
    t_filt: np.ndarray,
    t_meas: np.ndarray,
    t_prev: np.ndarray,
    ratio: float,
    floor: float,
) -> np.ndarray:
    """
    Pull the filtered camera center back toward the visual measurement if the Kalman
    update drifted too far. Cap distance ||t_filt - t_meas|| by
    max(floor, ratio * ||t_meas - t_prev||) so the trajectory stays tied to feature geometry.
    """
    t_filt = np.asarray(t_filt, dtype=np.float64).reshape(3)
    t_meas = np.asarray(t_meas, dtype=np.float64).reshape(3)
    t_prev = np.asarray(t_prev, dtype=np.float64).reshape(3)
    if ratio <= 0.0:
        return t_filt
    motion = np.linalg.norm(t_meas - t_prev)
    cap = max(float(floor), float(ratio) * motion) if motion > 1e-9 else float(floor)
    delta = t_filt - t_meas
    d = float(np.linalg.norm(delta))
    if d <= cap + 1e-12:
        return t_filt
    return t_meas + (cap / d) * delta
