import numpy as np
import cv2
import pyceres

class Observations:
    def __init__(self):
        self.frames = []
        self.map_points = []
        self.pixel_points = []

class ReprojectionCost(pyceres.CostFunction):

    def __init__(self, observed_px, K):
        pyceres.CostFunction.__init__(self)
        self.observed = observed_px   # (2,)
        self.K = K
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])  # pose(6), point(3)

    def Evaluate(self, parameters, residuals, jacobians):
        pose = parameters[0]   # [rvec(3), tvec(3)]
        point = parameters[1]  # [X, Y, Z]

        rvec = pose[:3].reshape(3, 1)
        tvec = pose[3:].reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        p_cam = R @ point + tvec.ravel()

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        x, y, z = p_cam
        proj_x = fx * x / z + cx
        proj_y = fy * y / z + cy

        #write into residuals: x,y that pyceres expects
        residuals[0] = proj_x - self.observed[0]
        residuals[1] = proj_y - self.observed[1]

        if jacobians is not None:
            if jacobians[0] is not None:
                # d(residual)/d(pose) — shape (2, 6) stored row-major
                # Jacobian w.r.t. tvec
                J = np.zeros((2, 6))
                zinv = 1.0 / z
                zinv2 = zinv * zinv
                # w.r.t. tvec (last 3)
                J[0, 3] = fx * zinv
                J[0, 5] = -fx * x * zinv2
                J[1, 4] = fy * zinv
                J[1, 5] = -fy * y * zinv2
                # w.r.t. rvec: use numerical diff for now???
                eps = 1e-6
                for k in range(3):
                    rvec_p = rvec.ravel().copy()
                    rvec_p[k] += eps
                    R_p, _ = cv2.Rodrigues(rvec_p)
                    pc_p = R_p @ point + tvec.ravel()
                    J[0, k] = (fx * pc_p[0]/pc_p[2] + cx - (fx * x/z + cx)) / eps
                    J[1, k] = (fy * pc_p[1]/pc_p[2] + cy - (fy * y/z + cy)) / eps
                jacobians[0][:] = J.ravel()

            if jacobians[1] is not None:
                # d(residual)/d(point) — shape (2, 3)
                J = np.zeros((2, 3))
                zinv = 1.0 / z
                zinv2 = zinv * zinv
                J[0, 0] = fx * R[0, 0] * zinv - fx * x * R[2, 0] * zinv2
                J[0, 1] = fx * R[0, 1] * zinv - fx * x * R[2, 1] * zinv2
                J[0, 2] = fx * R[0, 2] * zinv - fx * x * R[2, 2] * zinv2
                J[1, 0] = fy * R[1, 0] * zinv - fy * y * R[2, 0] * zinv2
                J[1, 1] = fy * R[1, 1] * zinv - fy * y * R[2, 1] * zinv2
                J[1, 2] = fy * R[1, 2] * zinv - fy * y * R[2, 2] * zinv2
                jacobians[1][:] = J.ravel()

        return True

class Ceres_solver:
    def __init__(self, K):
        self.K = K
        # self.obs = obs
        self.options = pyceres.SolverOptions()
        self.options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
        self.options.num_threads = 4
        self.options.max_num_iterations = 50
        self.options.minimizer_progress_to_stdout = True
        self.summary = pyceres.SolverSummary()

    def start(self, window):
        obs = Observations()
        for frame in window:
            for kp_idx, mp in frame.observations.items():
                obs.frames.append(frame)
                obs.pixel_points.append(frame.kpx_px[kp_idx])
                obs.map_points.append(mp)
        

        problem = pyceres.Problem()
        pose_params  = {}   # id(frame)    -> np.array (6,)
        point_params = {}   # id(mappoint) -> np.array (3,)
        frame_map    = {}   # id(frame)    -> frame object
        mp_map       = {}   # id(mappoint) -> mappoint object
     
        for f, mp , px in zip(obs.frames, obs.map_points, obs.pixel_points):
            fid = id(f)
            pid = id(mp)

            if fid not in pose_params:
                rvec, _ = cv2.Rodrigues(f.pose[:3, :3])
                tvec = f.pose[:3, 3].copy()
                pose_params[fid] = np.concatenate([rvec.ravel(), tvec]).astype(np.float64)
                frame_map[fid] = f

            if pid not in point_params:
                point_params[pid] = mp.xyz.astype(np.float64).copy()
                mp_map[pid] = mp
            px = px
            cost = ReprojectionCost(px.astype(np.float64), self.K)
            loss = pyceres.HuberLoss(np.sqrt(5.991))
            pose_p  = pose_params[fid]
            point_p = point_params[pid]
            # print(f"pose_p:  shape={pose_p.shape}  dtype={pose_p.dtype}")
            # print(f"point_p: shape={point_p.shape} dtype={point_p.dtype}")
            problem.add_residual_block(cost, loss,
                                    [pose_params[fid], point_params[pid]])

        # fix first frame to remove gauge freedom
        if not obs.frames or obs.frames[0] is None:
            return [], []
            breakpoint()
        first_fid = id(obs.frames[0])
        problem.set_parameter_block_constant(pose_params[first_fid])

        pyceres.solve(self.options, problem, self.summary)
        # print("---------------Ceres Solver---------------")
        # print(self.summary.BriefReport())

        # write results back
        test_frame_poses = []
        test_mp = []
        T = np.eye(4)
        # for fid, param in pose_params.items():
        #     R, _ = cv2.Rodrigues(param[:3])
        #     frame_map[fid].pose[:3, :3] = R
        #     frame_map[fid].pose[:3,  3] = param[3:]
        #     T[:3, :3] = R
        #     T[:3, 3] = param[3:].reshape(3)
            
        #     test_frame_poses.append(T)
        for fid, param in pose_params.items():
            R, _ = cv2.Rodrigues(param[:3])
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = param[3:].reshape(3)
            test_frame_poses.append(T)
            
            if fid == first_fid:   # don't overwrite the anchor frame
                continue
            
            frame_map[fid].pose[:3, :3] = R
            frame_map[fid].pose[:3,  3] = param[3:]
        for pid, param in point_params.items():
            mp_map[pid].xyz = param.astype(np.float32)
            # test_mp.append(param.astype(np.float32))
            test_mp.append(mp_map[pid])

        return test_frame_poses, test_mp

