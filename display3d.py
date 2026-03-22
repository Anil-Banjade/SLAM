import glfw
from OpenGL.GL import *
import numpy as np
import threading
import time

class Display3d:
    def __init__(self, W=1024, H=768):
        # Initialize GLFW in main thread (this is safe)
        if not glfw.init():
            raise Exception("GLFW failed to init")

        self.W = W
        self.H = H
        self.window = None
        
        self.points = np.zeros((0, 3))
        self.poses = np.zeros((0, 4, 4))
        
        # Lock for thread-safe updates
        self.lock = threading.Lock()
        self.window_ready = threading.Event()
        
        # Camera controls (ORB-SLAM3 style)
        self.auto_follow = True  # Auto-follow the trajectory
        self.view_mode = "follow"  # "follow", "free", "top", "side"
        self.camera_distance = 15.0
        self.camera_angle_x = 45.0  # Elevation angle
        self.camera_angle_y = 0.0   # Azimuth angle
        self.follow_distance = 20.0  # Distance behind camera in follow mode

        # start thread - window creation happens in thread
        self.running = True
        self.th = threading.Thread(target=self.render_loop, daemon=True)
        self.th.start()
        
        # Wait for window to be created
        self.window_ready.wait(timeout=5.0)
        if self.window is None:
            raise Exception("Window creation failed in thread")

    def update_map(self, points, poses):
        with self.lock:
            # Ensure points are 2D array with shape (N, 3)
            if len(points) > 0:
                points_arr = np.array(points).reshape(-1, 3)
                # Filter out invalid points (NaN, Inf) and outliers
                valid_mask = np.isfinite(points_arr).all(axis=1)
                # Filter extreme outliers (beyond reasonable range)
                dist_mask = np.linalg.norm(points_arr, axis=1) < 500
                self.points = points_arr[valid_mask & dist_mask]
            else:
                self.points = np.zeros((0, 3))
            
            # Ensure poses are 3D array with shape (N, 4, 4)
            if len(poses) > 0:
                poses_arr = np.array(poses).reshape(-1, 4, 4)
                # Filter out invalid poses
                valid_poses = []
                for pose in poses_arr:
                    if pose.shape == (4, 4) and np.isfinite(pose).all():
                        cam_pos = pose[:3, 3]
                        # Filter extreme outliers
                        if np.linalg.norm(cam_pos) < 500:
                            valid_poses.append(pose)
                if len(valid_poses) > 0:
                    self.poses = np.array(valid_poses)
                else:
                    self.poses = np.zeros((0, 4, 4))
            else:
                self.poses = np.zeros((0, 4, 4))

    def filter_points_for_display(self, points, latest_cam, scene_scale=10.0, max_points=5000):
        """Filter and sample points to show scene structure (road, buildings, etc.)"""
        if len(points) == 0:
            return points
        
        # Filter points near the trajectory (road area)
        if len(self.poses) > 0:
            trajectory_points = np.array([pose[:3, 3] for pose in self.poses if pose.shape == (4, 4)])
            if len(trajectory_points) > 0:
                # Subsample trajectory so pairwise distance checks stay cheap
                if len(trajectory_points) > 300:
                    step = max(1, len(trajectory_points) // 300)
                    trajectory_points = trajectory_points[::step]
                # Proximity in the same arbitrary units as the SLAM map; scale with scene size
                # so narrow monocular reconstructions still show structure around the path.
                proximity = max(30.0, 0.14 * float(scene_scale))
                try:
                    from scipy.spatial.distance import cdist

                    distances = cdist(points, trajectory_points)
                    min_distances = np.min(distances, axis=1)
                except Exception:
                    # Pure numpy fallback (no SciPy): chunk over map points
                    tp = trajectory_points.astype(np.float64)
                    pts = points.astype(np.float64)
                    min_distances = np.empty(len(pts), dtype=np.float64)
                    chunk = 2000
                    for s in range(0, len(pts), chunk):
                        e = min(s + chunk, len(pts))
                        diff = pts[s:e, None, :] - tp[None, :, :]
                        min_distances[s:e] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
                near_trajectory = min_distances < proximity
                points = points[near_trajectory]
        
        # If still too many points, randomly sample
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        return points

    def compute_bounds(self, points, poses):
        """Compute bounding box and center of all data - ORB-SLAM3 style"""
        all_coords = []
        
        # Add all point coordinates
        if len(points) > 0:
            all_coords.append(points)
        
        # Add all camera positions
        if len(poses) > 0:
            cam_positions = []
            for pose in poses:
                if pose.shape == (4, 4):
                    cam = pose[:3, 3]
                    if np.isfinite(cam).all():
                        cam_positions.append(cam)
            if len(cam_positions) > 0:
                all_coords.append(np.array(cam_positions))
        
        if len(all_coords) == 0:
            return np.array([0.0, 0.0, 0.0]), 10.0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
        
        all_coords = np.vstack(all_coords)
        
        # Compute bounding box
        min_coords = np.min(all_coords, axis=0)
        max_coords = np.max(all_coords, axis=0)
        center = (min_coords + max_coords) / 2.0
        
        # Compute scale based on bounding box diagonal
        bbox_size = max_coords - min_coords
        scale = np.max(bbox_size) * 0.7  # 70% of max dimension for padding
        
        # Get latest camera position and direction for following
        latest_cam = np.array([0.0, 0.0, 0.0])
        latest_dir = np.array([0.0, 0.0, -1.0])
        if len(poses) > 0:
            latest_pose = poses[-1]
            if latest_pose.shape == (4, 4):
                latest_cam = latest_pose[:3, 3]
                # Camera forward direction (negative Z in camera frame)
                latest_dir = -latest_pose[:3, 2]
                latest_dir = latest_dir / (np.linalg.norm(latest_dir) + 1e-6)
        
        # Ensure minimum scale
        scale = max(scale, 5.0)
        
        return center, scale, latest_cam, latest_dir

    def draw_camera_frustum(self, pose, scale=1.0):
        """Draw a camera frustum at the pose position"""
        if pose.shape != (4, 4):
            return
        
        # Camera frustum parameters (smaller, more visible)
        f = 2.0 * scale  # Frustum depth
        w = f * 0.3
        h = f * 0.3
        
        # Frustum corners in camera frame
        corners_cam = np.array([
            [0, 0, 0],      # Camera center
            [-w, -h, f],   # Bottom-left
            [w, -h, f],    # Bottom-right
            [w, h, f],     # Top-right
            [-w, h, f]     # Top-left
        ])
        
        # Transform to world frame
        R = pose[:3, :3]
        t = pose[:3, 3]
        corners_world = (R @ corners_cam.T).T + t
        
        # Draw frustum wireframe
        glLineWidth(1.5)
        glBegin(GL_LINES)
        # From center to corners
        center = corners_world[0]
        for i in range(1, 5):
            glVertex3f(center[0], center[1], center[2])
            glVertex3f(corners_world[i][0], corners_world[i][1], corners_world[i][2])
        # Frustum edges
        glVertex3f(corners_world[1][0], corners_world[1][1], corners_world[1][2])
        glVertex3f(corners_world[2][0], corners_world[2][1], corners_world[2][2])
        glVertex3f(corners_world[2][0], corners_world[2][1], corners_world[2][2])
        glVertex3f(corners_world[3][0], corners_world[3][1], corners_world[3][2])
        glVertex3f(corners_world[3][0], corners_world[3][1], corners_world[3][2])
        glVertex3f(corners_world[4][0], corners_world[4][1], corners_world[4][2])
        glVertex3f(corners_world[4][0], corners_world[4][1], corners_world[4][2])
        glVertex3f(corners_world[1][0], corners_world[1][1], corners_world[1][2])
        glEnd()
        glLineWidth(2.0)

    def render_loop(self):
        # Create window and context in THIS thread (critical!)
        self.window = glfw.create_window(self.W, self.H, "SLAM Viewer (ORB-SLAM3 Style)", None, None)
        if not self.window:
            glfw.terminate()
            return
        
        # Make context current in the rendering thread
        glfw.make_context_current(self.window)
        
        # Set up key callback for interactivity
        def key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS or action == glfw.REPEAT:
                if key == glfw.KEY_UP:
                    self.camera_angle_x = min(90, self.camera_angle_x + 5)
                elif key == glfw.KEY_DOWN:
                    self.camera_angle_x = max(-90, self.camera_angle_x - 5)
                elif key == glfw.KEY_LEFT:
                    self.camera_angle_y -= 5
                elif key == glfw.KEY_RIGHT:
                    self.camera_angle_y += 5
                elif key == glfw.KEY_PAGE_UP:
                    self.camera_distance = max(5, self.camera_distance * 0.9)
                elif key == glfw.KEY_PAGE_DOWN:
                    self.camera_distance = min(200, self.camera_distance * 1.1)
                elif key == glfw.KEY_R:
                    # Reset view
                    self.camera_distance = 15.0
                    self.camera_angle_x = 45.0
                    self.camera_angle_y = 0.0
                    self.view_mode = "follow"
                elif key == glfw.KEY_F:
                    # Toggle auto-follow
                    self.auto_follow = not self.auto_follow
                elif key == glfw.KEY_T:
                    # Top view
                    self.view_mode = "top"
                elif key == glfw.KEY_C:
                    # Follow view (third person)
                    self.view_mode = "follow"
                    self.camera_angle_x = 45.0
                elif key == glfw.KEY_S:
                    # Side view
                    self.view_mode = "side"
                elif key == glfw.KEY_V:
                    # Free view
                    self.view_mode = "free"
        
        glfw.set_key_callback(self.window, key_callback)
        
        # Signal that window is ready
        self.window_ready.set()
        
        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        glPointSize(1.5)  # Smaller points for better scene visibility
        glLineWidth(2.5)
        glClearColor(0.0, 0.0, 0.0, 1)  # Black background like ORB-SLAM3
        
        # Set up projection matrix - use perspective with wide FOV
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.W / self.H
        # Wider field of view, larger far plane
        glFrustum(-aspect * 0.4, aspect * 0.4, -0.4, 0.4, 0.5, 2000)
        
        glMatrixMode(GL_MODELVIEW)
        
        frame_count = 0
        
        while not glfw.window_should_close(self.window) and self.running:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Get current data (thread-safe)
            with self.lock:
                points = self.points.copy()
                poses = self.poses.copy()
            
            # Compute bounds and latest camera position
            center, scale, latest_cam, latest_dir = self.compute_bounds(points, poses)
            
            # Filter points for better visualization
            display_points = self.filter_points_for_display(points, latest_cam, scene_scale=scale)
            
            # Debug output every 300 frames
            frame_count += 1
            if frame_count % 300 == 0:
                print(f"Map: {len(display_points)}/{len(points)} points, {len(poses)} poses | Scale: {scale:.1f} | Mode: {self.view_mode}")
            
            glLoadIdentity()
            
            # Set up view transformation based on mode
            if self.view_mode == "top":
                # Top-down view - looking straight down
                glTranslatef(0, 0, -scale * 1.5)
                glRotatef(90, 1, 0, 0)
                if self.auto_follow and len(poses) > 0:
                    glTranslatef(-latest_cam[0], -latest_cam[1], -latest_cam[2])
                else:
                    glTranslatef(-center[0], -center[1], -center[2])
                    
            elif self.view_mode == "side":
                # Side view - from the side of trajectory
                glTranslatef(0, 0, -scale * 1.2)
                glRotatef(0, 1, 0, 0)  # Horizontal view
                glRotatef(self.camera_angle_y, 0, 1, 0)
                if self.auto_follow and len(poses) > 0:
                    # Position to the side of latest camera
                    side_offset = np.array([-latest_dir[2], 0, latest_dir[0]]) * scale * 0.3
                    view_pos = latest_cam + side_offset
                    glTranslatef(-view_pos[0], -view_pos[1], -view_pos[2])
                else:
                    glTranslatef(-center[0], -center[1], -center[2])
                    
            elif self.view_mode == "follow":
                # Follow view - third person behind camera
                if len(poses) > 0 and self.auto_follow:
                    # Position camera behind the latest camera position
                    follow_pos = latest_cam - latest_dir * self.follow_distance
                    follow_pos[1] += 5.0  # Slightly above
                    
                    # Look at the latest camera
                    glTranslatef(0, 0, -self.follow_distance)
                    glRotatef(self.camera_angle_x, 1, 0, 0)
                    glRotatef(self.camera_angle_y, 0, 1, 0)
                    glTranslatef(-follow_pos[0], -follow_pos[1], -follow_pos[2])
                else:
                    glTranslatef(0, 0, -self.camera_distance * max(scale * 0.1, 1.0))
                    glRotatef(self.camera_angle_x, 1, 0, 0)
                    glRotatef(self.camera_angle_y, 0, 1, 0)
                    glTranslatef(-center[0], -center[1], -center[2])
            else:  # free view
                # Free view - user controlled
                glTranslatef(0, 0, -self.camera_distance * max(scale * 0.1, 1.0))
                glRotatef(self.camera_angle_x, 1, 0, 0)
                glRotatef(self.camera_angle_y, 0, 1, 0)
                if self.auto_follow and len(poses) > 0:
                    glTranslatef(-latest_cam[0], -latest_cam[1], -latest_cam[2])
                else:
                    glTranslatef(-center[0], -center[1], -center[2])

            # Draw trajectory path (ORB-SLAM3 style - thick red line showing road)
            if len(poses) > 1:
                glLineWidth(4.0)
                # Draw trajectory as a path
                glColor3f(1.0, 0.0, 0.0)  # Bright red for trajectory
                glBegin(GL_LINE_STRIP)
                for pose in poses:
                    if pose.shape == (4, 4):
                        cam = pose[:3, 3]
                        try:
                            glVertex3f(float(cam[0]), float(cam[1]), float(cam[2]))
                        except (ValueError, OverflowError):
                            pass
                glEnd()
                
                # Draw trajectory as a thicker "road" line
                glLineWidth(6.0)
                glColor3f(0.8, 0.2, 0.2)  # Darker red for road base
                glBegin(GL_LINE_STRIP)
                for pose in poses:
                    if pose.shape == (4, 4):
                        cam = pose[:3, 3]
                        try:
                            # Slightly below to show as road
                            glVertex3f(float(cam[0]), float(cam[1]) - 0.1, float(cam[2]))
                        except (ValueError, OverflowError):
                            pass
                glEnd()
                glLineWidth(2.5)

            # Draw map points (green - showing scene structure)
            if len(display_points) > 0:
                glColor3f(0.0, 1.0, 0.0)  # Bright green for map points
                glBegin(GL_POINTS)
                for point in display_points:
                    if len(point) >= 3:
                        try:
                            glVertex3f(float(point[0]), float(point[1]), float(point[2]))
                        except (ValueError, OverflowError):
                            pass
                glEnd()

            # Draw camera positions and frustums
            if len(poses) > 0:
                # Draw all previous camera positions
                glColor3f(1.0, 0.5, 0.0)  # Orange for keyframes
                glPointSize(5.0)
                glBegin(GL_POINTS)
                for pose in poses[:-1]:  # All except last
                    if pose.shape == (4, 4):
                        cam = pose[:3, 3]
                        try:
                            glVertex3f(float(cam[0]), float(cam[1]), float(cam[2]))
                        except (ValueError, OverflowError):
                            pass
                glEnd()
                
                # Draw current camera position (larger, yellow)
                latest_pose = poses[-1]
                if latest_pose.shape == (4, 4):
                    cam = latest_pose[:3, 3]
                    glColor3f(1.0, 1.0, 0.0)  # Yellow for current camera
                    glPointSize(10.0)
                    glBegin(GL_POINTS)
                    try:
                        glVertex3f(float(cam[0]), float(cam[1]), float(cam[2]))
                    except (ValueError, OverflowError):
                        pass
                    glEnd()
                    glPointSize(1.5)
                    
                    # Draw camera frustum for current camera
                    glColor3f(1.0, 0.8, 0.0)  # Orange-yellow
                    self.draw_camera_frustum(latest_pose, 3.0)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def close(self):
        self.running = False
        if self.th.is_alive():
            self.th.join()


