# Bridging Classical Geometric SLAM using Depth-Supervised NeRF

This is a research-oriented framework that combines traditional Monocular Simultaneous Localization and Mapping (SLAM) with Neural Radiance Fields (NeRF). This project focuses on achieving robust camera tracking through epipolar geometry and Perspective-n-Point (PnP) constraints while simultaneously generating high-fidelity 3D volumetric reconstructions.


## Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/slam-nerf.git](https://github.com/yourusername/slam-nerf.git)
cd slam-nerf

# Install required dependencies
pip install -r requirements.txt
```
Running the Frontend
To view the real-time feature tracking and point cloud visualization:

```bash
python -m frontend --video test_nyc.mp4
```
Running the Pipeline
To execute the full SLAM-NeRF optimization and reconstruction process:

```bash
python -m nerfslam.pipeline --video test_nyc.mp4
```
