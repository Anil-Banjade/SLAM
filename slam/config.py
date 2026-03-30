import numpy as np
from utils.display import Display
disp_w = 1400
disp_h = 600
disp = Display(disp_w, disp_h)
K_mobile = np.array([
        [3.14155685e+03, 0.00000000e+00, 2.01474409e+03],
        [0.00000000e+00, 3.13835299e+03, 9.44067665e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
Dist_mobile = np.array([0.19056529, -0.82832292, 0.00305567, 0.0017555, 0.24343703])

K_mobile_video = np.array([
    [1.94141391e+03, 0.00000000e+00, 1.88812703e+03],
    [0.00000000e+00, 1.94472299e+03, 8.23596947e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

Dist_mobile_video = np.array([0.01991167, -0.15226447, -0.00020439, 0.0014133, 0.19155372])

K_test_nyc = np.array([
    [2800.0,    0.0, 1920.0],
    [   0.0, 2800.0,  822.0],
    [   0.0,    0.0,    1.0]
])

K_test_nyc_cropped = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 900.0, 160.0],
    [  0.0,   0.0,   1.0]
])


Dist_test_nyc = np.array([
    -0.05,  # k1
     0.01,  # k2
     0.0,   # p1
     0.0,   # p2
     0.0    # k3
])

K = K_mobile_video
dist_coeffs = Dist_mobile_video

# K = K_test_nyc_cropped
# dist_coeffs = Dist_test_nyc
args = None
