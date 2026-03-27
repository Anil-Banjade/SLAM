import numpy as np
K_mobile = np.array([
        [3.14155685e+03, 0.00000000e+00, 2.01474409e+03],
        [0.00000000e+00, 3.13835299e+03, 9.44067665e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
Dist_mobile = np.array([0.19056529, -0.82832292, 0.00305567, 0.0017555, 0.24343703])

K = K_mobile
dist_coeffs = Dist_mobile
args = None