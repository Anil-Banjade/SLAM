import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_matches(img1, img2, pts1, pts2, color=(0, 255, 0), radius=5, thickness=2):
    """
    Draw matches between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image (H x W x 3 or H x W).
    img2 : np.ndarray
        Second image (H x W x 3 or H x W).
    pts1 : list or np.ndarray
        Points in first image, shape (N,2)
    pts2 : list or np.ndarray
        Points in second image, shape (N,2)
    color : tuple
        Line and circle color (B,G,R)
    radius : int
        Circle radius
    thickness : int
        Line thickness

    Returns
    -------
    matched_img : np.ndarray
        Image showing both frames side by side with lines connecting points.
    """
    # Make sure images are color
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Stack images horizontally
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h = max(h1, h2)
    matched_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    matched_img[:h1, :w1] = img1
    matched_img[:h2, w1:w1 + w2] = img2

    # Draw points and lines
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        x2_shifted = int(x2 + w1)  # shift x2 for the second image
        y1 = int(y1)
        x1 = int(x1)
        y2 = int(y2)

        # Draw circles
        cv2.circle(matched_img, (x1, y1), radius, color, -1)
        cv2.circle(matched_img, (x2_shifted, y2), radius, color, -1)

        # Draw line connecting points
        cv2.line(matched_img, (x1, y1), (x2_shifted, y2), color, thickness)
    
    plt.figure()
    plt.imshow(matched_img)
    plt.title(f"Feature Matching from plot_utils")
    plt.show(block=False)
    input()
    plt.close()
    
    return matched_img



def draw_orb_keypoints(image, points, color=(0, 255, 255), radius=5, thickness=3):
    output = image.copy()

    points = np.asarray(points)

    # Ensure shape is (N, 2)
    points = points.reshape(-1, 2)

    for pt in points:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))

        cv2.circle(output, (x, y), radius, color, thickness)

    plt.figure()
    plt.imshow(output)
    plt.title(f"Features that correspond to 3d Map")
    plt.show(block=False)
    input()
    plt.close()