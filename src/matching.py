import cv2
import numpy as np

def create_bf_matcher(norm):
    return cv2.BFMatcher(norm)

def bf_knn_match(des1, des2, norm, k=2):
    bf = create_bf_matcher(norm)
    return bf.knnMatch(des1, des2, k)

def lowe_ratio_test(matches, ratio=0.8):
    good_matches = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

def apply_ransac(keypoints1, keypoints2, matches):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    inlier_matches = []

    for match, keep in zip(matches, mask.ravel()):
        if keep:
            inlier_matches.append(match)

    return inlier_matches, H