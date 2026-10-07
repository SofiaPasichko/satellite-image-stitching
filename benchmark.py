import time

from src.preprocessing import *
from src.orb import *
from src.sift import sift_features
from src.visualization import *
from src.matching import bf_knn_match, lowe_ratio_test, apply_ransac
from src.stitching import stitch_images

img1 = cv2.imread("./data/1.jpg")
img2 = cv2.imread("./data/4.jpg")

img1gray = to_grayscale(img1)
img2gray = to_grayscale(img2)

# OBR
start = time.perf_counter()

keypoints1, descriptors1 = orb_features(img1gray)
keypoints2, descriptors2 = orb_features(img2gray)

matches = bf_knn_match(descriptors1, descriptors2, cv2.NORM_HAMMING)
good_matches = lowe_ratio_test(matches)
inlier_matches, H = apply_ransac(keypoints1, keypoints2, good_matches)

end = time.perf_counter()

result = draw_matches(img1, keypoints1, img2, keypoints2, inlier_matches)
result = cv2.resize(result, None, fx=0.4, fy=0.4)

cv2.imwrite("./results/benchmark/orb_matches.jpg", result)

print("ORB")
print(f"Keypoints image 1: {len(keypoints1)}")
print(f"Keypoints image 2: {len(keypoints2)}")
print(f"Raw matches: {len(matches)}")
print(f"Good matches: {len(good_matches)}")
print(f"Inliers (RANSAC): {len(inlier_matches)}")
print(f"Inlier ratio: {len(inlier_matches) / len(good_matches):.2%}")
print(f"Execution time: {(end - start):.4f} s")

# SIFT
start = time.perf_counter()

keypoints1, descriptors1 = sift_features(img1gray)
keypoints2, descriptors2 = sift_features(img2gray)

matches = bf_knn_match(descriptors1, descriptors2, cv2.NORM_L2)
good_matches = lowe_ratio_test(matches)
inlier_matches, H = apply_ransac(keypoints1, keypoints2, good_matches)

end = time.perf_counter()

result = draw_matches(img1, keypoints1, img2, keypoints2, inlier_matches)
result = cv2.resize(result, None, fx=0.4, fy=0.4)

cv2.imwrite("./results/benchmark/sift_matches.jpg", result)

print("\nSIFT")
print(f"Keypoints image 1: {len(keypoints1)}")
print(f"Keypoints image 2: {len(keypoints2)}")
print(f"Raw matches: {len(matches)}")
print(f"Good matches: {len(good_matches)}")
print(f"Inliers (RANSAC): {len(inlier_matches)}")
print(f"Inlier ratio: {len(inlier_matches) / len(good_matches):.2%}")
print(f"Execution time: {(end - start):.4f} s")

panorama = stitch_images(img1, img2, H)
panorama = cv2.resize(panorama, None, fx=0.4, fy=0.4)

cv2.imwrite("./results/benchmark/panorama.jpg", panorama)