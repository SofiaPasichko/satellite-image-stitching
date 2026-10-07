import cv2

def sift_features(img):
    sift = cv2.SIFT_create(nfeatures=1000)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def create_sift_detector():
    return cv2.SIFT_create(nfeatures=1000)