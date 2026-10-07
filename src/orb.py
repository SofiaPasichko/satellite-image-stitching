import cv2

def orb_features(img):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors