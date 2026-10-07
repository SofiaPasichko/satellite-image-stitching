from src.matching import *

def stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    corners_img2 = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    all_corners = np.concatenate(
        (warped_corners_img1, corners_img2),
        axis=0
    )

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    translation = [-xmin, -ymin]

    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    panorama = cv2.warpPerspective(
        img1,
        T @ H,
        (xmax - xmin, ymax - ymin)
    )

    panorama[
        translation[1]:translation[1] + h2,
        translation[0]:translation[0] + w2
    ] = img2

    return panorama

def stitch_pair(img1, img2, detector, matcher):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return None

    matches = matcher.knnMatch(des1, des2, k=2)

    good = lowe_ratio_test(matches)

    if len(good) < 20:
        return None

    inliers, H = apply_ransac(kp1, kp2, good)

    if H is None:
        return None

    return stitch_images(img1, img2, H)

def build_panorama(images, detector, matcher):

    panorama = images[0]

    for i in range(1, len(images)):

        print(f"Stitching image {i+1}/{len(images)}")

        result = stitch_pair(
            panorama,
            images[i],
            detector,
            matcher
        )

        if result is None:
            print(f"Cannot stitch image {i}")
            continue

        panorama = result

    return panorama