import cv2

def draw_matches(img1, kp1, img2, kp2, matches):
    matched_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_image