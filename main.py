from pathlib import Path
import cv2

from src.sift import create_sift_detector
from src.matching import create_bf_matcher
from src.stitching import build_panorama

def main():
    detector = create_sift_detector()
    matcher = create_bf_matcher(cv2.NORM_L2)

    images = [cv2.imread(str(p)) for p in sorted(Path("./data").glob("*.jpg"))]

    if len(images) < 2:
        print("Need at least two images.")
        return

    panorama = build_panorama(
        images,
        detector,
        matcher
    )

    if panorama is None:
        print("Panorama construction failed.")
        return

    scale = 0.5

    panorama = cv2.resize(
        panorama,
        None,
        fx=scale,
        fy=scale
    )

    cv2.imwrite("./results/main/panorama.jpg", panorama)

if __name__ == "__main__":
    main()