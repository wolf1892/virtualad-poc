#!/usr/bin/env python

import cv2
import numpy as np
import time

MIN_MATCH_COUNT = 10

cap = cv2.VideoCapture('2.mp4')

img1 = cv2.imread('a.png', 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
    dict(checks=50),
)

background = cv2.imread("a.png")
height, width = background.shape[:2]

fore = cv2.imread("oh.png", -1)
foreGroundImage = cv2.resize(fore, (width, height))

b, g, r, a = cv2.split(foreGroundImage)
foreground = cv2.merge((b, g, r)).astype(float)
alpha = cv2.merge((a, a, a)).astype(float) / 255

replacement_img = cv2.add(
    cv2.multiply(alpha, foreground),
    cv2.multiply(1.0 - alpha, background.astype(float)),
).astype(np.uint8)

h_ref, w_ref = img1.shape
corners_ref = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kp2, des2 = sift.detectAndCompute(frame, None)

    if des1 is not None and des2 is not None and len(des1) > 2 and len(des2) > 2:
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                dst_corners = cv2.perspectiveTransform(corners_ref, M)
                warped = cv2.warpPerspective(replacement_img, M, (frame.shape[1], frame.shape[0]))
                cv2.fillConvexPoly(frame, dst_corners.astype(np.int32), 0, 16)
                frame = frame + warped

    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time) if cur_time != prev_time else 0
    prev_time = cur_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
