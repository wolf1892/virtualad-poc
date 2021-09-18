# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
MIN_MATCH_COUNT = 5
 
img1 = cv2.imread('a.png', 0) # Query picture
img2 = cv2.imread('d.png', 0) # training picture
 
 # Initialize SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
 
 # Use SIFT to find key points and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
 
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
 
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
 

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
 
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    print(dst)
 
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
 
else:
    print("Not enough matches are found", (len(good), MIN_MATCH_COUNT))
    matchesMask = None
 
 # Finally draw the inner point (if the object is successfully found) or match the key point (if it fails)
 
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   matchesMask=matchesMask,
                   flags=2)
 
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)


cv2.imshow("result", img3)
cv2.waitKey()
 
#plt.imshow(img3, 'gray'), plt.show()
