#!/usr/bin/env python

import cv2
import numpy as np
from utils import mouse_handler
from utils import get_four_points
import sys


cap = cv2.VideoCapture('2.mp4')
MIN_MATCH_COUNT = 10
#img1 = cv2.imread('b.png', cv2.IMREAD_UNCHANGED)
img1 = cv2.imread('a.png', 0) # Query picture
img2 = cv2.imread('d.png', 0) # training picture
sift = cv2.xfeatures2d.SIFT_create()

# Use SIFT to find key points and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)

# Read background image
background = cv2.imread("a.png")
dimensions = background.shape
height = background.shape[0]
width = background.shape[1]

#foreGroundImage = cv2.imread("ban.png", -1)
fore = cv2.imread("oh.png", -1)
foreGroundImage = cv2.resize(fore, (width, height))


# Split png foreground image
b, g, r, a = cv2.split(foreGroundImage)

# Save the foregroung RGB content into a single object
foreground = cv2.merge((b, g, r))

# Save the alpha information into a single Mat
alpha = cv2.merge((a, a, a))


# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)
alpha = alpha.astype(float) / 255

# Perform alpha blending
foreground = cv2.multiply(alpha, foreground)
background = cv2.multiply(1.0 - alpha, background)
outImage = cv2.add(foreground, background)

cv2.imwrite("f.png", outImage)

# Loop until the end of the video
while (cap.isOpened()):
	ret, frame = cap.read()
	# Display image.
	kp2, des2 = sift.detectAndCompute(frame, None)
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	if (des1 is not None and len(des1) > 2 and des2 is not None and len(des2) > 2):
		matches = flann.knnMatch(des1, des2, k=2)
	#matches = flann.knnMatch(des1, des2, k=2)

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
			#print(M)
			#print(pts)
			if (M is not None and pts is not None):
				dst = cv2.perspectiveTransform(pts, M)
				print(dst)
				pts_dst = dst
				# cv2.imshow("outImg", outImage/255)
				# cv2.waitKey(0)
				im_src = cv2.imread('f.png');
				size = im_src.shape

				# Create a vector of source points.
				pts_src = np.array(
					[
						[0, 0],
						[size[1] - 1, 0],
						[size[1] - 1, size[0] - 1],
						[0, size[0] - 1]
					], dtype=float
				);

				# Read destination image
				#im_dst = cv2.imread('d.png');
				im_dst = frame;
				# Get four corners of the billboard

				# pts_dst = dst

				print(pts_dst)
				pts_dst = pts_dst[[0, 3, 2, 1]]
				print(pts_dst)
				# Calculate Homography between source and destination points
				h, status = cv2.findHomography(pts_src, pts_dst);

				# Warp source image
				im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

				# Black out polygonal area in destination image.
				cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);

				# Add warped source image to destination image.
				im_dst = im_dst + im_temp;
				cv2.imshow("Image", im_dst);
				#cv2.waitKey(0);
			else:
				cv2.imshow("Image", frame);

		else:
			print("Not enough matches are found", (len(good), MIN_MATCH_COUNT))
			cv2.imshow("Image", frame);
			matchesMask = None

	# Display image.
	#cv2.imshow("Image", im_dst);
	#cv2.waitKey(0);







	#cv2.imshow("Image", frame);
	#cv2.waitKey(1);
	
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
