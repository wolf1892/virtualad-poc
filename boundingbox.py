# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
img = cv2.imread('b.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
background = cv2.imread('d.png')
background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)

rows,cols,ch = background.shape

pts1 = np.float32([[0,0],[264,0],[0,64],[264,64]]) 
#pts2 = np.float32([[610.23157,543.57886],[913.0846,617.9276],[610.46027,618.0193],[913.0431,543.3649]]) 

pts2 = np.float32([[610.23157,543.57886],[913.0431,543.3649],[610.46027,618.0193],[913.0846,617.9276]]) 

M = cv2.getPerspectiveTransform(pts1,pts2)    
dst = cv2.warpPerspective(img,M,(cols,rows))

overlay = cv2.add(background, dst)

cv2.imshow("result", overlay)
cv2.waitKey()
 

