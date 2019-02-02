import cv2
import numpy as np
import matplotlib.pyplot as plt

imgPath = ""

img = cv2.imread(imgPath)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(blur,50,150)

height = img.shape[0]
polygons = np.array([[]])
mask = np.zeros_like(img)
cv2.fillPoly(mask,polygons,255)
maskedImg = cv2.bitwise_and(canny,mask)

lines = cv2.HoughLinesP(maskedImg,2,np.pi/180,100,np.array([]),minLineLength=50,maxLineGap=5)
linesImg = np.zeros_like(img)
if lines is not None:
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		cv2.line(linesImg, (x1,y1), (x2,y2), (255,0,0), 10)
overlayImg = cv2.addWeighted(img, 0.8, linesImg, 1, 1)

plt.imshow(overlayImg)
plt.show()