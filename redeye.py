import cv2
import numpy as np
import argparse
import sys
from skimage import measure


import matplotlib.pyplot as plt
#https://github.com/Kundru69/DrowsinessAlert.git

s=0

def mse(imageA, imageB):
	
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	
	return err
 
def compare_images(imageA, imageB, title):
	
	global s
	m = mse(imageA, imageB)
	s = measure.compare_ssim(imageA, imageB)

	if s==1:
		print("No red detected")
		print("Driver in perfect condition")

	else:
		print("Red detected")
		print("Drowsiness Alert")
 
	



img=sys.argv[1]

frame = cv2.imread(img)

hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])
red_mask = cv2.inRange(hsv_frame, low_red, high_red)


red = cv2.bitwise_and(frame, frame, mask=red_mask)

cv2.imshow("frame",frame)
cv2.waitKey(0)

original = frame.copy()   #cv2.imread("p1.jpg")
contrast = red.copy()     #cv2.imread("p2.jpeg")resized_image = cv2.resize(image, (100, 50))

contrast = cv2.resize(contrast, (768, 1024))


blank_image = np.zeros(shape=[1024, 768], dtype=np.uint8)

original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast)

for (i, (name, image)) in enumerate(images):
	
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	
compare_images(blank_image, contrast, "Original vs. Original")  


if s==1:
	cv2.putText(frame, "**NO Drowsy**", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
else:
	cv2.putText(frame, "***Drowsy****", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


cv2.imshow("F1",frame)
cv2.waitKey(0)
sys.exit()

  
