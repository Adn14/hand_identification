# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:46:27 2018

@author: Adn
"""

import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage
from matplotlib import pyplot as plt
import scipy.misc
import csv

name = "percobaan/h41.jpg"
im = cv2.imread("percobaan/h41.jpg")
#cv2.imshow("h", im)

def resize(val):
    img = cv2.imread(val)
    #res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
    r = 256.0 / img.shape[1]
    dim = (256, int(img.shape[0] * r))
 
    # perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow("resized", resized)
    
    return resized

def crop_image(val):
    im1 = val
    
    mask = np.zeros([200,256], dtype=np.uint8)
    mask[64:384,0:256] = 255
    rect = cv2.boundingRect(mask) 
    
    im1 = im1[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]  # crop the image to the desired rectangle
    #cv2.imshow("crop",im1)

    return im1

def hsv(val1):
    
    im1 = val1
    im1Hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",im1Hsv)
    return im1Hsv

def skeleton(val):
    
    im1Hsv = val
    
    lower = np.array([10, 10, 20], dtype = "uint8")
    upper = np.array([30, 255, 255], dtype = "uint8")
    
    skinMask = cv2.inRange(im1Hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    img = skinMask
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
 
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
 
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
 
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
            
    #cv2.imshow("skel", skel)
    return skel
    
def preprocess(val):
    small = resize(val)
    crop = crop_image(small)
    hasil_hsv = hsv(crop)
    sk = skeleton(hasil_hsv)
    print(len(sk))
    
def hsvskin(val):
    
    im1Hsv = val
    
    lower = np.array([0, 10, 60], dtype = "uint8")
    upper = np.array([20, 150, 255], dtype = "uint8")
    
    skinMask = cv2.inRange(im1Hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))#whattochange
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    
    img = skinMask
    
    
    
    
    scipy.misc.imsave('percobaan/testData/predict.jpg', img)
    
    img = cv2.imread('percobaan/testData/predict.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
            # Convert to binary image
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # noise removal
    # to remove any small white noises use morphological opening
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)


    img = cv2.bitwise_not(sure_bg)
    scipy.misc.imsave('percobaan/testData/predict.jpg', img)
    return img
    
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(cX-20)

   
 
	# draw the countour number on the image
    #cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
	#	1.0, (255, 0 , 0), 2)
    
    mask = np.zeros([1280,720], dtype=np.uint8)
    mask[cX - 300: cX + 300,0:720] = 255
    rect = cv2.boundingRect(mask) 
    
    image = image[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]  # crop the image to the desired rectangle
   
    
	# return the image with the contour number drawn on it
    return image
    

    

def findHand(val):
    
    image = cv2.imread('percobaan/testData/predict.jpg')
    accumEdged = np.zeros(image.shape[:2], dtype = "uint8")
    
    for chan in cv2.split(image):
        
        chan = cv2.medianBlur(chan, 11)
        edged = cv2.Canny(chan, 50, 200)
        accumEdged = cv2.bitwise_or(accumEdged, edged)
        
    _, cnts, _ = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    orig = image.copy()
    
    for(i, c) in enumerate(cnts):
        orig = draw_contour(orig, c, i)
        
    (cnts, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")
    cut = draw_contour(image, cnts[0], 0)
    
    #cv2.imshow("ori", image)
    #cv2.imshow("cut", cut )
    scipy.misc.imsave('percobaan/testData/predictCut.jpg', cut)
    return cut

def crop24(val):
    image = val
    arImage = []
    
    width1 = [0, 301, 0, 301]
    width2 = [300, 600, 300, 600]
    height1 = [0, 0, 361, 361]
    height2 = [360, 360, 720, 720]
    
    for i in range(0,4):
        
        mask = np.zeros([600,720], dtype=np.uint8)
        mask[width1[i]:width2[i],height1[i]:height2[i]] = 255
        rect = cv2.boundingRect(mask)
        seg= image[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]  # crop the image to the desired rectangle  
        arImage.append(seg)
    
    rseg1 = arImage[0]
    rseg2 = arImage[1]
    rseg3 = arImage[2]
    rseg4 = arImage[3]
    #cv2.imshow("seg1", rseg1)
    #cv2.imshow("seg2", rseg2)
    #cv2.imshow("seg3", rseg3)
    #cv2.imshow("seg4", rseg4)
    
    return rseg1, rseg2, rseg3, rseg4

def exp(val):
    #cv2.imshow("ori", val)
    
    hasil_hsv = hsvskin(hsv(val))
    
    return hasil_hsv
    
    
    
#preprocess(name)

##exp(im)

#for i in range(1, 10):
    
#hsv = crop_image(name)
#skeleton(hsv)

##ex = cv2.imread("outfile.jpg")

##tangan = findHand(ex)
#skel = skeleton(tangan)
#crop24(skel)
def skelFocus(name):
    img = cv2.imread(name, 0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
 
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    return skel
    #cv2.imshow("skel",skel)

def run():
    arData = []
    im = cv2.imread("jari.jpg")

    exp(im)
    tangan = findHand(exp)
    skelet = skelFocus('percobaan/testData/predictCut.jpg')
    seg1, seg2, seg3, seg4 = crop24(skelet)
    sumSeg1 = np.sum(seg1, axis = None)
    sumSeg2 = np.sum(seg2, axis = None)
    sumSeg3 = np.sum(seg3, axis = None)
    sumSeg4 = np.sum(seg4, axis = None)
    
    arData.append(0)
    arData.append(sumSeg1);
    arData.append(sumSeg2);
    arData.append(sumSeg3);
    arData.append(sumSeg4);
        
        
    myFile = open("testData.csv", "a", newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows([arData])
    arData.clear()

cv2.waitKey(0)
cv2.destroyAllWindows()

