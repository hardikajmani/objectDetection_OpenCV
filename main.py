from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin


green = [0, 255, 8]

def show(image):

    #figure size inches
    #use of matplotlib
    plt.figure(figsize=[10,10])
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
    #convert color scheme of the mask to rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) # doing this coz actual image is in RGB format
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):

    #copy image
    image = image.copy()
    _,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #finding all possible contours   here the function returns 3 values instead of two as in 2.X

    #isolating the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    #return the biggest contour
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour):
    #drawing the ellipse
    #bounding the ellipse

    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)

    #addit
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)   #note its CV_AA in 2.X but we are using 3.X so its LINE_AA
    return image_with_ellipse






def find_strawberry(image): #main method of the code
    #RGB Red Green Blue
    #BGR Blue Green Red
    #step 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # changing image to correct color scheme
    
    #step 2- scale our image properly 
    max_dimension = max(image.shape)  #takes the size of the image
    scale = 700/max_dimension  #bringing down the scale
    image = cv2.resize(image, None, fx=scale, fy=scale)  #resizing the image and making sure that its a square

    #step3 - clean the image
    image_blur = cv2.GaussianBlur(image, (7,7), 0)  #this is done to remove noise from the image and make a image more smoother
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV) # COnvert of image from RGB to HSV

    #step-4 Define the filters
    #fliter by color
    min_red= np.array([0, 100, 80])
    max_red= np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    #filter-by- brightness
    min_red = np.array([170, 100, 80])
    max_red = np.array([100, 256, 256])

    mask2 = cv2.inRange(image_blur_hsv, min_red, max_red)

    #combine the masks
    mask = mask1 + mask2

    #step5- segemntation (to seperate strawberry form the background)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    #step6 find the biggest among all the strawberries
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

    #step 7 to overlay the masks on the images we created
    overlay = overlay_mask(mask_clean, image)

    #step8 cirlce the biggest strawbery
    circled = circle_contour(overlay, big_strawberry_contour)

    show(circled)

    #step 9 (last step) covert the image back to original color scheme
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr     #end of the main mathod

#now we have to write the main method


#read the image

image = cv2.imread('berry.jpg')
result = find_strawberry(image)
cv2.imwrite('yo3.jpg', result)
