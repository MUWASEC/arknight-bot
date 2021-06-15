import cv2
import numpy as np
from pytesseract import pytesseract

pytesseract.tesseract_cmd = '/usr/sbin/tesseract'

img = cv2.imread('./crop.png')
custom_oem=r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/'

# retval, img = cv2.threshold(img,100,255, cv2.THRESH_BINARY)
# img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=3,fy=2))
# img = cv2.GaussianBlur(img,(11,11),0)
# img = cv2.medianBlur(img,9)
# cv2.imwrite('./crop_new.png', img)
if pytesseract.image_to_string(img).split('\n')[0] == "Are you sure you want to exit?":
    print("yes")