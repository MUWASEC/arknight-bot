import os, cv2
import pytesseract
from pytesseract import Output
from random import randint
from cProfile import run
from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
from time import sleep, time
from collections import Counter
from string import ascii_letters

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def do_screenshot(device):
    while True:
        try:
            image = device.screencap()
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), 'wb') as fd:
                fd.write(image)
            with Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png')) as fd:
                image = np.array(fd, dtype=np.uint8)
            return image
        except:
            pass

def search_string_in_image(image_array, search_string):
    """
    Search for a string in an image and highlight occurrences.

    Parameters:
        image_array (np.array): Image data as a NumPy array.
        search_string (str): The string to search for.

    Returns:
        list: A list of tuples containing the (x, y) coordinates of the matches.
    """
    x, y, w, h = (500, 100, 1900, 900)
    image_array = image_array[y:y+h, x:x+w]
    
    # Convert the image to grayscale (optional, improves OCR accuracy)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text and their bounding boxes
    data = pytesseract.image_to_data(gray_image, output_type=Output.DICT, config='--psm 6')

    matches = []

    # Iterate through the detected text
    for i in range(len(data['text'])):
        word = data['text'][i]
        if search_string.lower() in word.lower():
            # Get the bounding box coordinates for the matched text
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Draw a rectangle around the matched text
            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Append the top-left corner coordinates to the matches list
            matches.append((x, y))

    # Display the image with highlighted matches
    cv2.imshow("Matches", image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return matches

def search_image_in_image(main_image, template_image, threshold=0.98):
    """
    Search for a template image within a larger image and return the match locations.

    Parameters:
        main_image (np.array): The larger image where the search is performed.
        template_image (np.array): The template image to search for.
        threshold (float): The similarity threshold for detecting matches (default: 0.8).

    Returns:
        list: A list of tuples containing the top-left corner coordinates of matches.
    """
    template_image = cv2.imread(template_image)
    # Convert images to grayscale for better matching
    gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Get the width and height of the template image
    template_height, template_width = gray_template.shape

    # Perform template matching
    result = cv2.matchTemplate(gray_main, gray_template, cv2.TM_CCOEFF_NORMED)

    # Find locations where the matching value exceeds the threshold
    match_locations = np.where(result >= threshold)

    matches = []

    # Iterate through the match locations and collect top-left corners
    for (y, x) in zip(*match_locations[::-1]):
        matches.append((y, x))

        # Draw a rectangle around the matched region (optional visualization)
        cv2.rectangle(main_image, (x, y), (x + template_width, y + template_height), (0, 255, 0), 2)

    # Display the result (optional)
    # cv2.imshow("Matches", main_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return matches

adb = Client(host='127.0.0.1', port=5037)
device = adb.devices()[0]
print(search_string_in_image(do_screenshot(device), "Ulpianus"))
# print(search_image_in_image(do_screenshot(device), 'stage_emergency.png'))