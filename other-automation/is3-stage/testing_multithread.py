import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from ppadb.client import Client
from pytesseract import pytesseract
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

adb = Client(host='127.0.0.1', port=5037)
device = adb.devices()[0]

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
        
img_src = do_screenshot(device)

def find_image_stable(main_image, template_image, threshold=0.98):
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

    return matches

def find_image_stable_multithread(device):
    stage_metadata = {}
    iter = []
    
    def find_images(img_src):
        """Helper function to search for multiple images in parallel."""
        templates = {
            # "rogue_trader": ('rogue_trader.png', 0.95),
            "stage_encounter": ('stage_encounter.png', 0.98),
            "stage_combat": ('stage_combat.png', 0.99),
            "stage_emergency": ('stage_emergency.png', 0.98),
            "stage_regional_commisions": ('stage_regional_commisions.png', 0.98),
            "stage_downtime_recreation": ('stage_downtime_recreation.png', 0.98),
            "stage_wish_fulfilled": ('stage_wish_fulfilled.png', 0.98)
        }

        results = {}
        with ThreadPoolExecutor(max_workers=len(templates)) as executor:
            future_to_name = {
                executor.submit(
                    find_image_stable, 
                    img_src, 
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), path), 
                    threshold
                ): name
                for name, (path, threshold) in templates.items()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    print(f"[ERROR] Exception occurred for {name}: {e}")
                    results[name] = []

        return results
    
    img_src = do_screenshot(device)
    results = find_images(img_src)
    print(results)
    
def manual():
    img_src = do_screenshot(device)
    resp_rogue_trader = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rogue_trader.png'), threshold=0.95)
    resp_encounter = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_encounter.png'), threshold=0.98)
    resp_combat = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_combat.png'), threshold=0.99)
    resp_emergency = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_emergency.png'), threshold=0.98)
    resp_regional_commisions = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_regional_commisions.png'), threshold=0.98)
    resp_downtime_recreation = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_downtime_recreation.png'), threshold=0.98)
    resp_wish_fulfilled = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_wish_fulfilled.png'), threshold=0.98)
    resp = [resp_encounter, resp_combat, resp_emergency, resp_regional_commisions, resp_downtime_recreation, resp_wish_fulfilled]
    print(resp)


tstart=time()
find_image_stable_multithread(device)
# manual()
tend=time()
print(f'[*]===| run time spend {(int(tend-tstart)/60):.2f}m/{tend-tstart:.2f}s')