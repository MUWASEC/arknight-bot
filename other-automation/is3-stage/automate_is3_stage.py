#!/usr/bin/env python3

import os, cv2, signal, sys
from random import randint
from cProfile import run
from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
from time import sleep, time
from string import ascii_letters
from concurrent.futures import ThreadPoolExecutor, as_completed

# global var
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

total_io_farm = 0
total_run = 0
verbose=False
### either using ulpianus or mountain
operator_solo_lane = 'Ulpianus'
upgraded_solo_lane = False
### this is for rogue trader box
# the second row is not completely available if investment is not acquired
first_row_box = 4
second_row_box = 3
### this is default screen for dynamic x,y cordinate
screen_x = 1920
screen_y = 1080

'''
this only implemented on Mountain
#1 = move selected operator
#2 = set where operator facing
#3 = after 3 sec sleep, then activate the skill
'''
# sleep 3 = 22
# sleep 4 = 24
metadata_combat = {
    'symbiosis': {
        'priority':1,
        'sleep_iter': 4,
        'sleep_val': 1,
        'emergency': False,
        'auto': [
            ['left', (1484,779)],
            ['right', (1299,602)]
        ]
    },
    'cistern': {
        'priority':3,
        'sleep_iter': 4,
        'sleep_val': 1,
        'emergency': True,
        'auto': [
            ['left', (1490,725)],
            ['bottom', (1300,540)]
        ]
    },
    'insect infestation': {
        'priority':3,
        'sleep_iter': 4,
        'sleep_val': 1,
        'emergency': True,
        'auto': [
            ['left', (1291,616)],
            ['right', (787,598)]
        ]
    },
    'mutual aid': {
        'priority':1,
        'sleep_iter': 4,
        'sleep_val': 1,
        'emergency': False,
        'auto': [
            ['left', (1450,500)],
            ['up', (1310,625)]
        ]
    },
    'sniper squad': {
        'priority':3,
        'sleep_iter': 4,
        'sleep_val': 1,
        'emergency': True,
        'auto': [
            ['bottom', (1426,475)],
            ['right', (1278,456)]
        ]
    },
}

metadata_combat_operator = {
    'facing': {
        'up': (0,0),
        'bottom': (0,0),
        'right': (0,0),
        'left': (0,0),
    },
    'activate_skill': (0,0)
        
}

select_option_encounter = {
    '1_choice': [(1810,575)],
    '2_choice': [(1810,446), (1810,694)],
    '3_choice': [(1810,342), (1810,575), (1810,820)]
}

metadata_encounter = {
    'inheritance': {
        'choices': '1_choice',
        'select': 1
    },
    'devouring dust': {
        'choices': '1_choice',
        'select': 1
    },
    'cliffside burial': {
        'choices': '1_choice',
        'select': 1
    },
    'gathering stormclouds': {
        'choices': '1_choice',
        'select': 1
    },
    'overseas export': {
        'choices': '2_choice',
        'select': 2
    },
    'puppydog eyes': {
        'choices': '2_choice',
        'select': 2
    },
    'catastrophe messenger': {
        'choices': '2_choice',
        'select': 1
    },
    'seaborn scholar': {
        'choices': '2_choice',
        'select': 1
    },
    'delusions of lunacy': {
        'choices': '3_choice',
        'select': 3
    },
    'homecoming': {
        'choices': '3_choice',
        'select': 3
    },
}

metadata_stage = {
    'sp': [
        1800,740,
        1905,805
    ],
    'reward_box_name': [
        54,491,426,582
    ],
    'reward_box_button': [
        76,753,398,842
    ],
    'first_deploy_box': [
        1750, 900,
        1900, 1050
    ],
    'drifting_cache_skip':[
        1300,730,1600,790
    ],
    'recruitment_start_text': [
        30, 270, 487, 320
    ],
    'first_rogue_trader_box': [
        612, 261, 906, 559
    ]
}

metadata_shortcut = {
    'skip_touch': [960,920],
    'skip_check': [950,950],
}

def handle_exit_signal(signal_received, frame):
    """
    Handle the global signal for Ctrl+C to exit the program gracefully.
    """
    print("\nCtrl+C detected. Exiting the program.")
    os.kill(os.getpid(), signal.SIGTERM)

# Attach the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, handle_exit_signal)

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

def crop_image(source_image, output_image, cordinate):
    # Load image
    # with Image.open(source_image) as screen_image:
    # calculate crop
    # image_arr = np.array(screen_image)
    image_arr = source_image
    image_arr = image_arr[
        # y
        # top - bottom
        cordinate['top']:cordinate['bottom'], 
        # x
        # left - right
        cordinate['left']:cordinate['right']]
            
    # crop screen with xy cordinate
    image = Image.fromarray(image_arr)
    image.save(output_image)
    return image_arr

def find_image(haystack, needle, score=0.95, gray_mode=False):
    if gray_mode:
        img_rgb = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY) if type(haystack).__module__ == 'numpy' else cv2.imread(haystack)
        template = cv2.imread(needle, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape
    else:
        img_rgb = cv2.cvtColor(haystack, cv2.COLOR_BGR2RGB) if type(haystack).__module__ == 'numpy' else cv2.imread(haystack)
        template = cv2.imread(needle)
        w, h = template.shape[:-1]
    
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = score
    loc = np.where(res >= threshold)
    resp_xy = []
    for pt in zip(*loc[::-1]):  # Switch collumns and rows
        resp_xy.append((pt[0], pt[1]),)
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('result.png', img_rgb)
    return resp_xy

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

def calculate_offset(xy_loc, full=True):
    if full:
        resp='''
    {
        'top': (screen_y-%d),
        'bottom': (screen_y-%d),
        'left': (screen_x-%d),
        'right': (screen_x-%d)
    }
        ''' % (screen_y-xy_loc['top'], screen_y-xy_loc['bottom'], screen_x-xy_loc['left'], screen_x-xy_loc['right'])
    else:
        resp='''
    {
        'x': (screen_x-%d),
        'y': (screen_y-%d)
    }
        ''' % (screen_x-xy_loc['x'], screen_y-xy_loc['y'])
    print(resp)

def find_text_from_image_full(image_array, search_string, roi=None, config=''):    
    tstart=time()
    
    # Apply ROI if specified
    # roi (tuple): Optional (x, y, width, height) to specify a region of interest.
    if roi:
        x, y, w, h = roi
        image_array = image_array[y:y+h, x:x+w]
    else:
        roi = [0,0]
    
    # Convert the image to grayscale (optional, improves OCR accuracy)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text and their bounding boxes
    data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT, config='--psm 6')

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
            matches.append((x + roi[0], y + roi[1]))
                
    tend=time()
    print(f'[*]===============| time spend {(tend-tstart):.2f}s |===============[*]')
    if matches:
        return matches
    else:
        return False

# https://stackoverflow.com/a/35078614
def find_text_from_image(source_image, text, range=False, full_result=False, thresh_num=180, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.? "'):       
    tstart=time()
    img = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB) if type(source_image).__module__ == 'numpy' else cv2.imread(source_image)
    img_final = img
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, thresh_num, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, thresh_num, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation


    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours
    result = []
    tmp_result = []
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        
        # check with range for faster search
        if range:
            x_search = range['x']
            y_search = range['y']
            if x < x_search or y < y_search:
                # print(x,y)
                continue

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)


        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        # full result only
        if full_result:
            if type(options) == list:
                for option in options:
                    resp = pytesseract.image_to_string(cropped, config=option).strip().replace('\n', ' ')
                    if resp:
                        # print(0,tmp_result)
                        # print(1,resp, [x, y, w, h])
                        if tmp_result != [resp, [x, y, w, h]]:
                            if not resp.isnumeric() and not len(resp) > 5 or ((' of ' not in resp and not ' - ' in resp) and any([not x[0].isupper() for x in resp.split() if all(c in ascii_letters for c in x)])):
                                continue
                            result.append((resp, [x, y, w, h]))
                            tmp_result = [resp, [x, y, w, h]]
            else:
                resp = pytesseract.image_to_string(cropped, config=options).strip()
                if resp:
                    result.append((resp, [x, y, w, h]))
        else:
            resp = pytesseract.image_to_string(cropped, config=options).strip()
            if text in resp:
                result = [x, y, w, h]
                break
    
    # return
    tend=time()
    print(f'[*]===============| time spend {(tend-tstart):.2f}s |===============[*]')
    if result:
        return result
    else:
        return False

def crop_read_string(device, tag, thresh_num=150, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.? "'):    
    img_rgb = device if type(device).__module__ == 'numpy' else do_screenshot(device)
    crop_image(img_rgb, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
    img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
    _, img = cv2.threshold(img,thresh_num,255, cv2.THRESH_BINARY)
    # this resize lol
    img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=2,fy=2))

    return pytesseract.image_to_string(img, config=options).strip()


def startup_journey(device):
    # start journey 
    while True:
        resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'start_explore.png'), score=0.95)
        if resp != []:
            x,y = resp[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(1.5)
            break
        else:
            tag = [940,970]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
    
    # select checkmark
    tag = [940,968,978,1000]
    tag = {
        'top': tag[1],
        'bottom': tag[3],
        'left': tag[0],
        'right': tag[2]
    }
    resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_01.png'), score=0.95)
    if resp != []:
        x,y = resp[0]
        device.shell("input touchscreen tap {0} {1}".format(x,y))

    
    # select squad
    iter = 0
    squad_name = 'Leader Squad'
    while True:
        if iter > 2:
            tag = [
                [1700,300],
                [1500,300],
            ]
            device.shell("input touchscreen swipe %d %d %d %d 100" % (tag[0][0],tag[0][1], tag[1][0],tag[1][1]))
            print('[ERROR] squad is not found! we swiping to the left ...')
            sleep(0.5)
            iter=0
          
        resp = find_text_from_image(do_screenshot(device), squad_name, {'x': 1000, 'y': 600})
        if resp:
            while True:
                device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                sleep(0.5)
                temp_resp = find_text_from_image(do_screenshot(device), 'Are you sure', {'x': 1000, 'y': 800})
                if temp_resp:
                    device.shell("input touchscreen tap {0} {1}".format(temp_resp[0], temp_resp[1]))
                    break
            break
        else:
            print('[ERROR] squad is not found!')
            iter += 1
            continue
    

    # select recruitment set
    recruitment_set_name = 'Overcoming Your'
    while True:    
        sleep(0.5)
        resp = find_text_from_image(do_screenshot(device), recruitment_set_name, {'x': 1000, 'y': 600})
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            resp = False
            while not resp:
                resp = find_text_from_image(do_screenshot(device), 'Are you sure', {'x': 1000, 'y': 0})
            device.shell("input touchscreen tap {0} {1}".format(resp[0], resp[1]))
            break
        else:
            print('[ERROR] recruitment set is not found!')
            continue
        
    # initial recruitment
    voucher_operator = {
        'Guard': operator_solo_lane,
        'Medic': 'Hibiscus'
    }
    for voucher in ['Guard', 'Supporter', 'Medic']:
        # select voucher
        while True:    
            resp = find_text_from_image(do_screenshot(device), f'{voucher} Rec. Voucher', {'x': 350, 'y': 500})
            if resp:
                device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                sleep(1.5)
                break
            else:
                device.shell("input touchscreen tap {0} {1}".format(1800, 80))
                continue
        
        # check each voucher
        tag = [
            [1700,300],
            [1530,300],
        ]
        if voucher == 'Guard' or voucher == 'Medic':
            do_voucher_recruitment(voucher_operator[voucher])
        else:
            do_voucher_recruitment()
    
    # Enter Journey
    while True:    
        resp = find_text_from_image(do_screenshot(device), 'DEEP', {'x': 1600, 'y': 430}, options="--psm 6")
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            break
        else:
            ## to prevent some annoying situation ##
            device.shell("input touchscreen tap {0} {1}".format(metadata_shortcut['skip_check'][0], metadata_shortcut['skip_check'][1]))
       
    ### PHASE 2 ###    
    # setup squad
    while True:
        resp = find_text_from_image(do_screenshot(device), 'Squad')
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            break
        
    while True:
        resp = find_text_from_image_full(do_screenshot(device), 'Select')
        if resp:
            x,y=resp[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            break

    setup_operator(operator_solo_lane, 2)
    setup_operator('Hibiscus')
    
    tag = [1711,1017]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    sleep(0.8)
    print('[DONE] Confirm')

    tag = [148,48]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    print('[DONE] Backward')
    sleep(1)
    
def check_done_explore():
    # wait until end stage
    while True:
        img_src = do_screenshot(device)
        resp = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cleared_battle.png'), score=0.95)
        if resp != []:
            print('[DONE] battle cleared successfully')
            sleep(1.5)
            tag = metadata_shortcut['skip_check']
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            break
        else:
            # tag = [830,500,1062,580]
            if find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'signal_lost.png'), score=0.95):
                print('[WARNING] run stage failed !!!')
                tag = [940,970]
                # [device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1])) for i in range(15)]
                check_exit(device, forced=True)
                print('die')
                # re - enter
                startup_journey(device)
                return False
       
    while True:
        if after_battle():
            after_battle(False)
            print('[DEBUG] all reward done')
            tag = metadata_shortcut['skip_check']
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            break
    
    return
    iter = -5 
    while True:
        tag = [120,820]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        if iter == 5:
            resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cross_abandon.png'), score=0.98)
            sleep(0.5)
            if resp != []:
                # this will avoid any least HOPE
                # select any last 5 operator
                x,y = [randint(680, 1085), randint(770, 900)]
                device.shell("input touchscreen tap {0} {1}".format(x,y))
                sleep(0.5)
                # then select first operator
                x,y = [randint(680, 1085), randint(150, 280)]
                device.shell("input touchscreen tap {0} {1}".format(x,y))
                
                # confirm
                while True:    
                    resp = find_text_from_image(do_screenshot(device), f'Confirm', {'x':1500, 'y': 900})
                    if resp:
                        device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                        sleep(4.2)
                        device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                    else:
                        break
                    
                # another wait checkmark
                iter_checkmark = 0
                while True:
                    # sometimes it stuck on this recruitment shits
                    if iter_checkmark > 10:
                        x,y = [1830,60]
                        device.shell("input touchscreen tap {0} {1}".format(x,y))
                        iter_checkmark = 5
                        continue
                        
                    resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_01.png'), score=0.95)
                    if resp != []:
                        x,y = resp[0]
                        device.shell("input touchscreen tap {0} {1}".format(x,y))
                        return
                    else:
                        iter_checkmark+=1
            else:
                iter=0
        else:
            iter+=1

   
def select_stage(device):
    ### start inline function ###
    def remove_close_duplicates(data, threshold=5):
        """
        Removes close duplicates from a dictionary where values are lists of (x, y) tuples.
        
        Args:
            data (dict): A dictionary where keys are strings and values are lists of (x, y) tuples.
            threshold (int): Maximum distance considered as "close" to remove duplicates.
            
        Returns:
            dict: A cleaned dictionary with close duplicates removed from each key's list.
        """
        def is_close(coord1, coord2):
            """Check if two coordinates are within the threshold distance."""
            return abs(coord1[0] - coord2[0]) <= threshold and abs(coord1[1] - coord2[1]) <= threshold

        cleaned_data = {}
        for key, sublist in data.items():
            cleaned_sublist = []
            for coord in sublist:
                if all(not is_close(coord, existing) for existing in cleaned_sublist):
                    cleaned_sublist.append(coord)
            cleaned_data[key] = cleaned_sublist
        
        return cleaned_data
    
    def find_image_stable_multithread(img_src, templates):
        """Helper function to search for multiple images in parallel."""
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
    ### end inline function ###
    
    arr_resp = {
            "stage_encounter": [],
            "stage_combat": [],
            "stage_emergency": [],
            "stage_regional_commisions": [],
            "stage_downtime_recreation": [],
            "stage_wish_fulfilled": [],
    }
    templates = {
        "stage_encounter": ('stage_encounter.png', 0.98),
        "stage_combat": ('stage_combat.png', 0.99),
        "stage_emergency": ('stage_emergency.png', 0.98),
        "stage_regional_commisions": ('stage_regional_commisions.png', 0.98),
        "stage_downtime_recreation": ('stage_downtime_recreation.png', 0.98),
        "stage_wish_fulfilled": ('stage_wish_fulfilled.png', 0.98)
    }
    iter = 0
    while iter != 1:
        img_src = do_screenshot(device)
        resp_rogue_trader = find_image_stable(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rogue_trader.png'), threshold=0.95)
        if resp_rogue_trader:
            print('[INFO] Rogue Trader slots fuckk')
            x,y=resp_rogue_trader[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(1)
            tag = [1714,735]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            sleep(1.5)
            return True

        resp = find_image_stable_multithread(img_src, templates)
        if resp == {'stage_encounter': [], 'stage_combat': [], 'stage_emergency': [], 'stage_regional_commisions': [], 'stage_downtime_recreation': [], 'stage_wish_fulfilled': []}:
            print('[ERROR] stage not found')
            ## to prevent some annoying check popup ##
            device.shell("input touchscreen tap {0} {1}".format(metadata_shortcut['skip_check'][0], metadata_shortcut['skip_check'][1]))
            continue
        else:
            for key, value in resp.items():
                if key in arr_resp:
                    arr_resp[key].extend(value)
            iter += 1
    
    
    print(f'[DEBUG_arr_resp] {(arr_resp)}')
    print(f'[DEBUG_before_resp] {(resp)}')
    resp = remove_close_duplicates(arr_resp)
    print(f'[DEBUG_after_resp] {(resp)}')
         
    stage_metadata = {
        'stage_encounter' : {
            'total': len(resp['stage_encounter']),
            'cordinate': resp['stage_encounter']
        },
        'stage_combat' : {
            'total': len(resp['stage_combat']),
            'cordinate': resp['stage_combat']
        },
        'stage_emergency' : {
            'total': len(resp['stage_emergency']),
            'cordinate': resp['stage_emergency']
        },
        'stage_regional_commisions' : {
            'total': len(resp['stage_regional_commisions']),
            'cordinate': resp['stage_regional_commisions']
        },
        'stage_downtime_recreation' : {
            'total': len(resp['stage_downtime_recreation']),
            'cordinate': resp['stage_downtime_recreation']
        },
        'stage_wish_fulfilled' : {
            'total': len(resp['stage_wish_fulfilled']),
            'cordinate': resp['stage_wish_fulfilled']
        },
    }
    
    # select all type of open stage            
    print(f'[DEBUG] {(stage_metadata)}')
    # exit()
    multi_stage = []
    for key in stage_metadata:
        if stage_metadata[key]['total'] != 0:
            if key == 'stage_encounter' or key == 'stage_regional_commisions' or key == 'stage_downtime_recreation' or key == 'stage_wish_fulfilled':
                for cord in stage_metadata[key]['cordinate']:
                    multi_stage.append((key,cord))
            elif key == 'stage_combat' or key == 'stage_emergency':
                for cord in stage_metadata[key]['cordinate']:
                    print(cord)
                    tag = [
                        cord[0]+53,cord[1]+56,
                        cord[0]+256,cord[1]+120
                    ]
                    tag = {
                        'top': tag[1],
                        'bottom': tag[3],
                        'left': tag[0],
                        'right': tag[2]
                    }
                    found=True
                    while found:
                        crop_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
                        str_res = crop_read_string(device, tag, thresh_num=100, options=r'-c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "').lower().replace('x ', '').split('\n')
                        print(str_res)
                        for x in str_res:
                            if x in metadata_combat:
                                multi_stage.append((x,cord))
                                found=False
                                break
    
    ## sort multiple stage by their priority ##
    def get_stage_priority(stage_name):
        if stage_name == "stage_encounter" or stage_name == "stage_wish_fulfilled":
            return 2  # Static priority for stage_encounter
        elif stage_name in metadata_combat:
            return metadata_combat.get(stage_name, {}).get("priority", float("inf"))
        elif stage_name == "stage_regional_commisions" or stage_name == "stage_downtime_recreation":
            return 1
    print('[DEBUG] before', multi_stage)
    multi_stage=sorted(
        multi_stage, key=lambda stage: get_stage_priority(stage[0])
    )
    print('[DEBUG] after', multi_stage)

    ## get the last sorted stage (no-random-select)
    choose_stage = multi_stage[-1][0]
    x,y = multi_stage[-1][1]
    str_res = choose_stage
    print(f'[INFO] select {choose_stage} lol')
    device.shell("input touchscreen tap {0} {1}".format(x,y))
    sleep(0.8)
    
    if choose_stage == 'stage_encounter':
        # enter -> start mission
        tag = [1714,735]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        tag = metadata_shortcut['skip_touch']
        for i in range(10):
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        # get encounter name
        tag = [166,690,600,950]
        tag = {
            'top': tag[1],
            'bottom': tag[3],
            'left': tag[0],
            'right': tag[2]
        }
        
        str_res = ''
        while True:
            str_res = crop_read_string(device, tag, thresh_num=100, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "').lower().split('\n')
            print(str_res)
            for x in str_res:
                if any([x in [y for y in metadata_encounter.keys()]]):
                    str_res = (True, x)
                    break
                else:
                    str_res = (False, x)
            if str_res[0]:
                str_res = str_res[1]
                break
        print(str_res)
        choices = metadata_encounter[str_res]['choices']
        select = metadata_encounter[str_res]['select']
        x,y = select_option_encounter[choices][select-1]
        
        # if str_res == 'homecoming':
            # input()
        
        ## spam input ##
        for _ in range(20):
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.1)
            
        # if str_res == 'homecoming':
        #     input()
        #     x,y=metadata_shortcut['skip_check']
        #     for i in range(20): device.shell("input touchscreen tap {0} {1}".format(x,y))
        #     sleep(3)
        # else:
        #     device.shell("input touchscreen tap {0} {1}".format(x,y))
            
    elif choose_stage == 'stage_regional_commisions':
        # enter -> start mission
        tag = [1714,735]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        for i in range(10):
            x,y = [1800, 919]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.3)
            
    elif choose_stage == 'stage_downtime_recreation':
        # select second choice
        x,y = select_option_encounter['2_choice'][1]
        for _ in range(20):
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.1)
        
        # select obtained OI
        x,y = select_option_encounter['1_choice'][0]
        for _ in range(20):
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.1)
        
    elif choose_stage == 'stage_wish_fulfilled':
        # enter -> start mission
        tag = [1714,735]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        # select first choice
        x,y = select_option_encounter['2_choice'][0]
        for _ in range(20):
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.1)
        
    else:
        # enter -> start mission
        tag = [1714,735]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        sleep(1)
        tag = [1720,1000]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))           
        
        ### Stage Ongoing ####
        while True:
            resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ongoing_battle.png'), score=0.95)
            if resp != []:
                sleep(0.5)
                break
        
        # 2X Speed
        tag = [1650,75]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        sleep(3)
        
        deploy_operator(str_res)

        # collect all collectible item then continue ...
        check_done_explore()
    return False

def automate_rogue_trader(device):
    global total_io_farm, first_row_box, second_row_box
    
    tag = {
        'top': (screen_y-1050),
        'bottom': (screen_y-1010),
        'left': (screen_x-330),
        'right': (screen_x-230)
    }
    img_data = do_screenshot(device)
    for i in [100, 150, 180, 200]:
        originium_ingot = crop_read_string(img_data,tag,thresh_num=i,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        if originium_ingot.isnumeric():
            break
    
    originium_ingot = int(originium_ingot)
    # resp = find_text_from_image(do_screenshot(device), '', {'x': 500, 'y':140},full_result=True,thresh_num=180, options=["--psm 10 -c tessedit_char_whitelist=\"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-. \"", "-c tessedit_char_whitelist=\"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-. \""])
    # print(f'[DEBUG]\n {resp}')
    
    tag = metadata_stage['first_rogue_trader_box']
    tag = {
        'top': tag[1],
        'bottom': tag[3],
        'left': tag[0],
        'right': tag[2]
    }
    box_gap = 18
    box_width = 290
    box_height = 300
    result = {
        'first_row':[],
        'second_row':[]
    }

    ## first row ##
    for i in range(first_row_box):
        cur_tag = tag.copy()
        cur_tag['right'] += (i*(box_gap + box_width))
        cur_tag['left'] += (i*(box_gap + box_width))
        box_name = ''
        box_price = 0
        
        ## get box name ##
        while not box_name:
            box_name = crop_read_string(img_data, cur_tag, thresh_num=100, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. "').lower().split('\n')
            box_name=box_name[0]
        result['first_row'].append({
            box_name: {
                'price': 0,
                'cordinate': generate_random_point_in_box(cur_tag, with_pos=True)
            }
        })
        
        ## get price ##
        cur_tag['top'] += 210
        for x_tresh in [150, 100, 180, 200]:
            box_price = crop_read_string(img_data,cur_tag,thresh_num=x_tresh,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            if box_price.isnumeric():
                print(f'[INFO] found price : {box_price}')
                break
            else:
                box_price = -1
        
        result['first_row'][i][box_name]['price']=int(box_price)
    ## end first row ##
   
    ## second row ##
    for i in range(second_row_box):
        cur_tag = tag.copy()
        cur_tag['right'] += (i*(box_gap + box_width))
        cur_tag['left'] += (i*(box_gap + box_width))
        cur_tag['top'] += box_gap+box_height
        cur_tag['bottom'] += box_gap+box_height
        box_name = ''
        box_price = 0
        
        ## get box name ##
        while not box_name:
            box_name = crop_read_string(img_data, cur_tag, thresh_num=100, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. "').lower().split('\n')
            box_name=box_name[0]
            
        if len(box_name) < 3:
            break
        
        result['second_row'].append({
            box_name: {
                'price': 0,
                'cordinate': generate_random_point_in_box(cur_tag, with_pos=True)
            }
        })
        
        ## get price ##
        cur_tag['top'] += 210
        for x_tresh in [150, 100, 180, 200]:
            box_price = crop_read_string(img_data,cur_tag,thresh_num=x_tresh,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            if box_price.isnumeric():
                print(f'[INFO] found price : {box_price}')
                break
            else:
                box_price = -1
        
        result['second_row'][i][box_name]['price']=int(box_price)
    ## end second row ##
    
    if True:
        print(f'[DEBUG] RESULT:\n {result}')
        print(f'[INFO] total OI : {originium_ingot}')

        # check if investment system exists
        if 'prospective investment system' in result['first_row'][0] and True:
            print('[INFO] Investment exists!')

            # automate invest lul
            x,y= result['first_row'][0]['prospective investment system']['cordinate']
            device.shell("input touchscreen tap {0} {1}".format(x, y))
            sleep(1.5)

            # break after OOS
            total = 0
            while True:
                tag = [1175,455,1300,480]
                tag = {
                    'top': tag[1],
                    'bottom': tag[3],
                    'left': tag[0],
                    'right': tag[2]
                }
                resp_str = crop_read_string(device, tag)
                if 'Out of Service' == resp_str or originium_ingot == 0:
                    print('[WARN] Bad RNG, Out of Service (ฅ`w´ฅ)~')
                    print(f'[INFO] Total {total}x run ฅ(ﾐ・ᆽ・ﾐ)ฅ')
                    total_io_farm += total
                    x,y = generate_random_point_in_box([713, 717, 1732, 765])
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(1)
                    break
                else:
                    # tap investment entrance #
                    x,y = generate_random_point_in_box([714, 435, 1288, 622])
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(0.5)
                    
                    # tap confirm #
                    x,y = generate_random_point_in_box([1220, 715, 1730, 770])
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(0.8)
                    
                    # tap nahh #
                    x,y = generate_random_point_in_box([680, 715, 1200, 770])
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(2.5)
                    
                    originium_ingot-=1
                    total+=1
                    print(f'[INFO] Good RNG ฅ(≈>ܫ<≈)ฅ')

        # exit()
        # no investment
        # merge first + second row
        i,y = (result['first_row'], result['second_row'])
        z = i.copy()
        [z.append(y[i]) for i in range(len(y))]
        
        # parsing all item in the shop #
        unique_accessories = []
        support_item = []
        voucher_recruit = []
        for i,data in enumerate(z):
            if i != 0:
                if data[[v for v in data][0]]['price'] != 0:
                    if 'rec. voucher' in [v for v in data][0]:
                        voucher_recruit.append(data)
                    elif any([i in [v for v in data][0] for i in ['support', ' advancement']]):
                        support_item.append(data)
                    else:
                        unique_accessories.append(data)

        print(f'[DEBUG] Accessories :\n {unique_accessories}\n[DEBUG] Supportive :\n {support_item}\n[DEBUG] Voucher :\n {voucher_recruit}')        
        # # get all accessories
        # blacklist_item = ['training']
        # if unique_accessories:
        #     for item in unique_accessories:
        #         if originium_ingot >= item[[v for v in item][0]]['price'] and [v for v in item][0] not in blacklist_item:
        #             print(f'[INFO][1] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
        #             x,y= item[[v for v in item][0]]['cordinate']
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(1)
        #             x,y = generate_random_point_in_box([1230, 720, 1730, 760])
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(1)
        #             originium_ingot -= item[[v for v in item][0]]['price']
        #             print(f'[INFO] total OI : {originium_ingot}')
        
        # # get all supportive item
        # if support_item:
        #     for item in support_item:
        #         # skip useless Training lol
        #         if '- Training' in [v for v in item][0]:
        #             continue
        #         elif originium_ingot >= item[[v for v in item][0]]['price']:
        #             print(f'[INFO][2] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
                                                
        #             x,y = item[[v for v in item][0]]['cordinate']
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(1)
        #             x,y = generate_random_point_in_box([1230, 720, 1730, 760])
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(1)
        #             print(item[[v for v in item][0]])
        #             originium_ingot -= item[[v for v in item][0]]['price']
        #             print(f'[INFO] total OI : {originium_ingot}')
        
        
        # # get all recruit voucher
        # if voucher_recruit:
        #     for item in voucher_recruit:
        #         if originium_ingot >= item[[v for v in item][0]]['price']:
        #             print(f'[INFO][3] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
        #             x,y = item[[v for v in item][0]]['cordinate']
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(1)
        #             x,y = generate_random_point_in_box([1230, 720, 1730, 760])
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(2.5)
                    
        #             # this will avoid any least HOPE
        #             # select any last 5 operator
        #             x,y = generate_random_point_in_box([680, 770, 1085, 900])
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
        #             sleep(0.5)
        #             # then select first operator
        #             x,y = generate_random_point_in_box([680, 150, 1085, 280])
        #             device.shell("input touchscreen tap {0} {1}".format(x,y))
                    
        #             # confirm
        #             while True:
        #                 print(0,)  
        #                 resp = find_text_from_image(do_screenshot(device), f'Confirm', {'x':1500, 'y': 900})
        #                 if resp:
        #                     device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
        #                     sleep(4.2)
        #                     device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
        #                 else:
        #                     break
                    
    
        # input('sukebe')
        # exit
        sleep(0.5)

        # x,y = [randint(980, 1900), randint(700, 785)]
        # device.shell("input touchscreen tap {0} {1}".format(x,y))
        # sleep(1)
        end_journey()

def end_journey():
    # checkmark
    while True:
        device.shell("input keyevent 4")
        sleep(1.5)
        x,y = [randint(1691, 1787), randint(477, 543)]
        device.shell("input touchscreen tap {0} {1}".format(x,y))
        sleep(0.5)
        resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_02.png'), score=0.9)
        if resp != []:
            x,y = resp[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            break
        
    check_exit(device)
              
def check_exit(device, forced=False):
    print('[INFO] enter Exit function')
    # exit
    iter = 0
    iter_forced = 0
    # tag = [940,970]
    tag = metadata_shortcut['skip_touch']
    print('|',end='', flush=True)
    while True:
        if iter > 10:
            resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'start_explore.png'), score=0.98)
            if resp != []:
                sleep(0.5)
                iter_forced+=1
                if forced and iter_forced < 2:
                    print('*>', end='', flush=True)
                    continue
                break
            else:
                print('->',end='', flush=True)
                iter = 5
        else:    
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            sleep(0.5)
            print('+>',end='', flush=True)
            iter += 1
    print('|')

def setup_operator(name, skill=1):
    roi_x, roi_y, roi_width, roi_height = 500, 100, 1900, 900
    while True:
        sleep(0.5)
        resp = find_text_from_image_full(do_screenshot(device), name, roi=(roi_x, roi_y, roi_width, roi_height), config="--psm 6")
        print(resp)
        if resp:
            x,y = resp[0][0], resp[0][1]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(0.5)
            break
        else:
            tag = [
                [1700,300],
                [1500,300],
            ]
            device.shell("input touchscreen swipe %d %d %d %d 200" % (tag[0][0],tag[0][1], tag[1][0],tag[1][1]))
            print(f'[ERROR] {name} is not found! we swiping to the right ...')
            sleep(0.5)

    if skill == 2:
        tag = [360,840]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        sleep(0.1)
        print('[DONE] Second Skill')        

def check_sp():
    tag = metadata_stage['sp']
    tag = {
        'top': tag[1],
        'bottom': tag[3],
        'left': tag[0],
        'right': tag[2]
    }
    while True:
        resp = crop_read_string(device,tag,thresh_num=200,options=r'--psm 6 -c tessedit_char_whitelist=0123456789')
        print(resp)
        
def generate_random_point_in_box(box, with_pos=False):
    """
    Generate a random (x, y) point inside a box.

    Parameters:
        box (list): A list of four integers [x1, y1, x2, y2] representing the box's top-left and bottom-right corners.

    Returns:
        tuple: A random (x, y) point inside the box.
    """
    if with_pos:
        box = (box['left'],box['top'], box['right'],box['bottom'])
    x1, y1, x2, y2 = box
    random_x = randint(x1, x2)
    random_y = randint(y1, y2)
    return (random_x, random_y)

def do_voucher_recruitment(operator_name='temporary_recruitment'):
    def check_resp(resp, tag):
        if resp:
            x,y = resp[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            return True
        else:
            device.shell("input touchscreen swipe %d %d %d %d 2000" % (tag[0],tag[1], tag[2],tag[3]))
            print(f'[ERROR] {operator_name} is not found! we swiping to the right ...')
            return False
    
    ## check until recruitment start ##
    str_res=''
    while not str_res and 'operator' not in str_res:
        str_res = crop_read_string(device, {
        'top': metadata_stage['recruitment_start_text'][1],
        'bottom': metadata_stage['recruitment_start_text'][3],
        'left': metadata_stage['recruitment_start_text'][0],
        'right': metadata_stage['recruitment_start_text'][2]
    }, thresh_num=200, options=r'-c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "').lower()
        print('[DEBUG] recruitment start:', str_res)
    
    ## check recruitment ##
    tag = [1755,320,1000,320]
    iter_not_found = 0
    while True:
        if operator_name != 'temporary_recruitment':
            roi_x, roi_y, roi_width, roi_height = 530, 90, 1900, 900
            resp = find_text_from_image_full(do_screenshot(device), operator_name, roi=(roi_x, roi_y, roi_width, roi_height), config="--psm 6")
        else:
            resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'select_temporary_recruitment.png'), gray_mode=True)
        
        print(resp)
        if check_resp(resp, tag):
            break
        else:
            iter_not_found += 1
            
        ## if not found (anomaly) then we swipe reverse ##
        if iter_not_found > 10:
            x1,x2=tag[0],tag[2]
            tag[2]=x1
            tag[0]=x2
            iter_not_found=0
        
    ## confirm recruitment operator ##
    while True:    
        resp = find_text_from_image(do_screenshot(device), f'Confirm', {'x':1500, 'y': 900})
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            sleep(4.2)
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            break
        else:
            continue

def after_battle(checked=True):
    global upgraded_solo_lane
    tag = {
        'top': metadata_stage['reward_box_name'][1],
        'bottom': metadata_stage['reward_box_name'][3],
        'left': metadata_stage['reward_box_name'][0],
        'right': metadata_stage['reward_box_name'][2]
    }
    str_res=''
    while not str_res and checked:
        str_res = crop_read_string(device, tag, thresh_num=100, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. "').lower()
        if not str_res:
            ## to prevent some annoying situation ##
            device.shell("input touchscreen tap {0} {1}".format(metadata_shortcut['skip_check'][0], metadata_shortcut['skip_check'][1]))
    print(f"[DEBUG] found reward: {str_res}")
    sleep(0.5)
    x,y = generate_random_point_in_box(metadata_stage['reward_box_button'])
    device.shell("input touchscreen tap {0} {1}".format(x, y))
    
    if 'rec. voucher' in str_res:
        if str_res == 'guard rec. voucher' and not upgraded_solo_lane:
            do_voucher_recruitment(operator_solo_lane)
            upgraded_solo_lane = True
        else:
            do_voucher_recruitment()
        return True
    elif 'drifting cache' == str_res:
        sleep(0.8)
        x,y = generate_random_point_in_box(metadata_stage['drifting_cache_skip'])
        device.shell("input touchscreen tap {0} {1}".format(x, y))
        return False
        
def deploy_operator(stage_name):
    def get_position(metadata_auto):
        x,y = metadata_auto[1]
        if metadata_auto[0] == 'up':
            y -= 200
        elif metadata_auto[0] == 'bottom':
            y += 200
            x += 50
        elif metadata_auto[0] == 'right':
            x += 200
        elif metadata_auto[0] == 'left':
            x -= 200
        return (x,y)
    
    # stage_name = 'mutual aid'
    for i in range(metadata_combat[stage_name]['sleep_iter']):
        sleep(metadata_combat[stage_name]['sleep_val'])
        print('.',end='',flush=True)
    print()
    
    # deploy first operator
    x,y = metadata_combat[stage_name]['auto'][0][1]
    x_facing,y_facing = get_position(metadata_combat[stage_name]['auto'][0])
    tag_deploy_x, tag_deploy_y = generate_random_point_in_box(metadata_stage['first_deploy_box'])
    device.shell("input touchscreen swipe %d %d %d %d 1000" % (tag_deploy_x, tag_deploy_y, x+150,y-30))
    sleep(0.5)
    device.shell("input touchscreen swipe %d %d %d %d 500" % (x,y, x_facing,y_facing))
    
    sleep(7)
    
    x,y = metadata_combat[stage_name]['auto'][1][1]
    x_facing,y_facing = get_position(metadata_combat[stage_name]['auto'][1])
    tag_deploy_x, tag_deploy_y = generate_random_point_in_box(metadata_stage['first_deploy_box'])
    device.shell("input touchscreen swipe %d %d %d %d 1000" % (tag_deploy_x, tag_deploy_y, x+150,y-30))
    sleep(0.5)
    device.shell("input touchscreen swipe %d %d %d %d 500" % (x,y, x_facing,y_facing))
    # # wait Mountain skill sp
    # sleep(3) 
    # # active Mountain skill 2
    # device.shell("input touchscreen tap %d %d" % (metadata_combat[str_res][2][0], metadata_combat[str_res][2][1]))
    # sleep(0.5)
    # device.shell("input touchscreen tap %d %d" % (metadata_combat[str_res][2][2] ,metadata_combat[str_res][2][3]))
    # sleep(0.5)

def debug():
    tag = [515,450,717,523]  
    tag = {
        'top': tag[1],
        'bottom': tag[3],
        'left': tag[0],
        'right': tag[2]
    }
    crop_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
    # resp = find_image_stable(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_combat.png'))
    # print(resp)
    # resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_02.png'), score=0.95)
    # print(resp)
        
    # check_sp()
    # check_exit(device)
    # startup_journey(device)
    # select_stage(device)
    # select_stage(device)
    select_stage(device)
    # do_voucher_recruitment()
    # check_done_explore()
    # after_battle()
    # do_voucher_recruitment('Ulpianus')
    # do_voucher_recruitment('Hibiscus')
    # automate_rogue_trader(device)

    # end_journey()
    # deploy_operator()
    
    exit(0)

if __name__ == "__main__":
    adb = Client(host='127.0.0.1', port=5037)
    device = adb.devices()[0]
    # debug()
    while True:
        ## initialize global var ##
        upgraded_solo_lane = False
        tstart=time()
        startup_journey(device)
        while True:
            if select_stage(device):
                break
        
        automate_rogue_trader(device)
        tend=time()
        total_run +=1
        print(f'\n[*]====================[*]')
        print(f'[*]===| run time spend {(int(tend-tstart)/60):.2f}m/{tend-tstart:.2f}s')
        print(f'[*]===| total run {total_run}x')
        print(f'[*]===| total farm OI {total_io_farm}')
        print(f'[*]====================[*]\n')