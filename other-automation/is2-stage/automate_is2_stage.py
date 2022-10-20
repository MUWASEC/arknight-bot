#!/usr/bin/env python3

from ast import iter_child_nodes
import os, cv2
from random import randint
from cProfile import run
from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
from time import sleep, time
from collections import Counter
from string import ascii_letters

# global var
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

total_io_farm = 0
total_run = 0
verbose=False
screen_x = 1920
screen_y = 1080

'''
this only implemented on Mountain
#1 = move selected operator
#2 = set where operator facing
#3 = after 3 sec sleep, then activate the skill
'''
metadata_combat = {
    'Beast Taming': [
        [1560,530],
        [1560,530, 1760,520],
        [1460,560, 1320,620]
    ],
    'Gun Salute': [
        [760,515],
        [760,515, 940,515],
        [600,480, 1270,610]
    ],
    'A Date With Slugs': [
        [1120,660],
        [1120,660, 1320,640],
        [1030,660, 1240,620]
    ],
    'Accident': [
        [1080,480],
        [1080,480, 1087,782],
        [960,490, 1270,610],
    ]
}

metadata_encounter = {
    'The Kind Puppet': [
        # (1795,699, 0.5),
        # (1795,699, 2.1),
        # (1795,699, 0),
        # (1795,699, 1),
        (1795,601, 0.5),
        (1795,601, 2.1),
        (1795,601, 0),
        (1795,601, 1),
    ],
    'Reprieve?': [[
        (1795,699, 0.5),
        (1795,699, 2.1),
        (1795,699, 0),
        (1795,699, 1),
        ],[
        (1795,455, 0.5),
        (1795,455, 1),
        (1795,455, 0),
        
        (1795,455, 0.5),
        (1795,455, 1),
        (1795,455, 0),
        
        # spam
        (1795,455, 0.5),
        (1795,455, 0.5),
        (1795,455, 0.5),
        (1795,455, 3),
        
        # got ursus doll
        # (randint(926, 996), randint(674, 733), 0)
        # (randint(930, 990), randint(670, 700), 1),
        (960,705, 2),
        (960,705, 0),
        (960,705, 1),
        
        
        ]
    ],
    'Adventurer Commission': [
        (1795,812, 0.5),
        (1795,812, 2.1),
        (1795,812, 0),
        (1795,812, 1),
    ],
    'Hallucinatory Candlelights': [
        (1795,699, 0.5),
        (1795,699, 2.1),
        (1795,699, 0),
        (1795,699, 1),
        ],
    'Coffin of Evil': [
        (1795,699, 0.5),
        (1795,699, 2.1),
        (1795,699, 0),
        (1795,699, 1),
    ],
    'Camp': [
        (1795,601, 0.5),
        (1795,601, 2.1),
        (1795,601, 0),
        (1795,601, 2),
        # insert *sometime meme
        (1795,601, 0),
        (1795,601, 2),
        (1795,601, 0),
        (1795,601, 0),
        (1795,601, 2),
        
    ],
    'Duck Lord the Lead Actor?': [
        (1795,715, 0.5),
        (1795,715, 2.1),
        (1795,715, 0),
        (1795,715, 1),
    ],
    'Secret Chamber': [
        (1795,715, 0.5),
        (1795,715, 2.1),
        (1795,715, 0),
        (1795,715, 1),
    ],
    'A Tiny Stage': [
        # (1795,488, 0.5),
        # (1795,488, 2.1),
        # (1795,488, 0),
        
        (1795,715, 0.5),
        (1795,715, 0.5),
        (1795,715, 2.1),
        # insert *sometime meme
        (1795,715, 0),
        (1795,715, 2),
        (1795,715, 0),
    ],
    'Secret Entrance': [
        (1795,488, 0.5),
        (1795,488, 2.1),
        (1795,488, 0),
        (1795,488, 1),
    ],

    # downtime recreation
    'Potent Potables': [
        (1795,924, 0.5),
        (1795,924, 1.5),
        (1795,924, 0),
        (1795,924, 1),
    ],
    'Curio Keeper':[
        (1795,715, 0.5),
        (1795,715, 2.1),
        (1795,715, 0),
        (1795,715, 1),
    ],
    
    
}

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
    # img = cv2.GaussianBlur(img,(11,11),2)
    # print(pytesseract.image_to_string(img, config=options).strip())
    # cv2.imshow('test', img)
    # cv2.waitKey()
    # exit()
    # img = cv2.medianBlur(img,9)
    return pytesseract.image_to_string(img, config=options).strip()


def automate_is2_stage(device):
    # start journey 
    while True:
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
        
        # restart if this existed
        # tag = [546,543,873,603]
        if find_text_from_image(do_screenshot(device), 'Gaul Mantle', {'x': 500, 'y': 500}):
            print('[WARNING] Gaul Mantle Detected!')
            device.shell("input keyevent 4")
            # tag = [62,48]
            # device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            sleep(0.5)
            tag = [1747,498]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            # checkmark
            while True:
                resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_02.png'), score=0.9)
                if resp != []:
                    x,y = resp[0]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    print('[WARNING] Sleeping for 55s')
                    sleep(55.5)
                    break
        else:
            break
    
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
    while True:
        if iter > 2:
            tag = [
                [1700,300],
                [1500,300],
            ]
            device.shell("input touchscreen swipe %d %d %d %d 100" % (tag[0][0],tag[0][1], tag[1][0],tag[1][1]))
            print('[ERROR] squad is not found! we swiping to the left ...')
            sleep(0.5)
          
        resp = find_text_from_image(do_screenshot(device), 'Tactical Assault', {'x': 1000, 'y': 600})
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            sleep(0.5)
            resp = find_text_from_image(do_screenshot(device), 'Are you sure', {'x': 1000, 'y': 800})
            device.shell("input touchscreen tap {0} {1}".format(resp[0], resp[1]))
            break
        else:
            print('[ERROR] squad is not found!')
            iter += 1
            continue
    
    # select recruitment set
    while True:    
        resp = find_text_from_image(do_screenshot(device), 'Overcoming Your', {'x': 1000, 'y': 600})
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            sleep(0.5)
            resp = find_text_from_image(do_screenshot(device), 'Are you sure', {'x': 1000, 'y': 0})
            device.shell("input touchscreen tap {0} {1}".format(resp[0], resp[1]))
            break
        else:
            print('[ERROR] recruitment set is not found!')
            continue
        
    # initial recruitment
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
            
        if voucher == 'Guard':           
            # select Mountain
            # tag = [1400,620]
            # device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            while True:
                    resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'select_Mountain.png'), score=0.98)
                    if resp:
                        x,y = resp[0]
                        device.shell("input touchscreen tap {0} {1}".format(x,y))
                        break
            
            # confirm
            while True:    
                resp = find_text_from_image(do_screenshot(device), f'Confirm', {'x':1500, 'y': 900})
                if resp:
                    device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                    sleep(4.2)
                    device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                    break
                else:
                    continue
        else:
            # select Abandon
            tag = [1440,1000]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            # checkmark
            while True:
                resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_02.png'), score=0.98)
                if resp != []:
                    x,y = resp[0]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    break
    
    # Enter Journey
    while True:    
        resp = find_text_from_image(do_screenshot(device), 'ENTER', {'x': 1600, 'y': 430})
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            break
       
    ### PHASE 2 ###    
    # setup squad
    while True:
        resp = find_text_from_image(do_screenshot(device), 'Squad')
        if resp:
            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
            sleep(1)
            break

    # i choose you, Mountain
    tag = [353,262]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    sleep(0.8)
    print('[DONE] 1')

    tag = [999,222]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    sleep(0.1)
    print('[DONE] 2')

    tag = [360,840]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    sleep(0.1)
    print('[DONE] 3')
    
    tag = [1711,1017]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    sleep(0.8)
    print('[DONE] 4')

    tag = [148,48]
    device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
    print('[DONE] 5')
    sleep(1)
    
def check_done_explore(device):
    # wait until end stage
    while True:
        img_src = do_screenshot(device)
        resp = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cleared_battle.png'), score=0.95)
        if resp != []:
            print('[DONE] battle cleared successfully')
            break
        else:
            # tag = [827,320,1392,423]
            if find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'signal_lost.png'), score=0.95):
                print('[WARNING] run stage failed !!!')
                tag = [940,970]
                # [device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1])) for i in range(15)]
                check_exit(device, forced=True)
                print('die')
                # re - enter
                automate_is2_stage(device)
                return False
    
    iter = -5 
    while True:
        tag = [120,820]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        if iter == 5:
            resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cross_abandon.png'), score=0.98)
            sleep(0.5)
            if resp != []:
                # x,y = resp[0]
                # device.shell("input touchscreen tap {0} {1}".format(x,y))
                # # checkmark
                # while True:
                #     resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'check_mark_02.png'), score=0.98)
                #     if resp != []:
                #         x,y = resp[0]
                #         device.shell("input touchscreen tap {0} {1}".format(x,y))
                #         break
                
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
    stage_metadata = {}
    iter = []
    while True:
        img_src = do_screenshot(device)
        resp_rogue_trader = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rogue_trader.png'), score=0.99)
        if resp_rogue_trader:
            print('[INFO] Rogue Trader slots fuckk')
            x,y=resp_rogue_trader[0]
            device.shell("input touchscreen tap {0} {1}".format(x,y))
            sleep(1)
            tag = [1714,735]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            sleep(1.5)
            return True
        resp_encounter = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_encounter.png'), score=0.99)
        resp_combat = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_combat.png'), score=0.99)
        resp_recreation = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_recreation.png'), score=0.99)
        resp_emergency = find_image(img_src, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stage_emergency.png'), score=0.99)
        resp = [resp_encounter, resp_combat, resp_recreation, resp_emergency]
        if resp == [[], [], [], []]:
            continue
        else:
            iter.append(resp)

        # max loop
        if len(iter) > 3:
            break
        
    len_iter = [(len(i[0]), len(i[1]), len(i[2])) for i in iter]
    # https://www.geeksforgeeks.org/python-get-duplicate-tuples-from-list/
    resp = [x for x,y in enumerate(len_iter) if y == [ele for ele, count in Counter(len_iter).items() if count > 1][0]]
    print(f'[DEBUG] {(iter)}')
    print(f'[DEBUG] {(resp)}')
    print(f'[DEBUG] {(len_iter)}')
    print(f'[DEBUG] {(iter[resp[0]])}')
    
    # remove duplicate tuples cordinate
    resp = iter[resp[0]]
    for idata in [x for x in range(len(resp)) if len(resp[x]) > 1]:
        result = []
        for i in resp[idata]:
            print(f'[DEBUG] {(0,i,resp[idata])}')
            if len(result) > 0:
                duplicate = False
                for j in result:
                    if (i[0] == j[0] or i[1] == j[1]) and (abs(i[0]-j[0]) < 10 and abs(i[1]-j[1]) < 10):
                        duplicate = True
                        break
                if duplicate:
                    continue
            result.append(i)
        resp[idata] = result
    print(f'[DEBUG] {(resp)}')
                
    stage_metadata = {
        'stage_encounter' : {
            'total': len(resp[0]),
            'cordinate': resp[0]
        },
        'stage_combat' : {
            'total': len(resp[1]),
            'cordinate': resp[1]
        },
        'stage_recreation' : {
            'total': len(resp[2]),
            'cordinate': resp[2]
        },
        'stage_emergency' : {
            'total': len(resp[3]),
            'cordinate': resp[3]
        }
    }
    
    # select all type of open stage            
    print(f'[DEBUG] {(stage_metadata)}')
    choose_stage = [key for key in stage_metadata if stage_metadata[key]['total'] != 0]
    choose_stage = choose_stage[randint(0, len(choose_stage)-1)]
    print(f'[INFO] select {choose_stage} lol')
    
    # select random lol
    x,y=stage_metadata[choose_stage]['cordinate'][randint(0, stage_metadata[choose_stage]['total']-1)]
    device.shell("input touchscreen tap {0} {1}".format(x,y))
    sleep(0.8)
    
    if choose_stage == 'stage_encounter' or choose_stage == 'stage_recreation':
        # get hope
        # tag = {
        #     'top': (screen_y-1050),
        #     'bottom': (screen_y-1005),
        #     'left': (screen_x-590),
        #     'right': (screen_x-450)
        # }
        tag = {
            'top': (screen_y-1050),
            'bottom': (screen_y-1000),
            'left': (screen_x-552),
            'right': (screen_x-450)
        }
        hope = crop_read_string(device,tag,thresh_num=180,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
        if not hope.isnumeric():
            print(f'[ERROR] HOPE : {hope}')
            return True
        else:
            hope = int(hope)
            print(f'[INFO] total HOPE : {hope}')
        
        tag = [1714,735]
        device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        tag = [940,940]
        for i in range(5):
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
        
        # get encounter name
        tag = [166,700,600,950]
        tag = {
            'top': tag[1],
            'bottom': tag[3],
            'left': tag[0],
            'right': tag[2]
        }
        str_res = ''
        while True:
            str_res = crop_read_string(device, tag, thresh_num=100, options=r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-? "').split('\n')
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
        
        # work on here
        if str_res == 'Reprieve?':
            for x,y,t in metadata_encounter[str_res][1 if hope > 0 else 0]:
                print([1 if hope > 0 else 0])
                device.shell("input touchscreen tap {0} {1}".format(x,y))
                sleep(t)
        else:
            for x,y,t in metadata_encounter[str_res]:
                device.shell("input touchscreen tap {0} {1}".format(x,y))
                sleep(t)
            
    elif choose_stage == 'stage_combat' or choose_stage == 'stage_emergency':
        if choose_stage == 'stage_combat':
            tag = [1333,255,1585,315]
        else:
            tag = [1333,210,1585,315]
        tag = {
            'top': tag[1],
            'bottom': tag[3],
            'left': tag[0],
            'right': tag[2]
        }
        str_res = crop_read_string(device, tag, thresh_num=100, options=r'--psm 7 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "')
        if str_res in metadata_combat:
            print(f'[INFO] found "{str_res}"')

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
            
            # get Mountain position to deploy
            tag_deploy_x =0
            tag_deploy_y =0
            while True:
                    resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'deploy_Mountain.png'), score=0.95)
                    if resp:
                        # print('--->', resp)
                        tag_deploy_x,tag_deploy_y = resp[0]
                        break
                    
            # 2X Speed
            tag = [1650,75]
            device.shell("input touchscreen tap {0} {1}".format(tag[0], tag[1]))
            sleep(3)
            # deploy Mountain
            device.shell("input touchscreen swipe %d %d %d %d 1000" % (tag_deploy_x, tag_deploy_y, metadata_combat[str_res][0][0] ,metadata_combat[str_res][0][1]))
            sleep(0.5)
            device.shell("input touchscreen swipe %d %d %d %d 500" % (metadata_combat[str_res][1][0], metadata_combat[str_res][1][1], metadata_combat[str_res][1][2] ,metadata_combat[str_res][1][3]))
            # wait Mountain skill sp
            sleep(3) 
            # active Mountain skill 2
            device.shell("input touchscreen tap %d %d" % (metadata_combat[str_res][2][0], metadata_combat[str_res][2][1]))
            sleep(0.5)
            device.shell("input touchscreen tap %d %d" % (metadata_combat[str_res][2][2] ,metadata_combat[str_res][2][3]))
            sleep(0.5)
            
            # collect all collectible item then continue ...
            check_done_explore(device)
    return False

def automate_rogue_trader(device):
    global total_io_farm
    
    if True:
        tag = {
            'top': (screen_y-1050),
            'bottom': (screen_y-1010),
            'left': (screen_x-120),
            'right': (screen_x-30)
        }
        img_data = do_screenshot(device)
        for i in [100, 150, 180, 200]:
            originium_ingot = crop_read_string(img_data,tag,thresh_num=i,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
            if originium_ingot.isnumeric():
                print(f'[INFO] total OI : {originium_ingot}')
                break
        originium_ingot = int(originium_ingot)
        # [r'--psm 7 -c tessedit_char_whitelist="0123456789"', r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. "'])
        resp = find_text_from_image(do_screenshot(device), '', {'x': 500, 'y':140},full_result=True,thresh_num=180, options=["--psm 10 -c tessedit_char_whitelist=\"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-. \"", "-c tessedit_char_whitelist=\"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'-. \""])
        print(f'[DEBUG]\n {resp}')

        # get first row item
        result = {'first_row':[]}
        tmp_result = []
        for data in resp:
            x,y,w,h = data[1]
            if (x > 550 and x < 600) and (y > 140 and y < 150):
                tmp_result.append([data[0],x+50,y+50,w,h])   
            elif (x > 600 and x < 1830) and (y > 140 and y < 200):
                tmp_result.append([data[0],x,y,w,h])
        # sort by x
        tmp_result.sort(key=lambda x: x[1])
        print(f'[DEBUG]\n {tmp_result}')
        for enum in tmp_result:
            name,x,y,w,h = enum
            name = ' '.join( [w for w in name.split() if w not in ascii_letters and len(w) != 2] )
            result['first_row'].append({
                name: {
                    'price': 0,
                    'cordinate': [x,y,w,h]
                }
            })
        # get price
        tmp_result = []
        tmp_x = 0
        for data in resp:
            x,y,w,h = data[1]
            if (x > 600 and x < 1830) and (y > 300 and y < 440):
                if data[0].isnumeric():
                    # this check is for big text number on prospect investment box
                    if tmp_x != 0:
                        if abs(tmp_x-x) < 150:
                            print(data[0])
                            continue
                    tmp_x = x
                    tmp_result.append([data[0],x,y,w,h])
        #　check if price is not wrong at all, lol
        if len(tmp_result) < len(result['first_row']):
            print('[WARNING] low OI detected 1')
            z = []
            z_check = [(600, 905), (920, 1215), (1230, 1520), (1540, 1830)][:len(result['first_row'])]
            for iz in range(len(z_check)):
                x1,x2 = z_check[iz]
                resp_idata = [idata for idata in tmp_result if (idata[1] > x1 and idata[1] < x2)]
                if resp_idata:
                    z.append(resp_idata[0])
                else:
                    z.append(['0', x1, 0, 0, 0])
            tmp_result = z
        # sort by x
        tmp_result.sort(key=lambda x: x[1])
        print(f'[DEBUG]\n {tmp_result}')
        for enum in enumerate(tmp_result):
            name,x,y,w,h = enum[1]
            # print(enum[0])
            result['first_row'][enum[0]][[x for x in result['first_row'][enum[0]]][0]].update(
            {
                'price': 1337 if [x for x in result['first_row'][enum[0]]][0] == 'Prospective Investment System' else int(name)
            })
                        

        # get second row item
        result.update({'second_row':[]})
        tmp_result = []
        for data in resp:
            x,y,w,h = data[1]
            if (x > 550 and x < 600) and (y > 400 and y < 460):
                tmp_result.append([data[0],x+50,y+50,w,h])
            elif (x > 600 and x < 1830) and (y > 400 and y < 500):
                tmp_result.append([data[0],x,y,w,h])
        # sort by x
        tmp_result.sort(key=lambda x: x[1])
        print(f'[DEBUG]\n {tmp_result}')
        for enum in tmp_result:
            name,x,y,w,h = enum
            name = ' '.join( [w for w in name.split() if w not in ascii_letters and len(w) != 2] )
            num = 0
            # check last box on second row if exist
            if x > 1540:
                device.shell("input touchscreen tap {0} {1}".format(x,y))
                sleep(0.8)
                tag = [874,634,1513,686]  
                tag = {
                    'top': tag[1],
                    'bottom': tag[3],
                    'left': tag[0],
                    'right': tag[2]
                }
                str_res = crop_read_string(device,tag,thresh_num=170,options=r'--psm 10 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrtuvwxy? "')
                device.shell("input touchscreen tap {0} {1}".format(randint(680, 1200),randint(715, 770)))
                print(str_res)
                sleep(0.1)
                try:
                    str_res = str_res.split(' to')[0].split(' ')[1]
                    # https://stackoverflow.com/a/51312242
                    str_res = ''.join([n for n in str_res if n.isdigit()])
                except:
                    print(f'[DEBUG] Exception reading number: {str_res}')
                    # input()
                    str_res='0'

                if str_res.isnumeric():
                    num = int(str_res)
                else:
                    print('[DEBUG] Error reading last number lol')
                    exit()
            result['second_row'].append({
                name: {
                    'price': num,
                    'cordinate': [x,y,w,h]
                }
            })
        #　check if price is not wrong at all, lol
        if len(tmp_result) < len(result['second_row']):
            print('[WARNING] low OI detected 2')
            z = []
            z_check = [(600, 905), (920, 1215), (1230, 1520), (1540, 1830)][:len(result['second_row'])]
            for iz in range(len(z_check)):
                x1,x2 = z_check[iz]
                resp_idata = [idata for idata in tmp_result if (idata[1] > x1 and idata[1] < x2)]
                if resp_idata:
                    z.append(resp_idata[0])
                else:
                    z.append(['0', x1, 0, 0, 0])
            tmp_result = z
        # get price
        tmp_result = []
        for data in resp:
            x,y,w,h = data[1]
            if (x > 600 and x < 1535) and (y > 670 and y < 760):
                if data[0].isnumeric():
                    tmp_result.append([data[0],x,y,w,h])
        # sort by x
        tmp_result.sort(key=lambda x: x[1])
        print(f'[DEBUG]\n {tmp_result}')
        for enum in enumerate(tmp_result):
            name,x,y,w,h = enum[1]
            result['second_row'][enum[0]][[x for x in result['second_row'][enum[0]]][0]].update(
            {
                'price': int(name)
            })

        print(f'[DEBUG]\n {result}')
        print(f'[INFO] total OI : {originium_ingot}')

        # check if investment system exists
        if 'Prospective Investment System' in result['first_row'][0] and False:
            print('[INFO] Investment exists!')

            # automate invest lul
            x,y,w,h = result['first_row'][0]['Prospective Investment System']['cordinate']
            device.shell("input touchscreen tap {0} {1}".format(int(x+w/2), int(y+h/2)))
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
                    x,y = [randint(713, 1732), randint(717, 765)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(1)
                    break
                else:
                    # tap investment entrance
                    x,y = [randint(720, 1285), randint(445, 620)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(0.5)
                    # tap confirm
                    x,y = [randint(1220, 1730), randint(715, 770)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(0.8)
                    # tap nahh
                    x,y = [randint(680, 1200), randint(715, 770)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(2.5)
                    originium_ingot-=1
                    total+=1
                    print(f'[INFO] Good RNG ฅ(≈>ܫ<≈)ฅ')


        # no investment
        # merge first + second row
        x,y = (result['first_row'], result['second_row'])
        z = x.copy()
        [z.append(y[i]) for i in range(len(y))]
        
        # parsing all item in the shop
        unique_accessories = []
        support_item = []
        voucher_recruit = []
        for i,data in enumerate(z):
            if i != 0:
                if data[[v for v in data][0]]['price'] != 0:
                    if 'Rec. Voucher' in [v for v in data][0]:
                        voucher_recruit.append(data)
                    elif any([x in [v for v in data][0] for x in ['Support', ' - Advancement', '- Training']]):
                        support_item.append(data)
                    else:
                        unique_accessories.append(data)

        print(f'[DEBUG] Accessories :\n {unique_accessories}\n[DEBUG] Supportive :\n {support_item}\n[DEBUG] Voucher :\n {voucher_recruit}')        
        # get all accessories
        blacklist_item = ['Laughing Joker', 'Rusted Blade', 'Bend Spears']
        if unique_accessories:
            for item in unique_accessories:
                if originium_ingot >= item[[v for v in item][0]]['price'] and [v for v in item][0] not in blacklist_item:
                    print(f'[INFO][1] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
                    x,y,w,h = item[[v for v in item][0]]['cordinate']
                    device.shell("input touchscreen tap {0} {1}".format(int(x+w/2), int(y+h/2)))
                    sleep(1)
                    x,y = [randint(1230, 1730), randint(720, 760)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(1)
                    originium_ingot -= item[[v for v in item][0]]['price']
                    print(f'[INFO] total OI : {originium_ingot}')
        
        # get all supportive item
        if support_item:
            for item in support_item:
                # skip useless Training lol
                if '- Training' in [v for v in item][0]:
                    continue
                elif originium_ingot >= item[[v for v in item][0]]['price']:
                    print(f'[INFO][2] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
                                                
                    x,y,w,h = item[[v for v in item][0]]['cordinate']
                    device.shell("input touchscreen tap {0} {1}".format(int(x+w/2), int(y+h/2)))
                    sleep(1)
                    x,y = [randint(1230, 1730), randint(720, 760)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(1)
                    print(item[[v for v in item][0]])
                    originium_ingot -= item[[v for v in item][0]]['price']
                    print(f'[INFO] total OI : {originium_ingot}')
        
        
        # get all recruit voucher
        if voucher_recruit:
            for item in voucher_recruit:
                if originium_ingot >= item[[v for v in item][0]]['price']:
                    print(f'[INFO][3] Try to buy "{[v for v in item][0]}" for {item[[v for v in item][0]]["price"]} OI')
                    x,y,w,h = item[[v for v in item][0]]['cordinate']
                    device.shell("input touchscreen tap {0} {1}".format(int(x+w/2), int(y+h/2)))
                    sleep(1)
                    x,y = [randint(1230, 1730), randint(720, 760)]
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                    sleep(2.5)
                    
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
                        print(0,)  
                        resp = find_text_from_image(do_screenshot(device), f'Confirm', {'x':1500, 'y': 900})
                        if resp:
                            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                            sleep(4.2)
                            device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
                        else:
                            break
                    
    
        # input('sukebe')
        # exit
        sleep(0.5)

        # x,y = [randint(980, 1900), randint(700, 785)]
        # device.shell("input touchscreen tap {0} {1}".format(x,y))
        # sleep(1)
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
    tag = [940,970]
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

if __name__ == "__main__":
    adb = Client(host='127.0.0.1', port=5037)
    device = adb.devices()[0]
    
    # tag = [1618,942,1739,1036]    
    # tag = {
    #     'top': tag[1],
    #     'bottom': tag[3],
    #     'left': tag[0],
    #     'right': tag[2]
    # }
    # # crop_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
    # resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'deploy_Mountain.png'), score=0.95)
    # print(resp)
    # x,y = resp[0]
    # device.shell("input touchscreen tap {0} {1}".format(x,y))
    # automate_rogue_trader(device)
    while True:
        tstart=time()
        automate_is2_stage(device)
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
    
    # tag = {
    #     'top': (screen_y-1050),
    #     'bottom': (screen_y-1000),
    #     'left': (screen_x-552),
    #     'right': (screen_x-500)
    # }
    
    # hope = crop_read_string(device,tag,thresh_num=150,options=r'--psm 7 -c tessedit_char_whitelist=0123456789')
    # print(hope)

    # tag = [600,135,1850,780]    
    # tag = {
    #     'top': tag[1],
    #     'bottom': tag[3],
    #     'left': tag[0],
    #     'right': tag[2]
    # }
    # crop_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
    # resp = find_text_from_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), '',full_result=True,thresh_num=150, options=[r'--psm 7 -c tessedit_char_whitelist="0123456789"', r'-c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. "'])
    # print(resp)
    # do_screenshot(device)
    # crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
    # img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
    # _, img = cv2.threshold(img,100,255, cv2.THRESH_BINARY)
    # img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=3,fy=2))
    # img = cv2.GaussianBlur(img,(11,11),0)
    # # img = cv2.medianBlur(img,9)
    # str_res = pytesseract.image_to_string(img, config=r'--psm 7 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-? "').strip()
    # print(str_res)      
    # tag = [893,452,1073,525]
    # tag = {
    #     'top': tag[1],
    #     'bottom': tag[3],
    #     'left': tag[0],
    #     'right': tag[2]
    # }
    # do_screenshot(device)
    # crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)

    # while True:
    #     resp = find_image(do_screenshot(device), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rogue_trader.png'), score=0.99)
    #     print(len(resp), resp)
    # do_screenshot(device)
    # resp = find_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'rogue_trader.png'), score=0.99, gray_mode=True)
    # print(resp)
    # resp = [743, 421, 674, 107]
    # device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))


    # print(resp)
    # device.shell("input touchscreen tap {0} {1}".format(resp[0], int(resp[1]+resp[3]/2)))
