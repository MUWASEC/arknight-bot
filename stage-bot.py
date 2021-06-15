#!/usr/bin/env python3

from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
import cv2
from time import sleep, time
from datetime import datetime,timezone,timedelta


# global var
pytesseract.tesseract_cmd = '/usr/sbin/tesseract'
total_done=0
magic_xy = {
    'autodeploy_check' : {
        'x': 1899,
        'y': 891
    },
    'autodeploy_tap' : {
        'x': 1983,
        'y': 985
    },

    'mission_start_check' : {
        'x': 1707,
        'y': 66
    },
    'mission_start_tap' : {
        'x': 1800,
        'y': 800
    },

    'mission_complete_check' : {
        'x': 1400,
        'y': 730
    },
    'mission_complete_tap' : {
        'x': 1080,
        'y': 500
    },

    'sanity_check' : {
        'x': 205,
        'y': 590
    },

    'back_button': {
        'x': 113,
        'y': 60 
    },

    'restore_sanity_button': {
        'x': 1345,
        'y': 274
    },
    
    'confirm_restore_sanity_button': {
        'x': 1815,
        'y': 1030
    },
}
    # 1728, 270
    # 345, 996
    # 300, 496
    # 1561, 265


select_mission = {
    'tactical_drill_01': {
        'x': 1728,
        'y': 270
    },
    'tactical_drill_02': {
        'x': 345,
        'y': 996
    },
    'tactical_drill_03': {
        'x': 300,
        'y': 496
    },
    'tactical_drill_04': {
        'x': 1561,
        'y': 265
    },

    'monday-tough_siege_01': {
        'x': 1728,
        'y': 270
    },
    'monday-tough_siege_02': {
        'x': 345,
        'y': 996
    },
    'monday-tough_siege_03': {
        'x': 705,
        'y': 500
    },
    'monday-tough_siege_04': {
        'x': 1561,
        'y': 265
    },
}
select_mission_job = {
    'tactical_drill': 4,
    'monday-tough_siege': 4,
}

magic_rgb = {
    'autodeploy': {
        'r':255,
        'g':255,
        'b':255
    },
    
    'mission_start': {
        'r':0,
        'g':152,
        'b':220
    },

    'mission_complete': {
        'r':255,
        'g':150,
        'b':2
    },

    'sanity_check': {
        'r':176,
        'g':176,
        'b':176
    },
}


def do_screenshot(device):
    image = device.screencap()
    with open('screen.png', 'wb') as fd:
        fd.write(image)
    image = np.array(Image.open('screen.png'), dtype=np.uint8)
    return image

def get_sanity():
    custom_oem=r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/'
    # Load image
    image = Image.open('screen.png')
    image_arr = np.array(image)
    image_arr = image_arr[
        # y
        # top - bottom
        30:105, 
        # x
        # left - right
        1980:2155]
    image = Image.fromarray(image_arr)
    image.save('crop.png')

    total_res=[]
    try:
        # # default check (RGB)
        # img = cv2.imread('crop.png')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # str_res = pytesseract.image_to_string(img, config=custom_oem)
        # if '/' in str_res and str_res[0]!='/':
        #     raise Exception()
        # else:
        #     total_res.append(str_res)

        # # invert color check
        # img = cv2.imread('crop.png')
        # img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # str_res = pytesseract.image_to_string(img, config=custom_oem)
        # if '/' in str_res and str_res[0]!='/':
        #     raise Exception()
        # else:
        #     total_res.append(str_res)
        
        # # blur bitwise not
        # img = cv2.imread('crop.png')
        # retval, img = cv2.threshold(img,200,255, cv2.THRESH_BINARY)
        # img = cv2.bitwise_not(cv2.resize(img,(0,0),fx=3,fy=3))
        # str_res = pytesseract.image_to_string(img, config=custom_oem)
        # if '/' in str_res and str_res[0]!='/':
        #     raise Exception()
        # else:
        #     total_res.append(str_res)

        # brute fully blured bitwise not
        img = cv2.imread('crop.png')
        for i in [200, 190, 140, 120, 100]:
            retval, img = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=3,fy=2))
            img = cv2.GaussianBlur(img,(11,11),0)
            img = cv2.medianBlur(img,9)
            str_res = pytesseract.image_to_string(img, config=custom_oem)
            if '/' in str_res and str_res[0]!='/':
                print(f'complete brute force at {i}')
                raise Exception()
            
        img = cv2.imread('crop.png')
        for i in [200, 190, 140, 120, 100]:
            retval, img = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(cv2.resize(img,(0,0),fx=3,fy=3))
            img = cv2.GaussianBlur(img,(11,11),0)
            img = cv2.medianBlur(img,9)
            str_res = pytesseract.image_to_string(img, config=custom_oem)
            if '/' in str_res and str_res[0]!='/':
                print(f'complete brute force at {i}')
                raise Exception()

        # white pixels to black with GRAY
        # img = cv2.imread('crop.png')
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # img[thresh == 255] = 0
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # img = cv2.erode(img, kernel, iterations = 1)
        # str_res = pytesseract.image_to_string(img, config=custom_oem)
        # if '/' in str_res and str_res[0]!='/':
        #     raise Exception()
        # else:
        #     total_res.append(str_res)

        # return false
        return total_res
    except:
        cv2.imwrite('crop.png', img)
        return str_res
    
def restore_sanity(device, sanity, max_sanity, full=1):
    if full:
        # go back to the main menu
        iter=0
        while True:
            if iter>3:
                sleep(0.5)
                do_screenshot(device)
                image = Image.open('screen.png')
                image_arr = np.array(image)
                image_arr = image_arr[
                    # y
                    # top - bottom
                    450:535, 
                    # x
                    # left - right
                    862:1356]
                image = Image.fromarray(image_arr)
                image.save('crop.png')

                img = cv2.imread('./crop.png')
                device.shell("input keyevent 4")
                if pytesseract.image_to_string(img).split('\n')[0] == "Are you sure you want to exit?":
                    break
            else:
                device.shell("input keyevent 4")
                print(f'[{iter}] back button')
            iter+=1

        # main menu
        print(f'\n[!] do restore sanity, hope your orundum is enough :p\nsanity : {sanity}\nmax_sanity : {max_sanity}')
        for i in range(int((999+sanity)/max_sanity)):
            print('.', end='', flush=True)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['restore_sanity_button']['x'], magic_xy['restore_sanity_button']['y']))
            sleep(1)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['confirm_restore_sanity_button']['x'], magic_xy['confirm_restore_sanity_button']['y']))
            sleep(4.5)
        print('\n', end='', flush=True)

def bot_process(device, jobiter):
    global total_done
    tstart=0
    tend=0
    while True:
        # first stage
        while True:
            image = do_screenshot(device)
            r,g,b = [int(j) for j in [list(i[:3]) for i in image[magic_xy['autodeploy_check']['y']]][magic_xy['autodeploy_check']['x']]]
            if {'r': r, 'g': g, 'b': b} == magic_rgb['autodeploy']:
                sanity = get_sanity()
                if jobiter==total_done:
                    print(f'\n[!] {total_done} job is complete')
                    return False
                elif '/' not in sanity:
                    print(f'\n[!] ocr error :(\n{sanity}')
                    return False
                else:
                    max_sanity = int(sanity.split('/')[1])
                    sanity = int(sanity.split('/')[0])

                # output time spend & info
                tstart=time()
                if total_done:
                    print(f'\n[*]===============| time spend {int(tstart-tend)}s/{(int(tstart-tend)/60):.1f}m |===============[*]')
                    print(f'[+] {total_done} mission has been done')
                    print(f'[+] current sanity now is {sanity}\n')
                tend=time()
                total_done+=1

                device.shell("input touchscreen tap {0} {1}".format(magic_xy['autodeploy_tap']['x'], magic_xy['autodeploy_tap']['y']))
                break
            else:
                print(f'[{r},{g},{b}] autodeploy check not found')

        # second stage
        while True:
            image = do_screenshot(device)
            r,g,b = [int(j) for j in [list(i[:3]) for i in image[magic_xy['mission_start_check']['y']]][magic_xy['mission_start_check']['x']]]
            if {'r': r, 'g': g, 'b': b} == magic_rgb['mission_start']:
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_start_tap']['x'], magic_xy['mission_start_tap']['y']))
                break
            else:
                rs,gs,bs = [int(j) for j in [list(i[:3]) for i in image[magic_xy['sanity_check']['y']]][magic_xy['sanity_check']['x']]]
                if {'r': rs, 'g': gs, 'b': bs} == magic_rgb['sanity_check']:
                    print(f'[{rs},{gs},{bs}] sanity not enough :(')
                    restore_sanity(device, sanity, max_sanity)
                    return True
                print(f'[{r},{g},{b}] mission_start check not found')

        # third stage
        while True:
            image = do_screenshot(device)
            r,g,b = [int(j) for j in [list(i[:3]) for i in image[magic_xy['mission_complete_check']['y']]][magic_xy['mission_complete_check']['x']]]
            if {'r': r, 'g': g, 'b': b} == magic_rgb['mission_complete']:
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_complete_tap']['x'], magic_xy['mission_complete_tap']['y']))
                break
            else:
                if {'r': r, 'g': g, 'b': b} == {'r': 1, 'g': 1, 'b': 1}:
                    print(f'[{r},{g},{b}] level up')
                    device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_complete_tap']['x'], magic_xy['mission_complete_tap']['y']))
                else:
                    print(f'[{r},{g},{b}] mission_complete check not found')

def bot_select_mission(device, stage_name):
    if stage_name != 'tactical_drill':
        stage_name = '%s-%s' % (datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-7))).strftime('%A').lower(), stage_name)

    # must be on main menu
    for i in range(1,select_mission_job[stage_name]+1):
        device.shell("input touchscreen tap {0} {1}".format(select_mission[f'{stage_name}_0{i}']['x'], select_mission[f'{stage_name}_0{i}']['y']))
        sleep(1)

if __name__ == "__main__":
    adb = Client(host='127.0.0.1', port=5037)
    device = adb.devices()[0]#adb.device("192.168.18.182:5555")
    # image = do_screenshot(device)
    # r,g,b = [int(j) for j in [list(i[:3]) for i in image[magic_xy['mission_start_check']['y']]][magic_xy['mission_start_check']['x']]]
    # print(r,g,b)
    # exit()
    job=[
        'tactical_drill',
        'tough_siege'
    ]


    try:
        jobiter = int(input('~max_job = '))
    except:
        jobiter = 99

    # main
    while True:
        bot_select_mission(device, job[1])
        if not bot_process(device, jobiter):
            exit(0)