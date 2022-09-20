#!/usr/bin/env python3

import argparse, sys, os, cv2, base64
from random import randint
from cProfile import run
from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
from time import sleep, time
from datetime import datetime,timezone,timedelta


# global var
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
job_list=[]
total_done=0
restore_sanity=False
restore_sanity_check=False
all_opens=False
verbose=False
r,g,b = "","",""
screen_x = 1920
screen_y = 1080

recruitment_box_xy = {
    'box1': {
        'status': True,
        'x': (screen_x - 1432),
        'y': (screen_y - 645)
    },
    
    'box2': {
        'status': False,
        'x': (screen_x - 1432)+946,
        'y': (screen_y - 645)
    },
    
    'box3': {
        'status': True,
        'x': (screen_x - 1432),
        'y': (screen_y - 645)+418
    },
    
    'box4': {
        'status': False,
        'x': (screen_x - 1432)+946,
        'y': (screen_y - 645)+418
    }
}

job_tags_xy = {
    'box_top_1': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(0*251),
        'right':    778+(0*251)
    },
    
    'box_top_2': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(1*251),
        'right':    778+(1*251)
    },
    
    'box_top_3': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(2*251),
        'right':    778+(2*251)
    },
    
    'box_bot_1': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(0*251),
        'right':    778+(0*251)
    },
    
    'box_bot_2': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(1*251),
        'right':    778+(1*251)
    },
    
    'box_bot_3': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(2*251),
        'right':    778+(2*251)
    },
    
    # 'testing': {
    #     'top':      955,
    #     'bottom':   1023,
    #     'left':     625,
    #     'right':    780
    # }
    
}

helper_xy = {
    'confirm_button' : {
        'x': (screen_x - 480),
        'y': (screen_y - 320)
    },
    'back' : {
        'x': (screen_x - 1780),
        'y': (screen_y - 1020)
    }, 
    'recruit_button': {
        'x': (screen_x - 460),
        'y': (screen_y - 210)
    }, 
    'select_recruit_hour_time_down': {
        'x': (screen_x - 1245),
        'y': (screen_y - 636)
    }
}

recruitment_box_xy = {
    'box1': {
        'status': True,
        'x': (screen_x - 1432),
        'y': (screen_y - 645)
    },
    
    'box2': {
        'status': True,
        'x': (screen_x - 1432)+946,
        'y': (screen_y - 645)
    },
    
    'box3': {
        'status': True,
        'x': (screen_x - 1432),
        'y': (screen_y - 645)+418
    },
    
    'box4': {
        'status': True,
        'x': (screen_x - 1432)+946,
        'y': (screen_y - 645)+418
    }
}

job_tags_xy = {
    'box_top_1': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(0*251),
        'right':    778+(0*251)
    },
    
    'box_top_2': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(1*251),
        'right':    778+(1*251)
    },
    
    'box_top_3': {
        'top':      541+(0*107),
        'bottom':   610+(0*107),
        'left':     563+(2*251),
        'right':    778+(2*251)
    },
    
    'box_bot_1': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(0*251),
        'right':    778+(0*251)
    },
    
    'box_bot_2': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(1*251),
        'right':    778+(1*251)
    },
    
    'box_bot_3': {
        'top':      541+(1*107),
        'bottom':   610+(1*107),
        'left':     563+(2*251),
        'right':    778+(2*251)
    },    
}

magic_xy = {
    'autodeploy_check' : {
        'x': 1600,
        'y': 889
    },
    'autodeploy_tap' : {
        'x': 1700,
        'y': 988
    },

    'mission_start_check' : {
        'x': 1410,
        'y': 60
    },
    'mission_start_tap' : {
        'x': 1656,
        'y': 779
    },
    
    # this right here is even work, dont even touch it
    'mission_error_check' : {
        'x': 1400,
        'y': 730
    },
    'mission_complete_check' : {
        # 'x': 1400,
        # 'y': 730
        'x': 70,
        'y': 935
    },
    'mission_complete_tap' : {
        'x': 1080,
        'y': 500
    },

    'sanity_check' : {
        'x': 55,
        'y': 590
    },

    'back_button': {
        'x': 113,
        'y': 60 
    },

    'restore_sanity_onstage_tap': {
        'x': 1630,
        'y': 865
    },

    'restore_sanity_button': {
        'x': 1345,
        'y': 274
    },
    
    'confirm_restore_sanity_button': {
        'x': 1815,
        'y': 1030
    },

    'connection_error_tap': {
        'x': 1485,
        'y': 768
    },
    
    'skip_recruitment_tap': {
        'x': (screen_x - 85),
        'y': (screen_y - 1030)
    },
    'skip_recruitment_check': {
        'x': (screen_x - 340),
        'y': (screen_y - 1030)
    }
}

recruitment_tag_priority_exit = [
    "Senior Operator", "Top Operator",
    "Control", "Crowd-Control", 
    "Nuker", "Robot", "Summon"
]

recruitment_tag_priority = [
    "Specialist",
    "Debuff",
    "Fast-Redeploy",
    "Shift"    
]

recruitment_tag = [
    "Starter", "Senior Operator", "Top Operator",
    
    "Melee", "Ranged",
    
    "Caster", "Defender", "Guard", "Medic", "Sniper", "Specialist", "Supporter", "Vanguard",
    
    "AoE", "Control", "Crowd-Control", "Debuff",
    "Defense", "DP-Recovery", "OP-Recovery", "DPS", "Fast-Redeploy",
    "Healing", "Nuker", "Robot", "Shift",
    "Slow", "Starter", "Summon", "Support",
    "Survival"
]


def do_screenshot(device):
    image = device.screencap()
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), 'wb') as fd:
        fd.write(image)
    with Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png')) as fd:
        image = np.array(fd, dtype=np.uint8)
    return image

def crop_image(source_image, output_image, cordinate):
    # Load image
    with Image.open(source_image) as screen_image:
        # calculate crop
        image_arr = np.array(screen_image)
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

def find_image(haystack, needle, score=0.95):
    img_rgb = cv2.imread(haystack)
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

def calculate_offset(xy_loc):
    resp='''
{
    'top': (screen_y-%d),
    'bottom': (screen_y-%d),
    'left': (screen_x-%d),
    'right': (screen_x-%d)
}
    ''' % (screen_y-xy_loc['top'], screen_y-xy_loc['bottom'], screen_x-xy_loc['left'], screen_x-xy_loc['right'])
    print(resp)



def check_rgb(device, location, rgb={}, verbose=False):
    global r,g,b
    image = do_screenshot(device)
    resp_r,resp_g,resp_b = [int(j) for j in [list(i[:3]) for i in image[location['y']]][location['x']]]
    
    # just gather the rgb value from location
    if rgb == {}:
        return (resp_r,resp_g,resp_b)

    # if rgb still false, then continue to verbose -> check rgb exist/not
    elif verbose:
        #sys.stdout.write(f'[{r},{g},{b}] ')
        r = resp_r
        g = resp_g
        b = resp_b
    
    # check if rgb exist or not
    if {'r': resp_r, 'g': resp_g, 'b': resp_b} == rgb:
        return True
    else:
        return False

def automate_recruitment(device, runtime=1):
    custom_oem=r'--psm 7 -c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ- "'
    recruit_time = 9
    check_expedited_plan = True
    total_expedited_plan = 0
        
    ### DEBUG
    # each runtime, goes 4 block bar so 1x4
    for iter in range(runtime):
        for req_box_loc in recruitment_box_xy.keys():
            if not recruitment_box_xy[req_box_loc]['status']:
                continue
            
            # select box
            device.shell("input touchscreen tap {0} {1}".format(recruitment_box_xy[req_box_loc]['x'],recruitment_box_xy[req_box_loc]['y']))
            sleep(0.5)

            # get screen inside recruitment box
            do_screenshot(device)
            
            resp_tag_list = []
            for box_loc in job_tags_xy.keys():
                # skip box tag bottom 3 (because it's always empty)
                if box_loc == 'box_bot_3':
                    continue
                
                # get all box screen
                tag = job_tags_xy[box_loc]
                crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
                
                # read with tesseract
                str_res = ""
                invalid_tag = 0
                while True:
                    try:
                        # blured bitwise not
                        img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
                        _, img = cv2.threshold(img,165,255, cv2.THRESH_BINARY)
                        img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=3,fy=2))
                        img = cv2.GaussianBlur(img,(11,11),0)
                        img = cv2.medianBlur(img,9)
                        
                        # remove something unnecessary (false-positive "- "," -"," - " andy)
                        str_res = pytesseract.image_to_string(img, config=custom_oem).strip().replace('- ', '').replace(' -', '').replace(' - ', '')
                        # print(str_res,)
                        if str_res in recruitment_tag:
                            resp_tag_list.append(str_res)
                            break
                        else:
                            # refresh crop.png
                            do_screenshot(device)
                            crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
                            invalid_tag+=1
                            
                        # exit/return if found invalid tag
                        if invalid_tag > 3:
                            print(f'[-] tag invalid : {req_box_loc}|{box_loc}|{str_res}')
                            return False
                    except:
                        pass
                
                # check tags priority
                if str_res in recruitment_tag_priority:
                    print(f'[+] found tag priority : {str_res}')
                    # click tag priority
                    x,y=int((tag['left']+tag['right'])/2),int((tag['top']+tag['bottom'])/2)
                    device.shell("input touchscreen tap {0} {1}".format(x,y))
                elif str_res in recruitment_tag_priority_exit:
                    print(f'[+] found tag priority exit : {str_res}')
                    exit()
                # resp_tag_list.append(str_res)
            
            # check any invalid data           
            if len(resp_tag_list) < 5:
                print(f'\n[ERROR] {box_loc} only found {len(resp_tag_list)}')
                exit()
            elif any(data in recruitment_tag_priority for data in resp_tag_list):
                print(f"\n[WARNING] found {resp_tag_list} on recruitment_tag_priority")
                exit()
            else:
                print(f'{req_box_loc}|{box_loc}|{resp_tag_list}')
            
            # set recruitment time
            for i in range(9+1, recruit_time, -1):
                device.shell("input touchscreen tap {0} {1}".format(helper_xy['select_recruit_hour_time_down']['x'],helper_xy['select_recruit_hour_time_down']['y']))
                sleep(0.5)
            
            # click recruitment button
            device.shell("input touchscreen tap {0} {1}".format(helper_xy['recruit_button']['x'],helper_xy['recruit_button']['y']))
            sleep(1)
            
            
        ### skip with red card
        # tag = {
        # 'top':      525,
        # 'bottom':   615,
        # 'left':     550,
        # 'right':    870
        # }
        # tag = {
        # 'top':      530,
        # 'bottom':   615,
        # 'left':     445,
        # 'right':    530
        # }
        # crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hire_btn.png'), tag)      
        
        # image recognition (recruit button)
        sleep(1)
        do_screenshot(device)
        tag_list = []
        for x,y in find_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'recruit_btn.png'), score=0.97):
            # print(x,y)
            tag_list.append({
                'x': int(x+randint(100, 150)), 
                'y': int(y+randint(15, 65))
            })
        # do func
        tag_list.sort(key=lambda x: x['x'])
        print(tag_list)
        for i in range(len([req_box_loc for req_box_loc in recruitment_box_xy.keys() if recruitment_box_xy[req_box_loc]['status']])):
            # touch recruit button
            device.shell("input touchscreen tap {0} {1}".format(tag_list[i]['x'], tag_list[i]['y']))
            sleep(1)
            
            # get expedited plan with ocr            
            if check_expedited_plan:
                do_screenshot(device)
                # calculate_offset(job_tags_xy['testing'])
                tag = {
                    'top': screen_y-536,
                    'bottom': screen_y-495,
                    'left': screen_x-1171,
                    'right': screen_x-1122
                }
                crop_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), tag)
                img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
                _, img = cv2.threshold(img,100,255, cv2.THRESH_BINARY)
                img = cv2.bitwise_not(cv2.resize(img,(0,0),fx=3,fy=3))
                img = cv2.GaussianBlur(img,(11,11),0)
                img = cv2.medianBlur(img,9)
                str_res = pytesseract.image_to_string(img, config=r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789').strip()
                if str_res.isnumeric():
                    check_expedited_plan = False
                    total_expedited_plan = int(str_res)
                else:
                    print("[ERROR] when reading expedited plan -_-")
                    return
            elif total_expedited_plan < 1:
                print("[WARNING] expedited plan is not enough")
                return
            
            # click confirm button
            device.shell("input touchscreen tap {0} {1}".format(helper_xy[f'confirm_button']['x'],helper_xy[f'confirm_button']['y']))
            total_expedited_plan-=1
            print(f'[INFO] total expedited plan = {total_expedited_plan}')
            sleep(1)
            
            
        # image recognition (hire button)
        sleep(1)
        do_screenshot(device)
        tag_list = []
        for x,y in find_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hire_btn.png'), score=0.95):
            print(x,y)
            tag_list.append({
                'x': int(x+randint(0, 110)), 
                'y': int(y+randint(10, 80))
            })
        # do func
        tag_list.sort(key=lambda x: x['x'])
        print(tag_list)
        for i in range(len([req_box_loc for req_box_loc in recruitment_box_xy.keys() if recruitment_box_xy[req_box_loc]['status']])):
            # touch hire button
            device.shell("input touchscreen tap {0} {1}".format(tag_list[i]['x'], tag_list[i]['y']))
            sleep(1)
            
            # speeding up operator talking, especially you Kross !!!
            while True:
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['skip_recruitment_tap']['x'],magic_xy['skip_recruitment_tap']['y']))
                sleep(0.3)
                print('.', end='', flush=True)
                if check_rgb(device, {
                                    'x': magic_xy['skip_recruitment_check']['x'], 
                                    'y': magic_xy['skip_recruitment_check']['y']
                                    }, {
                                        'r': 100,
                                        'g': 18,
                                        'b': 18
                                    }):
                    print()
                    break
        

if __name__ == "__main__":
    adb = Client(host='127.0.0.1', port=5037)
    device = adb.devices()[0]
    automate_recruitment(device)
    
    