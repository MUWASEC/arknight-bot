#!/usr/bin/env python3

from ppadb.client import Client
from PIL import Image
import numpy as np
from pytesseract import pytesseract
import cv2
from time import sleep, time
from datetime import datetime,timezone,timedelta
import argparse


# global var
pytesseract.tesseract_cmd = '/usr/sbin/tesseract'
total_done=0
restore_sanity=False
restore_sanity_check=False
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

mission_stage_metadata = {
    'supplies': {
        'offset': {
            # select & open stage
            'open_combat': {
                'x': 1728,
                'y': 270
            },
            'select_stage': {
                'x': 345,
                'y': 996
            },

            # select operation number
            'operation_1': {
                'x': 553,
                'y': 860
            },
            'operation_2': {
                'x': 974,
                'y': 789
            },
            'operation_3': {
                'x': 1272,
                'y': 609
            },
            'operation_4': {
                'x': 1525,
                'y': 436
            },
            'operation_5': {
                'x': 1561,
                'y': 265
            },
        },

        # supplies stage selector
        'LS': {
            'schedule': {
                'Mon': {
                    'x': 300,
                    'y': 496
                },
                'Tue': {
                    'x': 300,
                    'y': 496
                },
                'Wed': {
                    'x': 300,
                    'y': 496
                },
                'Thu': {
                    'x': 300,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                }
            }
            
        },
        'CA': {
            'schedule': {
                'Tue': {
                    'x': 710,
                    'y': 496
                },
                'Wed': {
                    'x': 300,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        'CE': {
            'schedule': {
                'Tue': {
                    'x': 1100,
                    'y': 496
                },
                'Thu': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                } 
            }
        },
        'tough_siege': {
            'schedule': {
                'Mon': {
                    'x': 710,
                    'y': 496
                },
                'Thu': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        'resource_research': {
            'schedule': {
                'Mon': {
                    'x': 1100,
                    'y': 496
                },
                'Wed': {
                    'x': 300,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                }
            }
        }
    },

    'chips': {
        'offset': {
            # select & open stage
            'open_combat': {
                'x': 1728,
                'y': 270
            },
            'select_stage': {
                'x': 576,
                'y': 996
            },

            # select operation number
            'operation_1': {
                'x': 700,
                'y': 665
            },
            'operation_2': {
                'x': 1420,
                'y': 370
            }
        },

        # chips stage selector
        'PR-A': {
            'schedule': {
                'Mon': {
                    'x': 505,
                    'y': 496
                },
                'Thu': {
                    'x': 300,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        'PR-B': {
            'schedule': {
                'Mon': {
                    'x': 910,
                    'y': 496
                },
                'Tue': {
                    'x': 505,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        'PR-C': {
            'schedule': {
                'Mon': {
                    'x': 910,
                    'y': 496
                },
                'Tue': {
                    'x': 505,
                    'y': 496
                },
                'Fri': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        'PR-D': {
            'schedule': {
                'Tue': {
                    'x': 910,
                    'y': 496
                },
                'Wed': {
                    'x': 300,
                    'y': 496
                },
                'Sat': {
                    'x': 300,
                    'y': 496
                },
                'Sun': {
                    'x': 300,
                    'y': 496
                }
            }
        },
        
    }
}

#115,20,20 = error
#15,15,15 = update
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

#https://stackoverflow.com/a/29669787/13734176
def find_image(im, tpl):
    im = np.atleast_3d(im)
    tpl = np.atleast_3d(tpl)
    H, W, D = im.shape[:3]
    h, w = tpl.shape[:2]

    # Integral image and template sum per channel
    sat = im.cumsum(1).cumsum(0)
    tplsum = np.array([tpl[:, :, i].sum() for i in range(D)])

    # Calculate lookup table for all the possible windows
    iA, iB, iC, iD = sat[:-h, :-w], sat[:-h, w:], sat[h:, :-w], sat[h:, w:] 
    lookup = iD - iB - iC + iA
    # Possible matches
    possible_match = np.where(np.logical_and.reduce([lookup[..., i] == tplsum[i] for i in range(D)]))

    # Find exact match
    for y, x in zip(*possible_match):
        if np.all(im[y+1:y+h+1, x+1:x+w+1] == tpl):
            return f'{y+1}|{x+1}'

    return False
    
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
                print(f'\n[*] complete brute force at {i}\n')
                raise Exception()
            
        img = cv2.imread('crop.png')
        for i in [200, 190, 140, 120, 100]:
            retval, img = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(cv2.resize(img,(0,0),fx=3,fy=3))
            img = cv2.GaussianBlur(img,(11,11),0)
            img = cv2.medianBlur(img,9)
            str_res = pytesseract.image_to_string(img, config=custom_oem)
            if '/' in str_res and str_res[0]!='/':
                print(f'\n[*] complete brute force at {i}\n')
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

def return_to_main_menu(device):
    print('\n[!] going back to the main menu ...')
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

def do_restore_sanity(device, sanity, max_sanity, full=1):
    global restore_sanity_check
    if full:
        # go back to the main menu
        return_to_main_menu(device)

        # main menu
        print(f'\n[!] do restore sanity, hope your orundum is enough :p\nsanity : {sanity}\nmax_sanity : {max_sanity}')
        for i in range(int((131+sanity)/max_sanity)):
            print('.', end='', flush=True)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['restore_sanity_button']['x'], magic_xy['restore_sanity_button']['y']))
            sleep(1)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['confirm_restore_sanity_button']['x'], magic_xy['confirm_restore_sanity_button']['y']))
            sleep(4.5)
            sanity+=max_sanity
        restore_sanity_check=True
        print('\n', end='', flush=True)
        print(f'\n[!!!] sanity has been restored\nsanity : {sanity}\n')


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
                    return True
                elif '/' not in sanity:
                    print(f'\n[!] ocr error :(\n{sanity}')
                    return False
                else:
                    max_sanity = int(sanity.split('/')[1])
                    sanity = int(sanity.split('/')[0])

                # output time spend & info
                tstart=time()
                if tend:
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
                    print(f'[{rs},{gs},{bs}] sanity not enough :(\n')
                    if restore_sanity:
                        do_restore_sanity(device, sanity, max_sanity)
                        return True
                    else:
                        return False
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

def get_stage_group_name(stage_name):
    for mission_key in mission_stage_metadata:
        if stage_name in mission_stage_metadata[mission_key]:
            return mission_key
    return False



def bot_select_mission(device, stage_name):
    ops_name = stage_name[::-1].split('-', 1)[1][::-1]
    ops_num = stage_name[::-1].split('-', 1)[0][::-1]
    stage_group = get_stage_group_name(ops_name)
    if not stage_group:
        print('[!] record for stage name not found!')
        exit(1)

    date_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-11))).strftime('%a')
    if date_now not in mission_stage_metadata[stage_group][ops_name]['schedule']:
        print(f'[!] {stage_name} currently unavailable.')
        return False

    # must be on main menu
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset']['open_combat']['x'], mission_stage_metadata[stage_group]['offset']['open_combat']['y']))
    sleep(1)
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset']['select_stage']['x'], mission_stage_metadata[stage_group]['offset']['select_stage']['y']))
    sleep(1)
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group][ops_name]['schedule'][date_now]['x'], mission_stage_metadata[stage_group][ops_name]['schedule'][date_now]['y']))
    sleep(1)
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset'][f'operation_{ops_num}']['x'], mission_stage_metadata[stage_group]['offset'][f'operation_{ops_num}']['y']))
    sleep(1)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arknight Stage Bot')
    parser.add_argument('-j', '--job-list', dest='job_list', help='xx: -j/--job-list "CE-5:20;CA-5:30"')
    parser.add_argument('-a', '--automate', type=int, dest='only_automate', help='just do automation, select the stage to automate first!')
    parser.add_argument('-s', '--sanity', action='store_true', dest='sanity_restore', help='with sanity restoration.')
    args = parser.parse_args()

    if not args.job_list and not args.only_automate:
        parser.print_help()
        exit(1)
    else:
        adb = Client(host='127.0.0.1', port=5037)
        device = adb.devices()[0]
        
    # main
    if args.sanity_restore:
        restore_sanity=True
    if args.job_list:
        for x in args.job_list.split(';'):
            job_name = x.split(':')[0]
            job_count = int(x.split(':')[1])
            while True:
                if not bot_select_mission(device, job_name):
                    break
                if not bot_process(device, job_count):
                    exit(0)
                if total_done >= job_count:
                    total_done = 0
                    break
            if not restore_sanity_check:
                restore_sanity_check=False
                return_to_main_menu(device)
    elif args.only_automate:
        job_count = args.only_automate
        bot_process(device, job_count)