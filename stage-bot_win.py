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
    }, 
    'box1_recruit_skip_button': {
        'x': (screen_x - 1205),
        'y': (screen_y - 510)
    }
}

recruitment_box_xy = {
    'box1': {
        'status': False,
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
    
    # 'testing': {
    #     'top':      955,
    #     'bottom':   1023,
    #     'left':     625,
    #     'right':    780
    # }
    
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
        # 'r':255,
        # 'g':150,
        # 'b':2
        'r':255,
        'g':255,
        'b':255
    },

    'sanity_check': {
        'r':176,
        'g':176,
        'b':176
    },
}

mission_stage_metadata = {
    'supplies': {
        'total_stage': 5,
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
        # 300,710,1120,1530,1940
        # specially open: LS, CA, SK, AP, CE
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
                    'x': 710,
                    'y': 496
                },
                'Fri': {
                    'x': 710,
                    'y': 496
                },
                'Sun': {
                    'x': 1120,
                    'y': 496
                }
            }
        },
        'CE': {
            'schedule': {
                'Tue': {
                    'x': 1120,
                    'y': 496
                },
                'Thu': {
                    'x': 1120,
                    'y': 496
                },
                'Sat': {
                    'x': 1530,
                    'y': 496
                },
                'Sun': {
                    'x': 1530,
                    'y': 496
                } 
            }
        },
        'AP': {
            'schedule': {
                'Mon': {
                    'x': 710,
                    'y': 496
                },
                'Thu': {
                    'x': 710,
                    'y': 496
                },
                'Sat': {
                    'x': 710,
                    'y': 496
                },
                'Sun': {
                    'x': 710,
                    'y': 496
                }
            }
        },
        'SK': {
            'schedule': {
                'Mon': {
                    'x': 1120,
                    'y': 496
                },
                'Wed': {
                    'x': 300,
                    'y': 496
                },
                'Fri': {
                    'x': 1120,
                    'y': 496
                },
                'Sat': {
                    'x': 1120,
                    'y': 496
                }
            }
        },
        'specially_open': {
            'LS': {
                'x': 300,
                'y': 496
            },
            'CA': {
                'x': 710,
                'y': 496
            },
            'SK': {
                'x': 1120,
                'y': 496
            },
            'AP': {
                'x': 1530,
                'y': 496
            },
            'CE': {
                'x': 1940,
                'y': 496
            }
        }
    },

    'chips': {
        'total_stage': 2,
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
        # 505,910,1315,1720
        # specially open: PR-C, PR-D, PR-A, PR-B
        'PR-A': {
            'schedule': {
                'Mon': {
                    'x': 505,
                    'y': 496
                },
                'Thu': {
                    'x': 505,
                    'y': 496
                },
                'Fri': {
                    'x': 505,
                    'y': 496
                },
                'Sun': {
                    'x': 505,
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
                    'x': 910,
                    'y': 496
                },
                'Sat': {
                    'x': 505,
                    'y': 496
                }
            }
        },
        'PR-C': {
            'schedule': {
                'Wed': {
                    'x': 505,
                    'y': 496
                },
                'Thu': {
                    'x': 910,
                    'y': 496
                },
                'Sat': {
                    'x': 910,
                    'y': 496
                },
                'Sun': {
                    'x': 910,
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
                    'x': 910,
                    'y': 496
                },
                'Sat': {
                    'x': 1315,
                    'y': 496
                },
                'Sun': {
                    'x': 1315,
                    'y': 496
                }
            }
        },
        'specially_open': {
            'PR-C': {
                'x': 505,
                'y': 496 
            },
            'PR-D': {
                'x': 910,
                'y': 496 
            },
            'PR-A': {
                'x': 1315,
                'y': 496 
            },
            'PR-B': {
                'x': 1720,
                'y': 496 
            }
        }
    }
}

daily_mission_metadata = {
    'Supplies': {
        'Mon': {
            'LS': 'Tactical Drill (EXP)',
            'AP': 'Tough Siege (Red Tickets)',
            'SK': 'Resource Search (Base Material)',
        },
        'Tue': {
            'LS': 'Tactical Drill (EXP)',
            'CA': 'Aerial Threat (White, Green, Blue Skill Summary)',
            'CE': 'Cargo Escort (LMD)',
        },
        'Wed': {
            'LS': 'Tactical Drill (EXP)',
            'CA': 'Aerial Threat (White, Green, Blue Skill Summary)',
            'SK': 'Resource Search (Base Material)',
        },
        'Thu': {
            'LS': 'Tactical Drill (EXP)',
            'AP': 'Tough Siege (Red Tickets)',
            'CE': 'Cargo Escort (LMD)',
        },
        'Fri': {
            'LS': 'Tactical Drill (EXP)',
            'CA': 'Aerial Threat (White, Green, Blue Skill Summary)',
            'SK': 'Resource Search (Base Material)',
        },
        'Sat': {
            'LS': 'Tactical Drill (EXP)',
            'SK': 'Resource Search (Base Material)',
            'AP': 'Tough Siege (Red Tickets)',
            'CE': 'Cargo Escort (LMD)',
        },
        'Sun': {
            'LS': 'Tactical Drill (EXP)',
            'CA': 'Aerial Threat (White, Green, Blue Skill Summary)',
            'AP': 'Tough Siege (Red Tickets)',
            'CE': 'Cargo Escort (LMD)',
        }
    },
    'Chips': {
        'Mon': {
            'PR-A': 'Solid Defense (Defender/Medic)',
            'PR-B': 'Fierce Attack (Caster/Sniper)'
        },
        'Tue': {
            'PR-B': 'Fierce Attack (Caster/Sniper)',
            'PR-D': 'Fearless Protection (Guard/Specialist)'
        },
        'Wed': {
            'PR-C': 'Unstoppable Charge (Vanguard/Support)',
            'PR-D': 'Fearless Protection (Guard/Specialist)'
        },
        'Thu': {
            'PR-A': 'Solid Defense (Defender/Medic)',
            'PR-C': 'Unstoppable Charge (Vanguard/Support)'
        },
        'Fri': {
            'PR-A': 'Solid Defense (Defender/Medic)',
            'PR-B': 'Fierce Attack (Caster/Sniper)'
        },
        'Sat': {
            'PR-B': 'Fierce Attack (Caster/Sniper)',
            'PR-C': 'Unstoppable Charge (Vanguard/Support)',
            'PR-D': 'Fearless Protection (Guard/Specialist)'
        },
        'Sun': {
            'PR-A': 'Solid Defense (Defender/Medic)',
            'PR-C': 'Unstoppable Charge (Vanguard/Support)',
            'PR-D': 'Fearless Protection (Guard/Specialist)'
        }
    }
}


#https://stackoverflow.com/a/35378944
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
    
def do_screenshot(device):
    image = device.screencap()
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), 'wb') as fd:
        fd.write(image)
    with Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png')) as fd:
        image = np.array(fd, dtype=np.uint8)
    return image

def get_sanity():
    custom_oem=r'digits --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789/'
    # Load image
    image = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'))
    image_arr = np.array(image)
    image_arr = image_arr[
        # y
        # top - bottom
        30:105, 
        # x
        # left - right
        1980-299:2155-299]
    image.close()
    image = Image.fromarray(image_arr)
    image.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))

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
        img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
        for i in [200, 190, 140, 120, 100]:
            retval, img = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(cv2.resize(img,(0,1),fx=3,fy=2))
            img = cv2.GaussianBlur(img,(11,11),0)
            img = cv2.medianBlur(img,9)
            str_res = pytesseract.image_to_string(img, config=custom_oem)
            if '/' in str_res and str_res[0]!='/':
                print(f'\n[*] complete brute force at {i}\n')
                raise Exception()
            
        img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
        for i in [200, 190, 140, 120, 100]:
            retval, img = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(cv2.resize(img,(0,0),fx=3,fy=3))
            img = cv2.GaussianBlur(img,(11,11),0)
            img = cv2.medianBlur(img,9)
            str_res = pytesseract.image_to_string(img, config=custom_oem)
            if '/' in str_res and str_res[0]!='/':
                print(f'\n[**] complete brute force at {i}\n')
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
    except Exception as e:
        print('[!] Exception get sanity')
        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'), img)
        return str_res

def return_to_main_menu(device):
    print('\n[!] going back to the main menu ...')
    iter=0
    while True:
        if iter>3:
            sleep(0.5)
            do_screenshot(device)
            image = Image.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'))
            image_arr = np.array(image)
            image_arr = image_arr[
                # y
                # top - bottom
                450:535, 
                # x
                # left - right
                862:1356]
            image.close()
            image = Image.fromarray(image_arr)
            image.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))

            img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crop.png'))
            device.shell("input keyevent 4")
            if pytesseract.image_to_string(img).split('\n')[0] == "Are you sure you want to exit?":
                break
        else:
            device.shell("input keyevent 4")
            print(f'[{iter}] back button')
        iter+=1

def do_restore_sanity(device, sanity, max_sanity, full=True):
    global restore_sanity_check, total_done
    if full:
        # go back to the main menu
        return_to_main_menu(device)

        # main menu
        print(f'\n[!] do restore sanity, hope your orundum is enough :p\nsanity : {sanity}\nmax_sanity : {max_sanity}')
        for i in range(int((999+sanity)/max_sanity)):
            print('.', end='', flush=True)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['restore_sanity_button']['x'], magic_xy['restore_sanity_button']['y']))
            sleep(1)
            device.shell("input touchscreen tap {0} {1}".format(magic_xy['confirm_restore_sanity_button']['x'], magic_xy['confirm_restore_sanity_button']['y']))
            sleep(4.5)
            sanity+=max_sanity
        restore_sanity_check=True
        print('\n', end='', flush=True)
    else:
        print(f'\n[!] do restore sanity with your 1 orundum :3\nsanity : {sanity}\nmax_sanity : {max_sanity}')
        device.shell("input touchscreen tap {0} {1}".format(magic_xy['restore_sanity_onstage_tap']['x'], magic_xy['restore_sanity_onstage_tap']['y']))
        sleep(4.5)
        sanity+=max_sanity
    total_done-=1
    print(f'\n[!!!] sanity has been restored\nsanity : {sanity}\n')

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
        
def bot_process(device, jobiter):
    global total_done, args, r, g, b
    tstart=0
    tend=0
    failure_count=0
    failure_data={}
    while True:
        # first stage
        while True:
            if check_rgb(device, {
                                    'x': magic_xy['autodeploy_check']['x'], 
                                    'y': magic_xy['autodeploy_check']['y']
                                 },
                                 magic_rgb['autodeploy'],
                                 verbose):
                # get sanity info
                sanity = get_sanity()
                if jobiter==total_done:
                    print(f'\n[!] {total_done}/{jobiter} job is complete')
                    if args.only_automate:
                        return False
                    else:
                        return_to_main_menu(device)
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
                    print(f'[+] {total_done}/{jobiter} mission has been done')
                    print(f'[+] current sanity now is {sanity}\n')
                tend=time()
                total_done+=1
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['autodeploy_tap']['x'], magic_xy['autodeploy_tap']['y']))
                break
            else:
                if verbose:
                    #17,18,18
                    print(f'[{r},{g},{b}] autodeploy check not found')
                    if failure_count == 60:
                        print('[failure autodeploy detected]')
                        device.shell("input touchscreen tap 1722 988")
                        failure_count=0
                        continue
                    elif failure_data == {'r': r, 'g': g, 'b': b}:
                        failure_count+=1

                    failure_data = {'r': r, 'g': g, 'b': b}
                else:
                    print('.', end='', flush=True)

        # second stage
        while True:
            if check_rgb(device, {
                                    'x': magic_xy['mission_start_check']['x'], 
                                    'y': magic_xy['mission_start_check']['y']
                                 },
                                 magic_rgb['mission_start'],
                                 verbose):
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_start_tap']['x'], magic_xy['mission_start_tap']['y']))
                break
            else:
                if check_rgb(device, {
                                        'x': magic_xy['sanity_check']['x'], 
                                        'y': magic_xy['sanity_check']['y']
                                    },
                                    magic_rgb['sanity_check'],
                                    verbose):
                    print(f'[{r},{g},{b}] sanity not enough :(\n')
                    if restore_sanity:
                        if args.only_automate:
                            do_restore_sanity(device, sanity, max_sanity, full=False)
                        else:
                            do_restore_sanity(device, sanity, max_sanity)
                        return True
                    else:
                        return False
                if verbose:
                    print(f'[{r},{g},{b}] mission_start check not found')
                else:
                    print('.', end='', flush=True)

        # third stage
        level_up=0
        conn_err=0
        while True:
            if check_rgb(device, {
                                    'x': magic_xy['mission_complete_check']['x'], 
                                    'y': magic_xy['mission_complete_check']['y']
                                 },
                                 magic_rgb['mission_complete'],
                                 verbose):
                sleep(2)
                device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_complete_tap']['x'], magic_xy['mission_complete_tap']['y']))
                break
            else:
                image = do_screenshot(device)
                rs,gs,bs = [int(j) for j in [list(i[:3]) for i in image[magic_xy['mission_error_check']['y']]][magic_xy['mission_error_check']['x']]]
                if level_up == 15:
                    level_up=0
                    print(f'[{rs},{gs},{bs}] level up')
                    device.shell("input touchscreen tap {0} {1}".format(magic_xy['mission_complete_tap']['x'], magic_xy['mission_complete_tap']['y']))
                elif conn_err == 5:
                    print('[!!!] connection error')
                    exit(0)
                elif ({'r': rs, 'g': gs, 'b': bs} == {'r': 1, 'g': 1, 'b': 1} or
                      {'r': rs, 'g': gs, 'b': bs} == {'r': 1, 'g': 1, 'b': 0} or
                      {'r': rs, 'g': gs, 'b': bs} == {'r': 12, 'g': 11, 'b': 10}):
                    level_up+=1
                elif ({'r': rs, 'g': gs, 'b': bs} == {'r': 115, 'g': 20, 'b': 20} or
                      {'r': rs, 'g': gs, 'b': bs} == {'r': 117, 'g': 20, 'b': 20}):
                    print('[!] connection error, re-enter')
                    device.shell("input touchscreen tap {0} {1}".format(magic_xy['connection_error_tap']['x'], magic_xy['connection_error_tap']['y']))
                    sleep(3)
                    conn_err+=1
                else:
                    if verbose:
                        print(f'[{r},{g},{b}]-[{rs},{gs},{bs}] mission_complete check not found')
                    else:
                        print('.', end='', flush=True)

def get_stage_group_name(stage_name):
    ops_name = stage_name[::-1].split('-', 1)[1][::-1]
    ops_num = stage_name[::-1].split('-', 1)[0][::-1]
    for mission_key in mission_stage_metadata:
        if ops_name in mission_stage_metadata[mission_key]:
            return [mission_key, ops_name, ops_num]
    return False


def check_stage_exist(stage_name):
    date_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-11))).strftime('%a')
    for stage_key in stage_name:
        for date in stage_key:
            print()


def bot_select_mission(device, stage_name):
    resp = get_stage_group_name(stage_name)
    if not resp:
        print('[!] record for stage name not found!')
        exit(1)
    else:
        stage_group = resp[0]
        ops_name = resp[1]
        ops_num = resp[2]

    if not all_opens:
        date_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-11))).strftime('%a')
        if date_now not in mission_stage_metadata[stage_group][ops_name]['schedule']:
            print(f'[!] {stage_name} currently unavailable.')
            exit(1)
        elif int(ops_num) > mission_stage_metadata[stage_group]['total_stage']:
            print(f'[!] {stage_name} is invalid.')
            return False

    # must be on main menu
    print('[open combat menu]')
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset']['open_combat']['x'], mission_stage_metadata[stage_group]['offset']['open_combat']['y']))
    sleep(1)
    print('[selecting stage]')
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset']['select_stage']['x'], mission_stage_metadata[stage_group]['offset']['select_stage']['y']))
    sleep(1)
    if all_opens:
        device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['specially_open'][ops_name]['x'], mission_stage_metadata[stage_group]['specially_open'][ops_name]['y']))
    else:
        device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group][ops_name]['schedule'][date_now]['x'], mission_stage_metadata[stage_group][ops_name]['schedule'][date_now]['y']))
    sleep(1)
    print(f'[go to {stage_name}]\n')
    device.shell("input touchscreen tap {0} {1}".format(mission_stage_metadata[stage_group]['offset'][f'operation_{ops_num}']['x'], mission_stage_metadata[stage_group]['offset'][f'operation_{ops_num}']['y']))
    sleep(1)

    return True

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
        for x,y in find_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)),  'resources\\' + 'recruit_btn.png'), score=0.97):
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
        for x,y in find_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'screen.png'), os.path.join(os.path.dirname(os.path.realpath(__file__)),  'resources\\' + 'hire_btn.png'), score=0.95):
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
    parser = argparse.ArgumentParser(description='Arknight Stage Bot')
    parser.add_argument('-j', '--job-list', dest='job_list', help='ex: -j/--job-list "CE-5:20;CA-5:30"')
    parser.add_argument('-a', '--automate', type=int, dest='only_automate', nargs='?', const=1000, help='just do automation, select the stage to automate first!')
    parser.add_argument('-r', '--recruitment', type=int, dest='recruit', nargs='?', const=10, help='just do automation recruitment')
    parser.add_argument('-s', '--sanity', action='store_true', dest='sanity_restore', help='with sanity restoration.')
    parser.add_argument('-S', '--show', action='store_true', dest='show', help='show current daily mission.')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug', help='only for debugging code.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='enable verbose message.')
    args = parser.parse_args()

    # argument parsing function
    if not args.job_list and not args.only_automate and not args.recruit and not args.show and not args.debug:
        parser.print_help()
        exit(1)
    elif args.show:
        date_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-11))).strftime('%a')
        for stage_key in daily_mission_metadata:
            print(f'\n{"." + "_"*(len(stage_key)) + "."}\n|{stage_key}|\n{"." + "-"*(len(stage_key)) + "."}')
            for stage_name in daily_mission_metadata[stage_key][date_now]:
                print(f'||> {stage_name}')
                print(f'|||> {daily_mission_metadata[stage_key][date_now][stage_name]}')
        exit(0)
    else:
        adb = Client(host='127.0.0.1', port=5037)
        device = adb.devices()[0]
        # screen_x = device.shell('wm size').split(' ')[-1].split('x')
        # screen_y = eval(screen_x[1])
        # screen_x = eval(screen_x[0])
        verbose = args.verbose

    # debugging section
    if args.debug:
        # image = do_screenshot(device)
        # r,g,b = [int(j) for j in [list(i[:3]) for i in image[magic_xy['mission_complete_check']['y']]][magic_xy['mission_complete_check']['x']]]
        # print(r,g,b)
        automate_recruitment(device)
        #do_screenshot(device)
        #crop_image('screen.png','recruit_btn.png',job_tags_xy['testing'])
        exit()
    elif args.sanity_restore:
        restore_sanity=True
        
    # main  
    if args.recruit:
        automate_recruitment(device, args.recruit)
        exit(0)
    elif args.job_list:
        job_list=[x for x in args.job_list.split(';') if x != '']
        if job_list[0] == "allopens":
            all_opens=True
            job_list.pop(0)
            
        # check stage
        for xname in job_list:
            if ':' not in xname or '-' not in xname:
                print(f'[!] job {job_list.index(xname)+1} doesn\'t have stage level splitter "-" and job count splitter ":" (Ex: CE-5:10).')
                exit(1)
            elif not get_stage_group_name(xname.split(':')[0]):
                print(f'[!] {xname} not a valid stage.')
                exit(1)

        for x in job_list:
            job_name = x.split(':')[0]
            job_count = int(x.split(':')[1])

            while True:
                if not bot_select_mission(device, job_name):
                    break
                if not bot_process(device, job_count):
                    exit(0)
                else:
                    restore_sanity_check=True

                if total_done >= job_count:
                    total_done = 0
                    break
                
            if not restore_sanity_check:
                restore_sanity_check=False
                return_to_main_menu(device)
    elif args.only_automate:
        job_count = args.only_automate
        while True:
            if not bot_process(device, job_count):
                exit(0)