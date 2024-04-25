#Old code. May not be supported anymore

import pyautogui
import keyboard
import time
import numpy
import imageio.v3
from PIL import Image
import screenshot2 as sc
from discord_webhook import DiscordWebhook as DW
import keyboard
import os
import pickle
import json

def compress_img(img, img_res=25, save_old = False):
    x,y,v = img.shape
    img = img.tolist()
    div = x*y // img_res ** 2
    full_img = []
    for y1 in range(img_res):
        row = []
        for x1 in range(img_res):
            colourv = [0,0,0]
            for x2 in range(x//img_res):
                for y2 in range(y//img_res):
                    r = img[x2+x1*x//img_res][y2+y1*y//img_res][0]#.item()
                    g = img[x2+x1*x//img_res][y2+y1*y//img_res][1]#.item()
                    b = img[x2+x1*x//img_res][y2+y1*y//img_res][2]#.item()
                    colourv[0] += r
                    colourv[1] += g
                    colourv[2] += b
            colourv = [j//div for j in colourv]
            row.append(colourv)
        full_img.append(row)
    arr = numpy.array(full_img)
    return(arr)
def screenshot(img_res=25, save_old = False):
    img = pyautogui.screenshot()
    x,y = img.size
    if save_old:
        img.save('old_img.png')
    div = x*y // img_res ** 2
    full_img = []
    for y1 in range(img_res):
        row = []
        for x1 in range(img_res):
            colourv = [0,0,0]
            for x2 in range(x//img_res):
                for y2 in range(y//img_res):
                    r,g,b = img.getpixel((x2+x1*x//img_res,y2+y1*y//img_res))
                    colourv[0] += r
                    colourv[1] += g
                    colourv[2] += b
            colourv = [j//div for j in colourv]
            row.append(colourv)
        full_img.append(row)
    arr = numpy.array(full_img)
    return(arr)
def read_from_bmp(file:str=r'C:\Users\brolf\Desktop\vscodecodefold\AI-framework-main\pymind v1\out.bmp'):
    img = imageio.v3.imread(file)
    return(img)
def make_bmp(arr,filename,size):
    if type(size) == int:
        I = Image.new('RGB', (size,size))
    else:
        I = Image.new('RGB', size)
    count1 = 0
    for x in arr:
        count2 = 0
        for y in x:
            y = tuple([int(i) for i in y.tolist()])
            I.putpixel((count1,count2), y)
            count2 += 1
        count1 += 1
    I.save(f'{filename}.bmp')
def make_grayscale(arr,filename,size):
    I = Image.new('L', (size,size))
    count1 = 0
    for x in arr:
        count2 = 0
        for y in x:
            y = int(y)
            I.putpixel((count1,count2), y)
            count2 += 1
        count1 += 1
    I.save(f'{filename}.bmp')
def numpy_com_optimised(arr):
    rgb_arrs = arr.transpose((2,0,1))
    new_arrs = rgb_arrs.reshape(3,40,40,-1)
    averages = numpy.uint8(numpy.average(new_arrs, 3))
    averages = averages.transpose(2,1,0)
    return averages
def one_liner(arr, shape=(30,36,30,64,3)):
    return numpy.mean(arr.reshape(*shape).transpose(1,3,0,2,4).reshape(-1,shape[0],shape[2],shape[4]), 0).astype(int).transpose(1,0,2)
def cf_optimised(final_res, starting_res, windowname, file=None, capture_rect=False, starting_point=(0,0)):
    if type(starting_res) == int:
        starting_res = (starting_res,starting_res)
    if type(final_res) == int:
        final_res = (final_res, final_res)
    if starting_res[0] % final_res[0] != 0 or starting_res[1] % final_res[1] != 0:
        raise Exception("final res must be a factor of starting res")
    sc.take_bmp_screenshot(starting_res, 'uncompressed', windowname, capture_rect, starting_point)
    img_arr = read_from_bmp('uncompressed.bmp')
    shape = (final_res[1], starting_res[1] // final_res[1], final_res[0], starting_res[0] // final_res[0], 3)
    img2 = one_liner(img_arr, shape)
    if file != None:
        make_bmp(img2, file, final_res)
    return(img2)
def numpy_compress(arr,gsc=True,size=25):
    if type(size) == type(1):
        x = size
        y = size
    else:
        x,y = size
    fullrgb = []
    rgb_arrs = numpy.split(arr, 3, 2)
    for arr1 in rgb_arrs:
        color = []
        img_arrs = numpy.array_split(arr1, x)
        for arr in img_arrs:
            row = []
            sections = numpy.array_split(arr, y, 1)
            for section in sections:
                row.append(int(numpy.average(section)))
            color.append(row)
        fullrgb.append(color)
    #start = time.time()
    #result = [[list(a) for a in zip(l1, l2, l3)] for l1, l2, l3 in zip(fullrgb[0], fullrgb[1], fullrgb[2])]
    #end = time.time()
    final_arr = numpy.array(fullrgb).transpose((2,1,0))
    #end2 = time.time()
    gray_sc = None
    if gsc:
        gray_sc = []
        for xarr in final_arr:
            row = []
            for yarr in xarr:
                row.append(int(numpy.average(yarr)))
            gray_sc.append(row)
        gray_sc = numpy.array(gray_sc)
    return((final_arr, gray_sc))
def win_optomised(final_resolution, starting_resolution, windowname, file=None, grayscale=False,capture_rect=False,starting_point=(0,0)):
    sc.take_bmp_screenshot(starting_resolution, 'uncompressed', windowname,capture_rect,starting_point)
    img_arr = read_from_bmp('uncompressed.bmp')
    img2,gray_img = numpy_compress(img_arr,grayscale,final_resolution)
    if file != None:
        make_bmp(img2, file)
        if grayscale:
            make_grayscale(gray_img, f'grayscale_{file}')
            return(gray_img)
    return(img2)
def get_times():
    #start = time.perf_counter()
    #img3 = screenshot()
    #print('time to screenshot with pyautogui:', time.perf_counter()-start)
    start = time.perf_counter()
    a = sc.take_bmp_screenshot(name='z_bmp_outp')
    print('time to screenshot with win32:', time.perf_counter()-start)
    start = time.perf_counter()
    img_arr = read_from_bmp('z_bmp_outp.bmp')
    print('time to read img:', time.perf_counter()-start)
    #diff = abs(abs(img_arr)-abs(a))
    #s = numpy.sum(diff)
    start = time.perf_counter()
    img = numpy_com_optimised(img_arr)
    print('time to fast numpy compress:', time.perf_counter()-start)
    start = time.perf_counter()
    img2,gray_img = numpy_compress(img_arr)
    print('time to numpy compress:', time.perf_counter()-start)
    start = time.perf_counter()
    make_bmp(img, 'zReg_test_bmp',40)
    print('time to save fast np img:', time.perf_counter()-start)
    start = time.perf_counter()
    make_bmp(img2, 'zNp_test_bmp',25)
    print('time to save np img:', time.perf_counter()-start)
    start = time.perf_counter()
    make_grayscale(gray_img, 'zNp_gray_bmp',25)
    print('time to save gray img:', time.perf_counter()-start)
def send_to_discord(filenames, sendnames):
    webhook = DW("https://discord.com/api/webhooks/1125017628566618182/DVn9UbvA_cHlpk5-Jh1wj-WCODsaH30VS-cY-RqKWwvl4s5RsMRfgUni9Pb4SkT5NGzo")
    if type(filenames) == type(''):
        with open(filenames, 'rb') as f:
            webhook.add_file(f.read(), f'{sendnames}')
    else:
        for index in range(len(filenames)):
            with open(filenames[index], 'rb') as f:
                webhook.add_file(f.read(), f'{sendnames[index]}')
    webhook.execute()
def pil_bmp_to_jpg(filenames,outputnames):
    if type(filenames) == type(''):
        Image.open(filenames).save(outputnames)
    else:
        for index in range(len(filenames)):
            Image.open(filenames[index]).save(outputnames[index])
#p
if __name__ == '__main__':
    ret_list = []
    update = False
    def logger(args):
        global update, ret_list
        update = True
        ret_list.append([args.name, args.event_type, args.time])
        return
    keyboard.hook(logger)
    count = 0
    cwd = os.getcwd()
    a = len(os.listdir(f"{cwd}\\screenies"))
    os.mkdir(f"screenies/iteration_{a}")
    count = 0
    istrue = False
    check = True
    with open(r"C:\Users\brolf\AppData\Roaming\.minecraft\config\ChatTriggers\modules\pymindLinker\rot_info.json", 'r') as f:
        string = f.read()
        try:
            last_rot = json.loads(string)
        except:
            a=1
    while True:
        if istrue:
            cf_optimised(40,(1920,1080), 'Minecraft 1.8.9', f"screenies/iteration_{a}/Cf_test_bmp_{count}")
            with open(r"C:\Users\brolf\AppData\Roaming\.minecraft\config\ChatTriggers\modules\pymindLinker\info_2.json", 'r') as f:
                string = f.read()
                if string != '':
                    all_pos = json.loads(string)
            with open(r"C:\Users\brolf\AppData\Roaming\.minecraft\config\ChatTriggers\modules\pymindLinker\rot_info.json", 'r') as f:
                string = f.read()
                if string != '':
                    rot = json.loads(string)
                    c_hange_in_rot = [rot[0]-last_rot[0], rot[1]-last_rot[1]]
                    last_rot = rot
            with open('current_keys.PICKLE', 'rb') as f:
                keys = pickle.load(f)
            savelist = [all_pos, c_hange_in_rot, keys]
            with open(f"screenies/iteration_{a}/extra_info_{count}.PKL", 'wb') as f:
                pickle.dump(savelist, f)
            #sc.take_bmp_screenshot((1920, 1080), f"screenies/iteration_{a}/Cf_test_bmp_{count}", 'Minecraft 1.8.9')
            count += 1
            for x in ret_list:
                if x[0] == 'p':
                    if check:
                        if x[1] == 'down':
                            print('ending')
                            istrue = False
                            check = False
                            exit('ended :)')
                    if x[1] == 'up':
                        check = True
            ret_list = []
        else:
            time.sleep(0.025)
            for x in ret_list:
                if x[0] == 'p':
                    if check:
                        if x[1] == 'down':
                            print('starting')
                            istrue = True
                            check = False
                    if x[1] == 'up':
                        check = True
            ret_list = []
"""
times = []
for x in range(25):
    start = time.perf_counter()
    #print(get_screen(25))
    img = pyautogui.screenshot()
    times.append(time.perf_counter()-start)
print(sum(times)/len(times))"""
"""a = numpy.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])
print(a)
a = a.tolist()
print(a)"""
"""r = img_arr[0][0][0].item()
fullrgb = []
rgb_arrs = numpy.split(img_arr, 3, 2)
for arr1 in rgb_arrs:
    color = []
    img_arrs = numpy.array_split(arr1, 25)
    for arr in img_arrs:
        row = []
        sections = numpy.array_split(arr, 25, 1)
        for section in sections:
            row.append(int(numpy.average(section)))
        color.append(row)
    fullrgb.append(color)
fullrgb
final_arr = numpy.array(fullrgb).transpose((2,1,0))
final_arr
make_bmp(final_arr, 'zTest_bmp')"""
"""
[[1,11,111],[2,22,222],[3,33,333]]
[[4,44,444],[5,55,555],[6,66,666]]
[[7,77,777],[8,88,888],[9,99,999]]
[[[1,4,7],[11,44,77],[111,444,777]],[[2,5,8],[22,55,88],[222,555,888]],[[3,6,9],[33,66,99],[333,666,999]]]
"""
