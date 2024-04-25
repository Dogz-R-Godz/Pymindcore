#Old code. May not be supported anymore

import win32gui
import win32ui
import win32con

def take_bmp_screenshot(dimensions:tuple=(3440,1440), name:str='out', window:str='-1'):
    #w = 1920 # set this
    #h = 1080 # set this
    #bmpfilenamename = "out.bmp" #set this
    w,h = dimensions
    bmpfilenamename = f'{name}.bmp'

    hwnd = win32gui.FindWindow(None, window)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)
    dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
if __name__ == '__main__':
    take_bmp_screenshot()

def get_screenshotable_windows():
    screenshotable_windows = []
    
    def is_visible(hwnd):
        return win32gui.IsWindowVisible(hwnd)

    def callback(hwnd, _):
        if is_visible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            screenshotable_windows.append((hwnd, window_title))

    # Enumerate through all open windows
    win32gui.EnumWindows(callback, None)
    
    return screenshotable_windows

if __name__ == '__main__':
    windows = get_screenshotable_windows()
    for hwnd, title in windows:
        print(f"Window Title: {title}, Handle: {hwnd}")
