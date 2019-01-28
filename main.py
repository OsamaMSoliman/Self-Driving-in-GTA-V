import time

import cv2
import numpy as np
from PIL import ImageGrab


def mainScreenCaptureLoop():
    last_time = time.time()
    screenWindow = lambda start, size: (start + (start[0] + size[0], start[1] + size[1]))
    while True:
        screen = np.array(ImageGrab.grab(bbox=screenWindow(screenStart, screenEnd)))
        print(f"time= {time.time()-last_time}")
        last_time = time.time()
        # PressKey(W)
        # ReleaseKey(W)
        # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        cv2.imshow('window', processImg(screen))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def processImg(original):
    processed = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed = cv2.Canny(processed, threshold1=200, threshold2=300)
    processed = cv2.GaussianBlur(processed, (5, 5), 0)
    # return processed
    processed = regionOfIntrest(processed, AreaOfIntersVert)
    # return processed
    lines = cv2.HoughLinesP(image=processed, rho=1, theta=np.pi / 180, threshold=180, minLineLength=100, maxLineGap=5)
    drawLines(processed, lines)
    return processed


def regionOfIntrest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(img, mask)


def drawLines(img, lines):
    if lines is not None:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), 255, 2)


def countDown(num):
    if num == 0: return
    for i in list(range(num))[::-1]:
        print(i + 1)
        time.sleep(1)


if __name__ == "__main__":
    screenStart = (0, 26)
    screenEnd = (800, 600)
    AreaOfIntersVert = np.array([[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]])
    # countDown(3)
    mainScreenCaptureLoop()
