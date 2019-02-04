import cv2
# import matplotlib.pyplot as plt
import numpy as np

isImg = False
imgPath = "test_image.jpg"
videoPath = "test_video.mp4"
HEIGHT = 0


def xyCoord(mbCoord):
    m, b = mbCoord
    py1 = HEIGHT
    py2 = int(HEIGHT * 0.5)
    px1 = int((py1 - b) / m)
    px2 = int((py2 - b) / m)
    return (px1, py1), (px2, py2)


def process(img):
    global HEIGHT
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    # plt.imshow(canny)
    # plt.show()

    HEIGHT = img.shape[0]
    polygons = np.array([[(300, HEIGHT), (1000, HEIGHT), (575, 250)]])
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polygons, 255)
    maskedImg = cv2.bitwise_and(canny, mask)

    # plt.imshow(cv2.bitwise_and(gray, mask))
    # plt.show()

    lines = cv2.HoughLinesP(maskedImg, 2, np.pi / 180, 100, np.array([]), minLineLength=50, maxLineGap=5)
    linesImg = np.zeros_like(img)
    if lines is not None:
        negative = []
        positive = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            (negative if slope < 0 else positive).append((slope, intercept))
        if negative:
            pt1, pt2 = xyCoord(np.average(negative, axis=0))
            cv2.line(linesImg, pt2, pt1, (255, 0, 0), 10)
        if positive:
            pt1, pt2 = xyCoord(np.average(positive, axis=0))
            cv2.line(linesImg, pt2, pt1, (255, 0, 0), 10)
    return cv2.addWeighted(img, 0.8, linesImg, 1, 1)


if __name__ == "__main__":
    if isImg:
        image = cv2.imread(imgPath)
        cv2.imshow('result', process(image))
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(videoPath)
        while cap.isOpened():
            good, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q') or not good:
                break
            cv2.imshow('result', process(frame))
        cap.release()
    cv2.destroyAllWindows()
