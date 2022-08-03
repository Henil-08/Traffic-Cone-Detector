import cv2 as cv
import numpy as np

cap = cv.VideoCapture('Data/Video/1.mp4')

while cap.isOpened():
    isSuccess, img = cap.read()
    if not isSuccess:
        print('Video Ended!')
        break

    img = cv.resize(img, (480, 360))

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    upper = np.array([55, 255, 255])
    lower = np.array([5, 70, 70])
    mask = cv.inRange(imgHSV, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    smoothed = cv.GaussianBlur(mask, (3, 3), 0)
    canny = cv.Canny(smoothed, 50, 150)

    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    approxContours = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 10, True)
        approxContours.append(approx)

    convexHull3To10 = []
    for approxContour in approxContours:
        ConvexHull = cv.convexHull(approxContour)
        if 3 <= len(ConvexHull) <= 10:
            convexHull3To10.append(ConvexHull)

    def convexHullPointingUp(ch):
        pointsAboveCenter, poinstBelowCenter = [], []

        _, y, w, h = cv.boundingRect(ch)
        aspectRatio = w / h

        if aspectRatio < 0.8:
            verticalCenter = y + h / 2

            for point in ch:
                if point[0][1] < verticalCenter:
                    pointsAboveCenter.append(point)
                elif point[0][1] >= verticalCenter:
                    poinstBelowCenter.append(point)

            leftX = poinstBelowCenter[0][0][0]
            rightX = poinstBelowCenter[0][0][0]
            for point in poinstBelowCenter:
                if point[0][0] < leftX:
                    leftX = point[0][0]
                if point[0][0] > rightX:
                    rightX = point[0][0]

            for point in pointsAboveCenter:
                if (point[0][0] < leftX) or (point[0][0] > rightX):
                    return False

        else:
            return False

        return True

    cones = []
    bounding_Rects = []

    for convexHulls in convexHull3To10:
        if convexHullPointingUp(convexHulls):
            cones.append(convexHulls)
            rect = cv.boundingRect(convexHulls)
            bounding_Rects.append(rect)

    imgTrafficCones = np.zeros_like(mask)
    cv.drawContours(imgTrafficCones, cones, -1, (255, 255, 255), 2)

    final = img.copy()

    for rect in bounding_Rects:
        cv.rectangle(final, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

    cv.imshow('Cone', img)
    cv.imshow('Mask', mask)
    cv.imshow('Debug', imgTrafficCones)
    cv.imshow('final', final)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()