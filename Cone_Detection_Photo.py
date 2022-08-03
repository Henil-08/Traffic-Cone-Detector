import cv2 as cv
import numpy as np

img = cv.imread('Data/3.jpg')
img = cv.resize(img, (360, 640))

imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
upper = np.array([55, 255, 255])
lower = np.array([22, 100, 100])
mask = cv.inRange(imgHSV, lower, upper)

kernel = np.ones((7, 7), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
smoothed = cv.GaussianBlur(mask, (3, 3), 0)
canny = cv.Canny(smoothed, 80, 160)
res = cv.bitwise_and(img, img, mask=mask)


contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
imgContours = np.zeros_like(img)
cv.drawContours(imgContours, contours, -1, (0, 255, 0), 1)


approxContours = []

for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    approxContours.append(approx)

img_Contours = np.zeros_like(mask)
cv.drawContours(img_Contours, approxContours, -1, 255, 1)


allConvexHulls = []

for approxContour in approxContours:
    allConvexHulls.append(cv.convexHull(approxContour))

imgAllConvexHulls = np.zeros_like(mask)
cv.drawContours(imgAllConvexHulls, allConvexHulls, -1, (255, 255, 255), 2)


convexHull3To10 = []

for convexHull in allConvexHulls:
    if 3 <= len(convexHull) <= 10:
        convexHull3To10.append(cv.convexHull(convexHull))

imgConvexHulls3To10 = np.zeros_like(mask)
cv.drawContours(imgConvexHulls3To10, convexHull3To10, -1, (255, 255, 255), 2)


def convexHullPointingUp(ch):
    pointsAboveCenter, poinstBelowCenter = [], []

    x, y, w, h = cv.boundingRect(ch)
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

for ch in convexHull3To10:
    if convexHullPointingUp(ch):
        cones.append(ch)
        rect = cv.boundingRect(ch)
        bounding_Rects.append(rect)

imgTrafficCones = np.zeros_like(mask)
cv.drawContours(imgTrafficCones, cones, -1, (255, 255, 255), 2)

finalcopy = img.copy()

for rect in bounding_Rects:
    cv.rectangle(finalcopy, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (1, 255, 1), 2)


cv.imshow('Cone', img)
# cv.imshow('mask', mask)
# # cv.imshow('canny', canny)
# cv.imshow('res', res)
# cv.imshow('contours', imgContours)
# cv.imshow('approxContours', img_Contours)
# cv.imshow('convexHull', imgAllConvexHulls)
# cv.imshow('convexHull3to10', imgConvexHulls3To10)
# cv.imshow('Up Cones', imgTrafficCones)
cv.imshow('final', finalcopy)
cv.waitKey(0)
cv.destroyAllWindows()