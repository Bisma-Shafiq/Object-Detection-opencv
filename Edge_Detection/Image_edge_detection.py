import cv2

img = cv2.imread('image.png')
imgGray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray , threshold1=100 , threshold2=100)
cv2.imshow('img',img)
cv2.imshow('imgGray',imgGray)
cv2.imshow('imgGray',imgCanny)
cv2.waitKey(0)

