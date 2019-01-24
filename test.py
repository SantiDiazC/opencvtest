import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv2.filter2D(frame, cv2.CV_32F, kernel)
    # Our operations on the frame come here

    sharp = np.float32(frame)
    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    # cv.imshow('Laplace Filtered Image', imgLaplacian)
    cv2.imshow('New Sharped Image', imgResult)

    # Create binary image from source image
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('Binary Image', bw)

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('Distance Transform Image', dist)

    #blur = cv2.GaussianBlur(dist, (3, 3), 0)
    dist1 = np.uint8(dist)
    dist1 = cv2.Canny(dist1, 255, 255)
    cv2.imshow('canny', dist1)
    _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv2.erode(dist, kernel1)
    cv2.imshow('Peaks', dist)

    '''gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 100)
    laplace = cv2.Laplacian(edges,cv2.CV_64F)

    ret1, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)'''

    # Display the resulting frame
    '''cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('edges', edges)
    cv2.imshow('laplace', laplace)
    cv2.imshow('binary', th1)'''
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()