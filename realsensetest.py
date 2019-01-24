'''import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()'''

import numpy as np
import cv2 as cv
import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Start pipeline
profile = pipeline.start(config)

erodeKernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))

fgbg = cv.createBackgroundSubtractorMOG2()
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    colour_frame = frames.get_color_frame()

    color_image = np.asanyarray(colour_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),0)
    fgmask = fgbg.apply(blur)

    im2, contours, hierarchy = cv.findContours(fgmask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv.contourArea(c) < 50:
            continue

        (x,y,w,h) = cv.boundingRect(c)
        cv.rectangle(color_image, (x,y), (x + w, y + h), (0,255,0), 2)

    cv.imshow('RealSense', color_image)
    cv.imshow("Depth", depth_colormap)
    cv.imshow("Blur", blur)
    cv.imshow('Mask', fgmask)
    if cv.waitKey(25) == ord('q'):
        break
cv.destroyAllWindows()
pipeline.stop()