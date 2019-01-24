import numpy as np
import cv2 as cv
import pyrealsense2 as rs

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from numpy.core.multiarray import ndarray


def get_3d_point_cloud(colour_img, depth_frame, mask):
    cols, rows, depth = color_image.shape
    points = []

    for i in range(0, rows):
        for j in range(0, cols):
            depth = depth_frame.get_distance(j, i)
            depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [j, i], depth)
            points.append(depth_point)
    return points


size = (640, 480)
fps = 30

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, fps)
config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, fps)

# Start pipeline
profile = pipeline.start(config)
counter = 0

# video writer to save the results
# video_normal = cv.VideoWriter('normal.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
# video_depth = cv.VideoWriter('depth.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
# video_depth_segmented = cv.VideoWriter('depth_segmented.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)
# video_mask = cv.VideoWriter('mask.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, size)


while True:
    # Capture frame-by-frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    colour_frame = frames.get_color_frame()

    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # color_intrin = colour_frame.profile.as_video_stream_profile().intrinsics
    # depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(colour_frame.profile)

    color_image = np.asanyarray(colour_frame.get_data())  # type: object
    depth_image = np.asanyarray(depth_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
    # Our operations on the frame come here

    hsv_image = cv.cvtColor(color_image, cv.COLOR_RGB2HSV)

    # in case of black Hsv values
    light_green = (0, 0, 0)
    dark_green = (180, 255, 40)

    # segment image on a HSV scale (light - lower, dark - higher)
    #light_green = (0, 0, 170)
    #dark_green = (255, 255, 255)
    mask = cv.inRange(hsv_image, light_green, dark_green)  # mask to be applied with only the desired color
    result = cv.bitwise_and(color_image, color_image, mask=mask)
    depth_segmented = cv.bitwise_and(depth_colormap, depth_colormap, mask=mask)

    # mask = cv.Canny(mask, 100, 200)

    dist = cv.distanceTransform(mask, cv.DIST_L2, 3)

    # alternative skeleton technique
    # skel = np.zeros(mask.shape, np.uint8)
    # skel_size = np.size(mask)
    # element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # done = False
    # while (not done):
    #     eroded = cv.erode(mask, element)
    #     temp = cv.dilate(eroded, element)
    #     temp = cv.subtract(mask, temp)
    #     skel = cv.bitwise_or(skel, temp)
    #     mask = eroded.copy()
    #
    #     zeros = skel_size - cv.countNonZero(mask)
    #     if zeros == size:
    #         done = True

    # 1-5 only to visualize Euclidean distance
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    # cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    # cv.imshow('Distance Transform Image', dist)
    # _, dst = cv.threshold(dist, 90, 255, cv.THRESH_BINARY)

    # apply the Laplacian to the euclidean distance transform in order to get the gradient

    # kernel = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]], dtype=np.float32)  # type: ndarray
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)  # type: ndarray
    imgLaplacian = cv.filter2D(dist, cv.CV_32F, kernel)
    cv.normalize(imgLaplacian, imgLaplacian, 0.45, 1.0, cv.NORM_MINMAX)
    _, dst1 = cv.threshold(imgLaplacian, 0.60, 255, cv.THRESH_BINARY_INV)
    _, dst2 = cv.threshold(imgLaplacian, 0.15, 255, cv.THRESH_BINARY_INV)
    _, dst = cv.threshold(imgLaplacian, 0.62, 255, cv.THRESH_BINARY)
    dst = dst - dst1
    dst = dst - dst2
    dst = 255-dst

    # change type to uint to get a binary image in the way opencv works with them
    dst = dst.astype('uint8')
    # dst = cv.Canny(dst, 100, 200)
    # dst = cv.medianBlur(dst, 5)

    struct_element = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

    dst = cv.dilate(dst, struct_element,iterations = 1)
    # dst = cv.erode(dst, struct_element, iterations=2)
    struct_element2 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(dst, struct_element2, iterations=1)
    struct_element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    dst = cv.dilate(dst, struct_element, iterations=1)
    # dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, struct_element)

    im2, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(color_image, contours, -1, (0, 255, 0), 3)

    # dst = cv.medianBlur(dst, 3)

    # dst = cv.bitwise_not(dst)
    # print(dst)

    # cols, rows, depth = color_image.shape
    # points = []
    #
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         depth = depth_frame.get_distance(j, i)
    #         depth_point = rs.rs2_deproject_pixel_to_point(
    #             depth_intrin, [j, i], depth)
    #         points.append(depth_point)

    # Display the resulting frame

    # show the processing

    # Probabilistic Line Transform
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 2)
    # # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(color_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    minLineLength = 30
    maxLineGap = 3
    # lines = cv.HoughLinesP(dst, cv.HOUGH_PROBABILISTIC, np.pi / 180, 30, None, minLineLength, maxLineGap)
    # if lines is not None:
    #     for x in range(0, len(lines)):
    #         for x1, y1, x2, y2 in lines[x]:
    #             # cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
    #             pts = np.array([[x1, y1], [x2, y2]], np.int32)
    #             cv.polylines(color_image, [pts], True, (0, 255, 0))

    #np.savetxt('laplacian.txt', depth_image, fmt="%.2f")
    # print(imgLaplacian)

    cv.imshow('Distance Transform Image Binary', imgLaplacian)
    #print(imgLaplacian)
    # cv.imshow('Morphological skel', skel)
    cv.imshow('skeleton', dst)
    cv.imshow('normal', color_image)
    cv.imshow('mask', mask)
    # cv.imshow('hsv', hsv_image)
    # cv.imshow('depth segmented', depth_segmented)
    # cv.imshow('hsv_segmentation', result)

    # imgLaplaciancp = imgLaplacian.astype('uint8')
    # video_normal.write(color_image)
    # video_depth.write(depth_colormap)
    # video_depth_segmented.write(depth_segmented)
    # video_mask.write(dst)

    if cv.waitKey(25) == ord('q'):
        break
cv.destroyAllWindows()
pipeline.stop()

# video_normal.release()
# video_depth_segmented.release()
# video_depth.release()
# video_mask.release()


# green_image = cv.imread('green_image.jpg')
# hsv_image = cv.cvtColor(green_image, cv.COLOR_BGR2HSV)
#
# h, s, v = cv.split(hsv_image)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
#
# pixel_colors = green_image.reshape((np.shape(green_image)[0]*np.shape(green_image)[1], 3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()
#
# axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Hue")
# axis.set_ylabel("Saturation")
# axis.set_zlabel("Value")
# plt.show()
# plt.savefig('hsv_map.png')
# print(pixel_colors)
