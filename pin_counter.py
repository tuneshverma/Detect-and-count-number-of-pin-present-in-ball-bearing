import cv2
import statistics
import math
import numpy as np
import argparse
import imutils
import time


def radiuswithcenter(raw_image, dict, sorted_dict_keys, pos, draw=True):
    x_list = []
    y_list = []

    for x in range(len(dict[sorted_dict_keys[pos]])):
        # print(x)
        x_list.append((dict[sorted_dict_keys[pos]])[x][0][0])
    # for y in range(len(dict[sorted_dict_keys[pos]])):
        # print(y)
        y_list.append((dict[sorted_dict_keys[pos]])[x][0][1])

    # x_list = [(dict[sorted_dict_keys[pos]])[x][0][0] for x in range(len(dict[sorted_dict_keys[pos]]))]
    # y_list = [(dict[sorted_dict_keys[pos]])[y][0][1] for y in range(len(dict[sorted_dict_keys[pos]]))]

    c_x = (statistics.mean(x_list))
    c_y = (statistics.mean(y_list))

    cord_x_org = x_list - c_x
    cord_y_org = y_list - c_y
    square_x = cord_x_org**2
    square_y = cord_y_org**2
    dist_square = square_x + square_y
    # print(square_x)

    max_distance_square = max(dist_square)
    # print(max_distance_square)

    radius = math.sqrt(max_distance_square)
    if draw:
        cv2.circle(raw_image, (c_x, c_y), int(radius), (0, 255, 0), 2)
    return c_x, c_y, radius


def auto_canny(image, sigma=1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def pincount(path_image, blur, kernel_size):
    real_raw_image = cv2.imread(path_image)
    # print(real_raw_image.shape)
    t00 = time.time()
    raw_image = imutils.resize(real_raw_image, 640)
    # raw_image_1 = raw_image.copy()
    t0 = time.time()

    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.GaussianBlur(gray, blur, 0)
    t1 = time.time()
    edge_detected_image = auto_canny(gray_filtered)
    # cv2.imshow('Edge', edge_detected_image)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilation = cv2.dilate(edge_detected_image, kernel, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    # erosion = cv2.erode(dilation, kernel, iterations=1)
    t2 = time.time()
    # cv2.imshow('Edge', opening)
    # cv2.waitKey(0)

    _, contours, hierarchy = cv2.findContours(opening.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    dict = {}

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > 8) & (len(approx) < 23) & (area > 150):
            dict[area] = contour
    t3 = time.time()
    # print(dict)

    sorted_dict_keys = sorted(dict.keys())
    # print(sorted_dict_keys)
    pin_list = []

    c_x, c_y, r_b = radiuswithcenter(raw_image, dict, sorted_dict_keys, -3, draw=True)

    s_center_x = []
    s_center_y = []

    t4 = time.time()
    for i in range(len(sorted_dict_keys)):
        if sorted_dict_keys[-1]/200.0 < sorted_dict_keys[i] < 2700.0:
            cc_x, cc_y, r_s = radiuswithcenter(raw_image, dict, sorted_dict_keys, i, draw=False)
            dis_from_cent_sqr = (c_x-cc_x)**2 + (c_y-cc_y)**2
            if r_b**2 > dis_from_cent_sqr:
                already_present = 0
                if s_center_x:
                    for j in range(len(s_center_x)):
                        dist_square = (cc_x - s_center_x[j]) ** 2 + (cc_y - s_center_y[j]) ** 2
                        if dist_square < sorted_dict_keys[-1]/85.0 / 3.14:
                            already_present = 1
                            break
                if already_present == 0:
                    if math.sqrt(sorted_dict_keys[-1]/200.0 / 3.14) < r_s < math.sqrt(2400 / 3.14):
                        pin_list.append(dict[sorted_dict_keys[i]])
                        s_center_y.append(cc_y)
                        s_center_x.append(cc_x)
    t5 = time.time()
    # print('....................')
    # print('filter and edge')
    # print(t2-t1)
    # print('contour couting')
    # print(t3-t2)
    # print('sorting and radius cal')
    # print(t4-t3)
    # print('main big loop')
    # print(t5-t4)
    # print('....................')
    num_of_pin = len(s_center_x)
    # print(num_of_pin)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.drawContours(raw_image, pin_list, -1, (255, 0, 0), 2)
    # cv2.putText(raw_image, 'o', (c_x, c_y), font, 0.1, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.imshow('Objects Detected', raw_image)
    # cv2.waitKey(0)

    # cnts = imutils.grab_contours(dict[sorted_dict_keys[-1]])
    # mask_1 = np.zeros(raw_image_1.shape[:2], dtype="uint8")
    # # loop over the contours
    # # for c in cnts:
    # cv2.drawContours(mask_1, [dict[sorted_dict_keys[-1]]], -1, 255, -1)
    # image = cv2.bitwise_and(raw_image_1, raw_image_1, mask=mask_1)
    # cv2.imshow("Mask", mask_1)
    # cv2.imshow("After", image)
    # cv2.waitKey(0)

    return num_of_pin, c_x, c_y, r_b


def max_pin_count(path_image):
    t1 = time.time()
    num_of_pin, c_x, c_y, r_b = pincount(path_image, (7, 7), (3, 3))
    # time_taken = time.time() - t1
    if num_of_pin == 16:
        max_num = 16
    # num_of_pin_1, c_x_1, c_y_1, r_b_1 = pincount(path_image, (9, 9), (7, 7))
    else:
        num_of_pin_2, c_x_2, c_y_2, r_b_2 = pincount(path_image, (9, 9), (3, 3))
        if num_of_pin_2 == 16:

            max_num = 16
        else:
            num_of_pin_3, c_x_3, c_y_3, r_b_3 = pincount(path_image, (7, 7), (7, 7))
            max_num = (max([num_of_pin, num_of_pin_2, num_of_pin_3]))
    time_taken = time.time() - t1
    print(max_num)
    print(time_taken)


# max_pin_count('good/z_1.png')


parser = argparse.ArgumentParser()
parser.add_argument('--imagepath', help='image path')
args = parser.parse_args()
max_pin_count(args.imagepath)

