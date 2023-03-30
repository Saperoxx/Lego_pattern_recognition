import os
import cv2
from constants import *
import math
from Image import Image


def load_labels(path):
    dirs = os.listdir(path)
    dirs.sort()
    labels = []

    for current_name in dirs:
        current_path = str(path + '/' + current_name)
        current_img = cv2.imread(current_path, 0)
        current_img = cv2.medianBlur(current_img, 7)
        current_img = cv2.medianBlur(current_img, 7)
        contours, hierarchy = cv2.findContours(current_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        labels.append((current_img, contours, current_name))

    return labels

def is_color(color, element, min_area, max_area):

    if color == 'white':
        lower = WHITER_LOWER
        upper = WHITE_UPPER
    elif color == 'yellow':
        lower = YELLOW_LOWER
        upper = YELLOW_UPPER
        element = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)
    elif color == 'blue':
        lower = BLUE_LOWER
        upper = BLUE_UPPER
        element = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)
    elif color == 'red':
        lower = RED_LOWER
        upper = RED_UPPER
        element = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)
    elif color == 'yellow':
        lower = YELLOW_LOWER
        upper = YELLOW_UPPER
        element = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)
    elif color == 'green':
        lower = GREEN_LOWER
        upper = GREEN_UPPER
        element = cv2.cvtColor(element, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(element, lower, upper)
    mask = cv2.medianBlur(mask, 7)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 127, 3)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            return 1
    return 0

path = "processing/train_pictures"
labels = load_labels(path)

def perform_processing(image: Image):
    shapes = {'square' : 0,
              'rectangle' : 0,
              'tetris' : 0,
              'z_shape' : 0,
              'L_shape' : 0}

    colors = {'blue' : 0,
            'red' : 0,
            'green' : 0,
            'white' : 0,
            'yellow' : 0,
            'mix' : 0}



    image.scale(30)
    edges = image.filter_to_edges()
    elements, elements_rgb = image.find_elements(edges)

    for element in elements:
        highest_ratio = math.inf
        contours, hierarchy = cv2.findContours(element, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for _, label in enumerate(labels):
            score = cv2.matchShapes(contours[0], label[1][0], 1, 0)
            if score < highest_ratio:
                the_best_name = label[2]
                highest_ratio = score

        if highest_ratio < 1:
            size = len(the_best_name)
            the_best_name = the_best_name[:size - 6]
            shapes[the_best_name] += 1


    for element_rgb in elements_rgb:

        white = is_color('white',element_rgb, 10, 10000)
        yellow = is_color('yellow',element_rgb, 0, 10000)
        blue = is_color('blue',element_rgb, 100, 10000)
        green = is_color('green',element_rgb, 10, 10000)
        red = is_color('red',element_rgb, 100, 10000)
        mix = 0

        if red + blue + yellow + green + white > 1:
            mix = 1
            red = 0
            blue = 0
            yellow = 0
            white = 0
            green = 0
        colors['red'] += red
        colors['yellow'] += yellow
        colors['blue'] += blue
        colors['green'] += green
        colors['white'] += white
        colors['mix'] += mix

    cv2.destroyAllWindows()
    return [shapes, colors]


