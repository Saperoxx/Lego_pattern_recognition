from typing import List
import numpy as np
import cv2
import os


def perform_processing(image: np.ndarray) -> List[int]:
    # image = cv2.imread("img_018.jpg")
    #Zmienne
    square_counter = 0
    rectangle_counter = 0
    tetris_counter = 0
    zetka_counter = 0
    elka_counter = 0
    red_counter = 0
    yellow_counter = 0
    blue_counter = 0
    green_counter = 0
    white_counter = 0
    mixed_counter = 0
    #Scaling photos
    scale_percent = 30
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dsize = (width, height)
    image = cv2.resize(image, dsize = dsize)
    original_image = image.copy()
    #Filtering
    final = image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(image, (7, 7), 0)
    # Sobel Edge Detection
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    sobel_x_45 = np.array([[2, 1, 0],
                           [1, 0, -1],
                           [0, -1, -2]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    sobel_y_45 = np.array([[0, 1, 2],
                           [-1, 0, 1],
                           [-2, -1, 0]])
    sobel_x_reverse = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    sobel_x_45_reverse = np.array([[-2, -1, 0],
                                   [-1, 0, 1],
                                   [0, 1, 2]])
    sobel_y_reverse = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
    sobel_y_45_reverse = np.array([[0, -1, -2],
                                   [1, 0, -1],
                                   [2, 1, 0]])
    sobelx = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_x)
    sobely = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_y)
    sobelx45 = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_x_45)
    sobely45 = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_y_45)
    sobelxrev = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_x_reverse)
    sobelyrev = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_y_reverse)
    sobelx45rev = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_x_45_reverse)
    sobely45rev = cv2.filter2D(src=img_blur, ddepth=-1, kernel=sobel_y_45_reverse)

    sobel_final = sobely + sobelx + sobelx45 + sobely45 + sobelxrev + sobelyrev + sobely45rev + sobely45rev

    sobel_final = cv2.dilate(sobel_final, kernel=(2, 2), iterations=1)
    sobel_final = cv2.erode(sobel_final, kernel=(2, 2), iterations=1)
    edges = cv2.Canny(image=sobel_final, threshold1=100, threshold2=200)

    #Szukanie konturow
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gray, contours, -1, [0, 0, 0], thickness=6)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gray, contours, -1, [255, 255, 255], thickness=cv2.FILLED)

    h, w = img_gray.shape[:2]
    ret, img_gray = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_gray, mask, (0, 0), 255)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gray, contours, -1, [0, 0, 0], thickness= 15)
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ret, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("img_gray",img_gray)
    # cv2.waitKey()
    elements = []
    elements_rgb = []
    # index = 0
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 1000 and h < 1000:
            if w > 50 and h > 50:
            # print("x: ", x)
            # print("y: ", y)
            #     name = "square_" + str(index) + ".jpg"
                shape = img_gray[y - 10:y+h + 10, x - 10:x+w + 10]
                shape_rgb = image[y - 10:y+h + 10, x - 10:x+w + 10]
                # cv2.imwrite(name, shape)
                elements.append(shape)
                elements_rgb.append(shape_rgb)
                #cv2.imwrite("image.jpg", shape_rgb)
                #cv2.rectangle(final,(x,y),(x+w,y+h),(0,155,255),4)
                #cv2.imshow("final", final)
                # index += 1

    # print(len(elements))
    path = "processing/train_pictures"
    dirs = os.listdir(path)
    dirs.sort()
    for element in elements:
        cv2.waitKey()
        best = 1000000
        contours, hierarchy = cv2.findContours(element, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(element, contours, -1, 127, thickness=4)
        cv2.imshow('current_find', element)
        for current_name in dirs:
            current_path = str(path + '/' + current_name)
            current_img = cv2.imread(current_path, 0)
            current_img = cv2.medianBlur(current_img, 7)
            contours_2, hierarchy = cv2.findContours(current_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(current_img, contours_2, -1 , 127, 3)
            cv2.imshow('current_test_image', current_img)
            #print("2", len(contours_2))
            cv2.waitKey()
            score = None
            score = cv2.matchShapes(contours[0], contours_2[0], 1, 0)
            if score < best:
                the_best_name = current_name
                the_best = current_img
                best = score
        counter = 0
        if best < 1:
            size = len(the_best_name)
            the_best_name = the_best_name[:size - 6]
            # cv2.imshow("best", the_best)
            print("best score: ", best)
            if the_best_name == "square":
                square_counter += 1
            if the_best_name == "rectangle":
                rectangle_counter += 1
            if the_best_name == "tetris":
                tetris_counter += 1
            if the_best_name == "zetka":
                zetka_counter += 1
            if the_best_name == "elka":
                elka_counter += 1

    white_lower = np.array([5, 0, 0])
    white_upper = np.array([172, 255, 255])

    green_lower = np.array([66, 72, 89])
    green_upper = np.array([84, 176, 121])

    yellow_lower = np.array([19, 116, 156])
    yellow_upper = np.array([29, 165, 210])

    red_lower = np.array([6, 0, 0])
    red_upper = np.array([177, 192, 227])

    blue_lower = np.array([85, 103, 96])
    blue_upper = np.array([150, 255, 227])
    for element_rgb in elements_rgb:
        blue = 0
        red = 0
        green = 0
        white = 0
        yellow = 0
        mix = 0
        #white
        mask = cv2.inRange(element_rgb, white_lower, white_upper)
        mask = cv2.medianBlur(mask, 7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 127, 3)
        print(len(contours))
        if len(contours) != 0:
            for contour in contours:
                print(cv2. contourArea(contour))
                if cv2.contourArea(contour) > 10 and cv2.contourArea(contour) < 10000:
                    white = 1
        #yellow
        element_hsv = cv2.cvtColor(element_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(element_hsv, yellow_lower, yellow_upper)
        mask = cv2.medianBlur(mask, 7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 127, 3)
        print(len(contours))
        if len(contours) != 0:
            for contour in contours:
                print(cv2.contourArea(contour))
                if cv2.contourArea(contour) < 10000:
                    yellow = 1
        # cv2.imshow("mask", mask)
        # cv2.imshow("element_rgb", element_rgb)
        # cv2.waitKey()
        #green
        element_hsv = cv2.cvtColor(element_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(element_hsv, green_lower, green_upper)
        mask = cv2.medianBlur(mask, 7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 127, 3)
        print(len(contours))
        if len(contours) != 0:
            for contour in contours:
                print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 10 and cv2.contourArea(contour) < 10000:
                    green = 1
        #red
        element_hsv = cv2.cvtColor(element_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(element_hsv, red_lower, red_upper)
        mask = cv2.medianBlur(mask, 7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 127, 3)
        print(len(contours))
        if len(contours) != 0:
            for contour in contours:
                print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 10000:
                    red = 1
        #blue
        element_hsv = cv2.cvtColor(element_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(element_hsv, blue_lower, blue_upper)
        mask = cv2.medianBlur(mask, 7)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 127, 3)
        print(len(contours))
        if len(contours) != 0:
            for contour in contours:
                print(cv2.contourArea(contour))
                if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 10000:
                    blue = 1
        if red + blue + yellow + green + white > 1:
            mix = 1
            red = 0
            blue = 0
            yellow = 0
            white = 0
            green = 0
        red_counter += red
        yellow_counter += yellow
        blue_counter += blue
        green_counter += green
        white_counter += white
        mixed_counter += mix
        # cv2.imshow("mask", mask)
        # cv2.imshow("element_rgb", element_rgb)
        # cv2.waitKey()
    # print("blue: ", blue_counter)
    # print("yellow: ", yellow_counter)
    # print("green: ", green_counter)
    # print("red: ", red_counter)
    # print("white: ", white_counter)
    # print("mixed: ", mixed_counter)
    cv2.destroyAllWindows()

    # TODO: add image processing here
    return [rectangle_counter, tetris_counter, elka_counter, square_counter, zetka_counter, red_counter, green_counter, blue_counter, white_counter, yellow_counter, mixed_counter]
