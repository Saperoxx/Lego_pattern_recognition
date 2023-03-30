import numpy as np

SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

SOBEL_X_45 = np.array([[2, 1, 0],
                       [1, 0, -1],
                       [0, -1, -2]])

SOBEL_Y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

SOBEL_Y_45 = np.array([[0, 1, 2],
                       [-1, 0, 1],
                       [-2, -1, 0]])

SOBEL_X_REVERSE = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

SOBEL_X_45_REVERSE = np.array([[-2, -1, 0],
                               [-1, 0, 1],
                               [0, 1, 2]])

SOBEL_Y_REVERSE = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

SOBEL_Y_45_REVERSE = np.array([[0, -1, -2],
                               [1, 0, -1],
                               [2, 1, 0]])

WHITER_LOWER = np.array([5, 0, 0])
WHITE_UPPER = np.array([172, 255, 255])

GREEN_LOWER = np.array([66, 72, 89])
GREEN_UPPER = np.array([84, 176, 121])

YELLOW_LOWER = np.array([19, 116, 156])
YELLOW_UPPER = np.array([29, 165, 210])

RED_LOWER = np.array([6, 0, 0])
RED_UPPER = np.array([177, 192, 227])

BLUE_LOWER = np.array([85, 103, 96])
BLUE_UPPER = np.array([150, 255, 227])