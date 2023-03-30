import cv2
from constants import *

class Image:
    def __init__(self, image_path):
        self.filename = str(image_path).split('/')[-1]
        self.image = cv2.imread(str(image_path))
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_blur = cv2.GaussianBlur(self.image, (7, 7), 0)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

    def scale(self, percent):
        self.width = int(self.width * percent / 100)
        self.height = int(self.height * percent / 100)
        dsize = (self.width, self.height)
        self.image = cv2.resize(self.image, dsize=dsize)
        self.image_gray = cv2.resize(self.image_gray, dsize=dsize)
        self.image_blur = cv2.resize(self.image_blur, dsize=dsize)

    def filter_to_edges(self):
        self.image_blur = cv2.GaussianBlur(self.image_blur, (7, 7), 0)

        # Sobel Edges Detection
        sobelx = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_X)
        sobely = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_Y)
        sobelx45 = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_X_45)
        sobely45 = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_Y_45)
        sobelxrev = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_X_REVERSE)
        sobelyrev = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_Y_REVERSE)
        sobelx45rev = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_X_45_REVERSE)
        sobely45rev = cv2.filter2D(src=self.image_blur, ddepth=-1, kernel=SOBEL_Y_45_REVERSE)

        sobel_final = sobely + sobelx + sobelx45 + sobely45 + sobelxrev + sobelyrev + sobelx45rev + sobely45rev
        sobel_final = cv2.dilate(sobel_final, kernel=(2, 2), iterations=1)
        sobel_final = cv2.erode(sobel_final, kernel=(2, 2), iterations=1)
        edges = cv2.Canny(image=sobel_final, threshold1=100, threshold2=200)
        return edges

    def find_elements(self, edges):
        temp = self.image_gray

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(temp, contours, -1, [0, 0, 0], thickness=6)
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(temp, contours, -1, [255, 255, 255], thickness=cv2.FILLED)

        h, w = temp.shape[:2]
        ret, temp = cv2.threshold(temp, 220, 255, cv2.THRESH_BINARY_INV)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(temp, mask, (0, 0), 255)
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(temp, contours, -1, [0, 0, 0], thickness=15)
        contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ret, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY_INV)

        elements = []
        elements_rgb = []

        for _, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w < 1000 and h < 1000:
                if w > 50 and h > 50:

                    shape = temp[y - 10:y + h + 10, x - 10:x + w + 10]
                    shape_rgb = self.image[y - 10:y + h + 10, x - 10:x + w + 10]

                    elements.append(shape)
                    elements_rgb.append(shape_rgb)

        return elements, elements_rgb

    def copy(self):
        return self.image
