import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import time
from math import *
from scipy.spatial import distance as dist

from skimage.metrics import structural_similarity as ssim
from itertools import combinations
from scipy.stats import mode


# Define Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD_FONT = cv2.FONT_HERSHEY_TRIPLEX
# Image directories
IMG_DIR = "test_imgs"
VID_DIR = "test_vids"
OUT_DIR = "output"
TEMPLATE_DIR = "template_imgs"

# Basic Image Operations
# WHITE
# WHITE_MIN = [0, 0, 100]
# WHITE_MAX = [180, 50, 255]
WHITE_MIN = [0, 0, 150]
WHITE_MAX = [180, 50, 255]


# # Red
RED_MIN = [0, 50, 10]
RED_MAX = [10, 255, 255]

# # Green
# H: 148deg, S: 55%, V: 64%
# 141, 19, 65 => 70.5  140 166
GREEN_MIN = [40, 15, 10]
GREEN_MAX = [105, 255, 205]

BW_THRESH = 100

# Area thresholds for contours
AREA_THRESH = 1000
AREA_THRESH_SYMBOL = 50

SIGMA_EXP = 0.5

# Distance between cards
CARD_MIN_DIST = 10

# Distance between Symbols
SYMBOL_MIN_DIST = 0.0

# COLORS
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# LABELS
COLOR_STR = ['Red', 'Green', 'Purple']
FILL_STR = ['Empty', 'Hash', 'Solid']
NUM_STR = ['1', '2', '3']
SHAPE_STR = ['Diamond', 'Oval', 'Squiggle']


# Utility Functions
def show(img_in_array):
    if not isinstance(img_in_array, list):
        img_in_array = [img_in_array]

    length = len(img_in_array)

    if length == 0:
        print("No Image Loaded")
        return

    height = int(np.rint(np.sqrt(length)))
    width = int(np.ceil(length / height))

    for i in range(length):
        img = img_in_array[i]
        if img is None:
            print("No Image Loaded")
            break
        if len(img.shape) == 2:
            plt.subplot(height, width, i + 1), plt.imshow(img, 'gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(height, width, i + 1), plt.imshow(img, 'gray')
    plt.show()


def crop_frame(img, x_amt=10, y_amt=50, rectLine=3):
    w, h = img.shape[:2]
    y_min, y_max = y_amt, h - y_amt
    x_min, x_max = x_amt, w - x_amt

    return img[y_min+rectLine:y_max-rectLine, x_min+rectLine:x_max-rectLine]


def time_function(function, iter):
    '''
    Input: Function, iterations
    Output: Average time for n cycles
    Method: Pass in a function with proper args and get the ave run time
    for n runs.
    '''
    start_time = time.time()
    for x in range(iter):
        function
    end_time = time.time()
    avg_time = (end_time - start_time) / iter
    return avg_time


def setup_image(n, return_all=False):
    # cwd = os.getcwd()
    # path = os.path.join(cwd, IMG_DIR)
    # os.chdir(path)

    path = "/home/marc/Documents/Projects/SET/"
    path = os.path.join(path, IMG_DIR)

    test_pics = []
    img = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            test_pics.append(os.path.join(path, file))

    for i in range(len(test_pics)):
        img.append(cv2.imread(test_pics[i], 1))

    if return_all:
        return img
    else:
        return img[n]


def setup_templates():
    path = "/home/marc/Documents/Projects/SET/"
    path = os.path.join(path, TEMPLATE_DIR)

    test_pics = []
    img = []

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img_path = os.path.join(path, file)
            loaded_img = bw_filter(cv2.imread(img_path, 1))
            img_name = file[0:len(file) - 4]

            img.append([loaded_img, img_name])
    return np.array(img)


def minmaxNorm(array, newmax=1.0):
    """ Takes a np.array and normalizes it between 0 and newmax. """
    arrmin = np.min(array)
    arrmax = np.max(array)

    if arrmax - arrmin == 0.0:
        return 0.0 * array
    else:
        return newmax*(array-arrmin)/(arrmax-arrmin)


# Basic Image Operations
def hsv_filter(img, low, high, opened=True):
    '''
    Input: Image, S Value, V Value
    Output: Binary filtered image
    Method:
    Find white cards using HSV filtering (more robust against different backgrounds)
    H: 0 - 180, S: 0 - 255, V: 0 - 255
    '''
    blur_img = cv2.GaussianBlur(img, (9, 9), 0)  # Little bit of blur

    low = np.array(low, np.uint8)
    high = np.array(high, np.uint8)
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    hsv_filtered = cv2.inRange(hsv_img, low, high)

    if opened:
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(hsv_filtered, cv2.MORPH_OPEN, kernel)
    else:
        return hsv_filtered


def bw_filter(img):
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return bw


def threshold_img(img, inverse=True):
    # bw = bw_filter(img)

    if inverse:
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 5)
    else:
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 5)
    return th2


def threshold_binary(img, cutoff):
    bw = bw_filter(img)
    ret, thresh1 = cv2.threshold(bw, cutoff, 255, cv2.THRESH_BINARY)
    return thresh1


def flood_fill(thresh_img):
    im_floodfill = thresh_img.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = thresh_img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = thresh_img | im_floodfill_inv

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)

    return opening


# Image Processing
def is_rect(pts):
    pts = np.array(pts).reshape(4, 2)
    pts = order_pts(pts)
    x_pts = np.append(pts[:, 0], pts[0, 0])
    y_pts = np.append(pts[:, 1], pts[0, 1])

    # Calculate all four angles
    theta = np.arctan2(np.diff(x_pts), np.diff(y_pts))

    # Correct angles based on initial line segment
    theta = abs(angle_trunc(theta - theta[0]))
    # print (theta)
    angle_min = 0.7 * pi / 2
    angle_max = 1.3 * pi / 2

    angle_test = np.where(np.logical_or(np.logical_and(theta >= 0, theta < 0.1),
                                        np.logical_and(theta > angle_min, theta < angle_max)))

    if len(angle_test[0]) == 4:
        return True

    else:
        return False


def order_pts(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def angle_trunc(a):
    """Helper function to map all angles onto [-pi, pi]

    Arguments:
        a(float): angle to truncate.

    Returns:
        angle between -pi and pi.
    """
    return ((a + pi / 2) % (pi / 2 * 2)) - pi / 2


def warp_cards(c, img):
    # Playing cards are 3.5in by 2.5in
    # Fix this ratio, it's a bit off (996x660)
    new_width = 100
    new_height = (2 / 3.) * new_width

    pts = c.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    top_len = np.linalg.norm(tl-tr)
    side_len = np.linalg.norm(tr-br)

    if side_len > top_len:
        rect = np.roll(rect, 1, axis=0)

    (tl, tr, br, bl) = rect

    maxWidth = int(new_width)
    maxHeight = int(new_height)

    # construct our destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped, min(side_len, top_len)


def largest_indices(arr, n):
    # Adapted from:
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array/38884051#38884051
    # https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi
    flat_indices = np.argpartition(arr.ravel(), -n)[-n:]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    ind = (row_indices, col_indices)
    return ind


def mse(x, y):
    return np.linalg.norm(x - y)


def check_overlap(cnts, distance):
    new_cnts = []
    points = []
    for c in cnts:
        m_sym = cv2.moments(c)
        sym_x = int(m_sym["m10"] / m_sym["m00"])
        sym_y = int(m_sym["m01"] / m_sym["m00"])

        new_pt = np.array([sym_x, sym_y])
        if points:
            # print(new_pt, points)
            dist = np.linalg.norm(new_pt - np.array(points), axis=1)

            # print(np.all(dist > CARD_MIN_DIST))
            if np.all(dist > distance):
                new_cnts.append(c)

        points.append([sym_x, sym_y])

    return new_cnts


# Video Processing
def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.

    """
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


class Card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self, c, warp):
        self.warp = warp  # 200x300, flattened, blurred image
        self.bw_warp = bw_filter(warp)
        self.contour = c  # Contour of card
        self.threshold = threshold_img(self.bw_warp)

        self.M = cv2.moments(c)  # compute the center of the contour
        self.x = int(self.M["m10"] / self.M["m00"])
        self.y = int(self.M["m01"] / self.M["m00"])
        self.center = (self.x, self.y)  # Center point of card

        self.contours = []

        '''
        IDX     Number     Shape       Fill        Color
        0       1          Diamond     Empty       Red
        1       2          Oval        Hatch       Green
        2       3          Squiggle    Solid       Purple
        '''

        self.number = self.count_number()  # Best match # of symbols, (integer)
        self.shape_str = 'None'
        self.fill_str = 'None'
        self.color_str = 'None'
        self.number_str = str(self.number)

        self.symbol_img = None
        self.shape = None
        self.fill = None
        self.color = None

        self.process_card()

        # self.bw_sym_img = None
        # self.thresh_sym = None
        # self.filled_sym = None

        # self.test = 0

    def process_card(self):
        self.temp = 0

        if self.number == 0:
            self.symbol_img = None
            self.shape = None
            self.fill = None
            self.color = None
        else:
            self.symbol_img = self.find_symbol()
            self.find_shape()
            self.find_fill()
            self.find_color()

    def count_number(self):
        cnts, _ = cv2.findContours(self.threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            area = cv2.contourArea(c)

            points = []
            if area > AREA_THRESH_SYMBOL:
                # print(area)
                m_sym = cv2.moments(approx)
                sym_x = int(m_sym["m10"] / (m_sym["m00"] + 1e-5))
                sym_y = int(m_sym["m01"] / (m_sym["m00"] + 1e-5))
                new_pt = np.array([sym_x, sym_y])

                if points:
                    distance = np.linalg.norm(new_pt - np.array(points), axis=1)
                    # print(distance)
                    if np.all(distance > SYMBOL_MIN_DIST):
                        self.contours.append(approx)
                else:
                    self.contours.append(approx)

                points.append([sym_x, sym_y])

        if len(self.contours) > 0:
            m_sym = cv2.moments(self.contours[0])
            sym_x = int(m_sym["m10"] / (m_sym["m00"] + 1e-5))
            sym_y = int(m_sym["m01"] / (m_sym["m00"] + 1e-5))
            self.sym_x = sym_x
            self.sym_y = sym_y
            self.sym_center = (sym_x, sym_y)

            self.rect = cv2.boundingRect(self.contours[0])
            self.sym_width = self.rect[2]
            self.sym_height = self.rect[3]


        return np.clip(len(self.contours), 0, 3)

    def find_symbol(self):
        height = int(np.ceil(self.sym_height / 2))
        width = int(np.ceil(self.sym_width / 2))
        pad = 3

        w_min = np.clip(self.sym_x - width - pad, 0, None)
        w_max = np.clip(self.sym_x + width + pad, 0, None)

        h_min = np.clip(self.sym_y - height - pad, 0, None)
        h_max = np.clip(self.sym_y + height + pad, 0, None)

        symbol_img = self.warp[h_min:h_max, w_min:w_max, :]
        return symbol_img

    def draw_contours(self, points=False):
        copy = self.warp.copy()

        if self.contours is not None:
            cv2.drawContours(copy, self.contours, -1, (255, 0, 0), 1)

            if points:
                for pts in self.contours:
                    # print len(pts)
                    for centers in pts:
                        center = centers[0]
                        cv2.circle(copy, (center[0], center[1]), 1, (0, 255, 0))
        return copy

    def find_shape(self):
        side_array = []
        if self.contours is not None:
            for pts in self.contours:
                side_array.append(len(pts))

        side_mean = np.mean(side_array)
        self.bw_sym_img = bw_filter(self.symbol_img)
        self.thresh_sym = threshold_img(self.bw_sym_img)
        self.filled_sym = flood_fill(self.thresh_sym)

        if 3.5 < side_mean < 6.0:
            shape_str = 'Diamond'
            shape_idx = 0

        else:
            err = []
            for shapes in TEMPLATE_ARRAY:
                cmp_shape = self.resize_to_shape(shapes[0], self.filled_sym)
                err.append(ssim(cmp_shape, self.filled_sym))
            err = np.array(err)
            shape_idx = np.argmax(err) + 1
            shape_str = TEMPLATE_ARRAY[np.argmax(err), 1]

        self.shape = shape_idx
        self.shape_str = shape_str

    def find_fill(self):
        width = 4
        height = 4

        h, w, d = self.symbol_img.shape

        h_min = int(h/2) - height
        h_max = int(h/2) + height
        w_min = int(w/2) - width
        w_max = int(w/2) + width

        self.normalize_patch = self.warp[0:2 * height, 0:2 * width, :]
        self.fill_patch = self.symbol_img[h_min:h_max, w_min:w_max]

        normal_mean = np.mean(self.normalize_patch, axis=(0,1))
        patch_mean = np.mean(self.fill_patch, axis=(0,1))

        distance = mse(normal_mean, patch_mean)

        '''normal_hsv = cv2.cvtColor(self.normalize_patch, cv2.COLOR_BGR2HSV)
        self.norm_h_flat = normal_hsv[:,:,0].flatten()
        self.top_norm = np.mean(self.norm_h_flat[np.argsort(self.norm_h_flat)][-5:])

        fill_hsv = cv2.cvtColor(self.fill_patch, cv2.COLOR_BGR2HSV)
        self.fill_h_flat = fill_hsv[:,:,0].flatten()
        self.top_fill = np.mean(self.fill_h_flat[np.argsort(self.fill_h_flat)][-5:])

        # distance2 = mse(self.top_norm, self.top_fill)
        # self.test = distance2
        # print(distance, self.top_fill, self.top_norm)'''

        sym_edge = cv2.Canny(self.fill_patch, 10, 50)
        fill_cnt = (sym_edge > 0).sum()
        # self.temp = np.round(distance,0)
        self.temp = (fill_cnt, np.round(distance, 0))
        self.edged = sym_edge


        if distance > 100:
            # fill equals full
            self.fill = 2
            self.fill_str = 'Solid'

        elif 0 <= distance <= 30:
        # elif 0 <= fill_cnt <= 5:
            # fill equals empty
            self.fill = 0
            self.fill_str = 'Empty'

        else:
            # fill equals hash
            self.fill = 1
            self.fill_str = 'Hash'

    def draw_center(self):
        copy = self.warp.copy()
        if self.sym_center:
            cv2.drawMarker(copy, self.sym_center, (255, 0, 0))
        return copy

    def return_symbol(self):
        return self.symbol_img

    def resize_to_shape(self, img_a, img_b):
        # img_a = to be resized
        height, width = img_b.shape
        return cv2.resize(img_a, (width, height))

    def find_color(self):
        """
        # Check for Red
        self.red_check = hsv_filter(self.sym_mask, RED_MIN, RED_MAX, opened=False)
        red_sum = (self.red_check > 0).sum()/self.red_check.size

        if red_sum >= 0.10:
            self.color = 0
            self.color_str = 'Red'

        # Check for Green
        self.green_check = hsv_filter(self.sym_mask, GREEN_MIN, GREEN_MAX, opened=False)
        green_sum = (self.green_check > 0).sum()/self.green_check.size
        if green_sum >= 0.20:
            self.color = 1
            self.color_str = 'Green'

        self.color = 2
        self.color_str = 'Purple'
        """

        self.sym_mask = cv2.bitwise_and(self.symbol_img, self.symbol_img, mask=self.thresh_sym)
        sym_hsv = cv2.cvtColor(self.sym_mask, cv2.COLOR_BGR2HSV)
        norm_hsv = cv2.cvtColor(self.normalize_patch, cv2.COLOR_BGR2HSV)

        # sym_mean = np.mean(sym_hsv[:, :, 0], axis=(0, 1))
        norm_mean = np.mean(norm_hsv[:, :, 0], axis=(0, 1))

        hue = sym_hsv[:, :, 0]
        sym_mean = hue[hue > 0].flatten()
        # print(sym_mean)
        purple_mean = ((sym_mean > 115) & (sym_mean < 160)).sum()
        green_mean = ((sym_mean > 60) & (sym_mean < 90)).sum()
        red_mean = ((sym_mean > 1) & (sym_mean < 10)).sum() + (sym_mean > 175).sum()
        color_choice = np.array([red_mean, green_mean, purple_mean])

        self.color = np.argmax(color_choice)
        self.color_str = COLOR_STR[self.color]


class ProcessScene:
    """Structure to store information about query cards in the camera image."""

    def __init__(self, image_in):
        self.cards = []  # Array of Card Classes

        self.img = image_in
        self.hsv_img = hsv_filter(self.img, WHITE_MIN, WHITE_MAX)
        self.bw_img = bw_filter(self.img)
        # self.edge_img = self.find_edges()

        self.contours = []
        self.card_array = []
        self.solution = []
        self.card_count = 0

        self.card_parameters = []
        self.card_info = []

    def process(self, only_count=False):
        self.find_cards()
        self.card_count = len(self.contours)

        if not only_count:
            for cnt in self.contours:
                self.ID_cards(cnt)

            self.generate_cards()

    def solve(self):
        # cards = self.generate_cards()
        cards = np.array(list(self.card_parameters))

        res = []
        for card in combinations(enumerate(cards), 3):
            test = np.array(card)
            indexes = test[:, 0]
            card = test[:, 1]

            if self.are_valid_set(card):
                self.solution = indexes

    def find_cards(self):
        cnts, _ = cv2.findContours(self.hsv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        points = []

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            area = cv2.contourArea(c)
            # if our approximated contour has four points, it's a card...
            # ony keep the ones that are square
            # and the contours that aren't too close to other contours
            if len(approx) == 4 and area > AREA_THRESH:
                if is_rect(approx):
                    m_sym = cv2.moments(approx)
                    sym_x = int(m_sym["m10"] / m_sym["m00"])
                    sym_y = int(m_sym["m01"] / m_sym["m00"])
                    new_pt = np.array([sym_x, sym_y])

                    if points:
                        distance = np.linalg.norm(new_pt - np.array(points), axis=1)
                        if np.all(distance > CARD_MIN_DIST):
                            self.contours.append(approx)
                    else:
                        self.contours.append(approx)

                    points.append([sym_x, sym_y])

    def draw_contours(self):
        """
        Return contours draw on the image
        """
        copy = self.img.copy()
        if self.contours is not None:
            cv2.drawContours(copy, self.contours, -1, (0, 255, 0), 7)

        return copy

    def draw_symbols(self, draw_num=False, draw_shape=False, draw_fill=False, draw_color=False):
        """
        Return contours draw on the card
        """

        copy = self.img.copy()
        # if self.contours is not None:
        #     cv2.drawContours(copy, self.contours, -1, (0, 255, 0), 5)

        for card in self.card_array:
            # cv2.circle(copy, card.center, 5, (0, 255, 0))

            offset = 0
            font_size = .006 * self.card_height
            offset_amt = int(33 * font_size)

            thickness = 1

            if draw_num:
                cv2.putText(copy, card.number_str, (card.x, card.y), BOLD_FONT, font_size, BLUE, thickness)

            if draw_shape:
                offset += offset_amt
                # cv2.putText(copy, card.shape_str, (card.x - 300, card.y - 100), BOLD_FONT, 0.75, BLUE, 3)
                cv2.putText(copy, card.shape_str, (card.x, card.y + offset), BOLD_FONT, font_size, BLUE, thickness)

            if draw_fill:
                offset += offset_amt
                # cv2.putText(copy, card.fill_str, (card.x - 300, card.y - 100), BOLD_FONT, 3.0, BLUE, 3)
                cv2.putText(copy, card.fill_str, (card.x, card.y + offset), BOLD_FONT, font_size, BLUE, thickness)
                # cv2.putText(copy, str(card.temp), (card.x, card.y + offset), FONT, 0.75, RED, 2)
                # print(card.fill_str)

            if draw_color:
                offset += offset_amt
                # cv2.putText(copy, card.color_str, (card.x - 300, card.y - 100), BOLD_FONT, 3.0, BLUE, 3)
                cv2.putText(copy, card.color_str, (card.x, card.y + offset), BOLD_FONT, font_size, BLUE, thickness)

        return copy

    def find_edges(self):
        blur_img = cv2.GaussianBlur(self.img, (9, 9), 0)  # Little bit of blur
        edges = cv2.Canny(blur_img, 100, 200)
        return edges

    def ID_cards(self, c):
        # Computed Flattened Card
        warped, self.card_height = warp_cards(c, self.img)
        new_card = Card(c, warped)
        self.card_array.append(new_card)

    def draw_descriptors(self):
        """
        Number     Shape       Fill        Color
        0          Diamond     Empty       Red
        1          Oval        Hatch       Green
        2          Squiggle    Solid       Purple
        """

        shape_names = ['diamond', 'oval', 'squiggle']
        fill_names = ['empty', 'hatch', 'solid']
        color_names = ['red', 'green', 'purple']

        color = (255, 0, 0)
        copy = self.img.copy()
        for card in self.card_array:
            (x, y) = card.center

            number_str = str(card.number)
            shape_str = shape_names[card.shape]
            fill_str = fill_names[card.fill]
            color_str = color_names[card.color]

            text_str1 = number_str + ' ' + color_str
            text_str2 = fill_str + ' ' + shape_str

            cv2.putText(copy, text_str1, (x, y), FONT, 0.75, color, 2)
            cv2.putText(copy, text_str2, (x,y+15), FONT, 0.75, color, 2)
            # cv2.putText(copy, shape_str, (x, y), FONT, 0.75, color, 2)
        return copy

    def print_descriptors(self):
        shape_names = ['diamond', 'oval', 'squiggle']
        fill_names = ['empty', 'hatch', 'solid']
        color_names = ['red', 'green', 'purple']

        for card in self.card_array:
            number_str = str(card.number)
            shape_str = shape_names[card.shape]
            fill_str = fill_names[card.fill]
            color_str = color_names[card.color]

            text_str = number_str + ' ' + color_str + ' ' + fill_str + ' ' + shape_str
            # print(text_str)

    def generate_cards(self):
        """
        Return array of card parameters
        """
        info = []

        for card in self.card_array:
            self.card_parameters.append([card.number, card.shape, card.fill, card.color])
            info.append([card.x, card.y, card.contour, card.number, card.shape, card.fill, card.color])

        if len(self.card_array):
            info = np.array(info)
            # self.card_info = info
            #
            idx = np.argsort(info[:, 0] + 2*info[:, 1])
            self.card_info = info[idx]

        else:
            self.card_info = info

    def show_solution(self):
        copy = self.img.copy()
        green = (0, 255, 0)
        red = (0, 0, 255)


        if self.solution is None:
            # print('here')
            text_str = 'No Solution Found'
            h,w,d = copy.shape
            x = int(w/2) - 25
            y = int(h/2)
            cv2.putText(copy, text_str, (x, y), FONT, 1.5, red, 3)
        else:
            for idx in self.solution:
                card = self.card_array[idx]
                cnt = card.contour
                cv2.drawContours(copy, [cnt], 0, green, 3)
        return copy

    @staticmethod
    def are_valid_set(cards):
        for i in range(4):
            if len(set(card[i] for card in cards)) == 2:
                return False
        return True


class PlayGame:
    """Structure to store information about Card IDs"""

    def __init__(self, image_in, scenes):
        self.img = image_in
        self.scenes = scenes

        self.cards = []
        self.solutions = []
        self.solution = None
        self.solved = False

    def solve(self):
        if len(self.scenes) == 1:
            self.solve_single()
        else:
            self.solve_multiple()

    def solve_single(self):
        card_array = np.array(self.scenes[0].card_parameters)

        for card in combinations(enumerate(card_array), 3):
            test = np.array(card)
            indexes = test[:, 0]
            card = test[:, 1]

            if self.are_valid_set(card):
                # print(card)
                self.solutions.append(indexes)
                self.solution = indexes
                self.solved = True

    def solve_multiple(self):
        card_array = []
        for scene in self.scenes:
            card_array.append(scene.card_parameters)

        card_array = np.array(card_array)
        if card_array.ndim > 1:
            if card_array.shape[1] > 0:
                same_cards = (card_array[0] == card_array[1]).all(axis=1) & \
                             (card_array[1] == card_array[2]).all(axis=1)

                for card in combinations(enumerate(card_array[0][same_cards]), 3):
                    test = np.array(card)
                    indexes = test[:, 0]
                    card = test[:, 1]

                    if self.are_valid_set(card):
                        self.solutions.append(indexes)
                        self.solution = indexes
                        self.solved = True

    def draw_solution(self):
        copy = self.img.copy()
        if self.solution is None:
            text_str = 'No Solution Found'
            h, w, d = copy.shape
            x = int(w / 2) - 100
            y = int(h / 2)
            cv2.putText(copy, text_str, (x, y), FONT, 1.5, RED, 3)
        else:
            # for solution in self.solutions:
            for idx in self.solutions[0]:
                last_scene = self.scenes[0]
                card = last_scene.card_array[idx]
                cnt = card.contour
                cv2.drawContours(copy, [cnt], 0, GREEN, 3)
        return copy

    @staticmethod
    def are_valid_set(cards):
        for i in range(4):
            if len(set(card[i] for card in cards)) == 2:
                return False
        return True


class AnalyzeGames:
    """Structure to store information about Card IDs"""

    def __init__(self):
        self.n_buffer = 5
        self.current_info = []
        self.frame_cnt = 0

        self.all_card_info = []
        self.solution = None
        self.solved = False
        self.solutions = []

    def add_to_buffer(self, current_frame_count, current_card_info):
        self.frame_cnt = current_frame_count
        self.current_info = current_card_info

        self.all_card_info.append(self.current_info)

        if self.frame_cnt >= self.n_buffer:
            self.all_card_info.pop(0)
            self.solve()

    def solve(self):
        card_array = np.array(self.all_card_info)
        self.solutions = []
        self.solution = None
        if card_array.ndim > 1 and card_array.shape[1] > 0 and None not in card_array[:, :, 3:]:
            # same_cards = (card_array[0, :, 3:] == card_array[1, :, 3:]).all(axis=1) & \
            #      (card_array[1, :, 3:] == card_array[2, :, 3:]).all(axis=1)

            # card_list = card_array[0, :, 3:][same_cards]
            card_list = mode(card_array[:, :, 3:])[0][0]

            for card in combinations(enumerate(card_list), 3):
                test = np.array(card)
                indexes = test[:, 0]
                card = test[:, 1]

                if self.are_valid_set(card):
                    self.solutions.append(indexes)
                    self.solution = indexes
                    self.solved = True

    def draw_solution(self, img_in):
        copy = img_in.copy()
        if self.solution is None:
            text_str = 'No Solution Found'
            h, w, d = copy.shape
            x = int(w / 2) - 100
            y = int(h / 2)
            cv2.putText(copy, text_str, (x, y), FONT, 1.5, RED, 3)

        else:
            for idx in self.solution:
                card = self.all_card_info[0][idx]
                if card[2] is not None:
                    cv2.drawContours(copy, [card[2]], -1, (0, 255, 0), 5)

        # print(self.solutions)

        return copy

    @staticmethod
    def are_valid_set(cards):
        for i in range(4):
            if len(set(card[i] for card in cards)) == 2:
                return False
        return True


TEMPLATE_ARRAY = setup_templates()