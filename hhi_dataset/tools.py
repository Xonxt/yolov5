##
# ingroup: DatasetAnnotation
# file:    tools.py
# brief:   This file contains useful utilities
# author:  Nikita Kovalenko (mykyta.kovalenko@hhi.fraunhofer.de)
# date:    02.09.2020
#
# Copyright:
# 2020 Fraunhofer Institute for Telecommunications, Heinrich-Hertz-Institut (HHI)
# The copyright of this software source code is the property of HHI.
# This software may be used and/or copied only with the written permission
# of HHI and in accordance with the terms and conditions stipulated
# in the agreement/contract under which the software has been supplied.
# The software distributed under this license is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either expressed or implied.
##

#----------------------
import os, sys
import cv2
import numpy as np
import math
import traceback
import itertools
import json
import datetime, time
import shutil
import random
import string
import getpass
import re, textwrap
#----------------------
from enum import Enum
#----------------------
if os.name != 'nt':
    import pwd
else:
    import ctypes
#----------------------

#----------------------
# KEYCODES
KEY_ENTER = 13
KEY_SPACEBAR = 32
KEY_ESC = 27
KEY_BACKSPACE = 8
KEY_TAB = 9

KEY_F1 = 7340032 if os.name == 'nt' else 65470
KEY_F2 = 7405568 if os.name == 'nt' else 65471
KEY_F3 = 7471104 if os.name == 'nt' else 65472
KEY_F4 = 7536640 if os.name == 'nt' else 65473

KEY_ARROW_LEFT = 2424832  if os.name == 'nt' else 65361
KEY_ARROW_RIGHT = 2555904 if os.name == 'nt' else 65363
KEY_ARROW_UP = 2490368    if os.name == 'nt' else 65362
KEY_ARROW_DOWN = 2621440  if os.name == 'nt' else 65364

KEY_PAGE_UP = 2162688   if os.name == 'nt' else 65365
KEY_PAGE_DOWN = 2228224 if os.name == 'nt' else 65366

KEY_DEL = 3014656 if os.name == 'nt' else 65535
KEY_HOME = 2359296
KEY_END = 2293760
KEY_INSERT = 2949120

# PRIMARY COLORS
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_MAGENTA = (255,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_CYAN = (255,255,0)
COLOR_GOLD = (0,215,255)
COLOR_MAROON = (0,0,128)
COLOR_PURPLE = (128,0,128)
COLOR_NAVY = (128,0,0)
COLOR_TEAL = (128,128,0)
COLOR_ORANGE = (0,69,255)
COLOR_OLIVE = (0,128,128)
COLOR_KHAKI = (140,230,240)
COLOR_FOREST = (0,128,0)
COLOR_WHITE = (255,255,255)
COLOR_GREY = (128,128,128)
COLOR_DARK_GREY = (200,200,200)
COLOR_BLACK = (0,0,0)

# a few primary colors
COLORS = [COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_MAGENTA, COLOR_YELLOW, \
          COLOR_CYAN, COLOR_GOLD, COLOR_MAROON, COLOR_PURPLE, COLOR_NAVY, \
          COLOR_TEAL, COLOR_ORANGE, COLOR_OLIVE, COLOR_KHAKI, COLOR_FOREST, \
          COLOR_WHITE, COLOR_GREY, COLOR_BLACK]

# default color for drawing polygons and rectangles:
DEFAULT_COLOR = COLOR_GREEN

## --------------------------------------------------------------------------------------------------------------------------
# generate a unique string ID of given length
def generate_uuid(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=length))

#--------------------------------------------
def generate_annotation_uuid(class_type):
    uuid = ""
    if class_type == 'rectangle':
        uuid += "rt_"
    elif class_type == 'polygon':
        uuid += "pg_"
    elif class_type == 'point':
        uuid += "pt_"
    elif class_type == 'vector':
        uuid += "vt_"

    uuid += f"{generate_uuid(4)}-{generate_uuid(2)}"

    return uuid

## --------------------------------------------------------------------------------------------------------------------------
def get_user_name():

    if os.name != 'nt':
        name = " ".join(pwd.getpwuid(os.getuid())[4].split(",")).strip()

    else:
        GetUserNameEx = ctypes.windll.secur32.GetUserNameExW
        NameDisplay = 3

        size = ctypes.pointer(ctypes.c_ulong(0))
        GetUserNameEx(NameDisplay, None, size)

        nameBuffer = ctypes.create_unicode_buffer(size.contents.value)
        GetUserNameEx(NameDisplay, nameBuffer, size)

        name = nameBuffer.value

    if not len(name):
        name = str(getpass.getuser())
    elif "," in name:
        name_parts = name.split(",")
        name = " ".join([p.strip() for p in name_parts[::-1]])

    return name

# -------------------------------------------------------------
def get_annotation_timestamp():
    try:
        author_name = str(get_user_name())
    except:
        author_name = str(getpass.getuser())
    timestamp = datetime.datetime.now().strftime('%d.%m.%Y - %H:%M:%S')
    return {'who': author_name,
            'when': timestamp}
    

## --------------------------------------------------------------------------------------------------------------------------
def split_long_line(text, max_len=40):
    return textwrap.wrap(text, max_len)

## --------------------------------------------------------------------------------------------------------------------------
# find something in a list based on condition
def list_find(lst, condition, default=-1):
    idx = [i for i, elem in enumerate(lst) if condition(elem)]
    if len(idx):
        return idx[0]
    else:
        return default

# ----------------------
# calculate Euclidian distance between two points in 2D...
def dist_2d(pt1, pt2=(0, 0)):
    try:
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        return dist
    except:
        return 0

# ...and in 3D
def dist_3d(pt1, pt2=(0,0,0)):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5

# check if point is contained within the rectangle
def rectContains(rect, pt):
    logic = min(rect[0], rect[2]) < pt[0] < max(rect[0], rect[2]) and min(rect[1], rect[3]) < pt[1] < max(rect[1], rect[3])
    return logic

# calculate rectangle area
def rectArea(rect):
    return abs(rect[2]-rect[0]) * abs(rect[3]-rect[1])

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = abs(x[:, 2] - x[:, 0])  # width
    y[:, 3] = abs(x[:, 3] - x[:, 1])  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# ----------------------
def resize_ratio(image, shape=(368,368), padding=True, pad_value=128, upscale=True, interpolation=cv2.INTER_CUBIC):
    """
    Resizes an input image while keeping the aspect ratio

    Arguments:
    image {np.ndarray} A numpy image
    shape {tuple} The new image size (height, width)
    constant_value {number} The value to pad the empty space on the image
    upscale {bool} If the input image is smaller than the output size, the image will be upscaled to fit the new dimensions
    interpolation {cv2.INTER_***} The interpolation algorithm for image resizing (default: cv.INTER_CUBIC)
    """

    if not is_valid_image(image):
        return None

    if isinstance(shape, int):
        shape = (shape, shape)

    if all(np.array(image.shape[:2]) < np.array(shape)) and not upscale:
        dim = image.shape[:2]
    else:
        img_sh = np.array(image.shape[:2])
        dim = (img_sh * (np.array(shape) / img_sh).min()).astype(np.int32)

    if padding:
        resized_image = np.ones(tuple(shape) + image.shape[2:], dtype=image.dtype) * pad_value
        resized_image[:dim[0],:dim[1]] = cv2.resize(image, tuple(dim[::-1]), interpolation=interpolation)
    else:
        resized_image = cv2.resize(image, tuple(dim[::-1]), interpolation=interpolation)

    return resized_image, dim

## --------------------------------------------------------------------------------------------------------------------------
## LOGGING

MESSAGE_LOG = []
message_colors = [(255, 255, 255), (0, 255, 255), (0, 0, 255), (0,255,0)]

class MessageType(Enum):
    LOG = 0
    WARNING = 1
    ERROR = 2
    INFO = 3

def log(msg, prefix="", msg_type=MessageType.LOG, msg_time=1.5, write_to_file=False, milliseconds=False):
    global MESSAGE_LOG

    timestamp = datetime.datetime.now()
    stamp_time = timestamp.strftime('%H:%M:%S')
    stamp_ms = timestamp.strftime('.%f') if milliseconds else ''

    message_string = f"{prefix}{msg_type.name} :: {stamp_time}{stamp_ms} :: {msg}"
    print(message_string)

    MESSAGE_LOG.append([message_string, msg_time, message_colors[msg_type.value]])

    if write_to_file:
        open("log.txt", "a+").write(message_string + "\n")

#----------------------

# check if the image is valid
def is_valid_image(image):

    if image is None:
        return False

    if not isinstance(image, np.ndarray):
        return False

    if image.size == 0:
        return False

    return True

# ----------------------
# wrapper around the Numpy's "min" function that returns False if the array is empty
def np_min(x, default=False):
    if isinstance(x, np.ndarray) and x.size == 0:
        return default
    else:
        return x.min()

# wrapper around the Numpy's "max" function that returns False if the array is empty
def np_max(x, default=False):
    if isinstance(x, np.ndarray) and x.size == 0:
        return default
    else:
        return x.max()

## --------------------------------------------------------------------------------------------------------------------------
## GUI FUNCTIONS

# ----------------------
# FUNCTIONS FOR DRAWING STYLED (DOTTED OR DASHED) FIGURES
def styledLine(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = dist_2d(pt1, pt2)
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1

def styledPoly(img, pts, color, thickness=1, style='dotted', gap=10):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        styledLine(img, s, e, color, thickness, style, gap)

def styledRect(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    styledPoly(img, pts, color, thickness, style, gap)

## --------------------------------------------------------------------------------------------------------------------------
# Draw an ASCII progress bar in the [======>........] style
def draw_progress_bar(step=0, total_steps=100, length=30, percentage=False):
    if step > total_steps:
        step = total_steps
    done = math.ceil(length * ((step+1) / total_steps))
    left = length - done

    bar = "["

    if step < (total_steps-1):
        bar += "".join(['=' for i in range(done)] + [">"] + ['.' for i in range(left)])
    else:
        bar += "".join(['=' for i in range(done+1)])

    bar += "]"

    if percentage:
        bar += " ({:.1f}%)".format( (step+1) / total_steps * 100 )
    return bar

def show_progress_bar(message, step=0, total_steps=100, length=30, percentage=False):
    print("{} {}".format(message, draw_progress_bar(step, total_steps, length=length, percentage=percentage)), end='\r', flush=True)

## --------------------------------------------------------------------------------------------------------------------------
class Align(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2

class Valign(Enum):
    TOP = 0
    MIDDLE = 1
    BOTTOM = 2

class Sorting(Enum):
    NONE = 0
    DSCN = 1
    ASCN = 2

def draw_text(canvas, string, org=(0,0), align=Align.LEFT, valign=Valign.TOP, inner_pad=5, outer_pad=[5, 5],
              font_face=cv2.FONT_HERSHEY_PLAIN, font_size=2, font_color=(255, 255, 255), font_width=1, line_type=cv2.LINE_4,
              outline=False, outline_color=(0, 0, 0), outline_width=2, darken_background=-1, background_color=None, sort=Sorting.NONE):
    """
    A function for a more sophisticated and flexible text rendering. Using this instead of the standard cv2.putText() may result
    in a small performance loss.
    """

    max_width = 0
    max_height = outer_pad[1] * 2

    if not isinstance(string, list):
        string = [string]

    str_quantity = len(string)

    # list of dimensions
    dims_array = list()

    # iterate through strings to calculate the box dimensions
    for i, s in enumerate(string):
        dims, _ = cv2.getTextSize(s, font_face, font_size, font_width)
        dims_array.append(dims)

        # calculate text block dimensions:
        if dims[0] > max_width:
            max_width = dims[0]
        max_height += dims[1]

        if i > 0 and i < str_quantity:
            max_height += inner_pad

    # set the maximum width of the text block
    max_width = max_width + outer_pad[0] * 2

    # if no image was supplied, just return the text block size
    if not is_valid_image(canvas):
        return (max_width, max_height)

    return_canvas = (org is None)
    if return_canvas:
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        org = [0, 0]
        if align == Align.RIGHT:
            org[0] = max_width - 1
        elif align == Align.CENTER:
            org[0] = max_width // 2
        if valign == Valign.MIDDLE:
            org[1] = max_height // 2
        elif valign == Valign.BOTTOM:
            org[1] = max_height - 1

    # calculate the top-left point for the text block
    tl = [org[0], org[1]]
    br = [tl[0] + max_width, tl[1] + max_height]

    # horizontal align:
    if align == Align.RIGHT:
        tl[0] = org[0] - max_width
        br[0] = org[0]
    elif align == Align.CENTER:
        tl[0] = org[0] - max_width // 2
        br[0] = org[0] + max_width // 2

    # vertical align:
    if valign == Valign.MIDDLE:
        tl[1] = org[1] - max_height // 2
        br[1] = org[1] + max_height // 2
    elif valign == Valign.BOTTOM:
        tl[1] = org[1] - max_height
        br[1] = org[1]

    if darken_background >= 0:
        darken_background = np.clip(darken_background, 0, 1)

    # if necessary, draw a background color
    if background_color is not None and isinstance(background_color, tuple):

        if darken_background < 0:
            cv2.rectangle(canvas, (tl[0], tl[1]), (br[0], br[1]), background_color, outline_width)
            cv2.rectangle(canvas, (tl[0], tl[1]), (br[0], br[1]), background_color, -1)
        else:
            overlay = np.zeros(canvas.shape, dtype=canvas.dtype)
            cv2.rectangle(overlay, (tl[0], tl[1]), (br[0], br[1]), background_color, outline_width)
            cv2.rectangle(overlay, (tl[0], tl[1]), (br[0], br[1]), background_color, -1)
            canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, darken_background, 1.0)
    else:
        # if necessary, darken the background of the text by the specified value
        if 0 <= darken_background <= 1:
            canvas[tl[1]:br[1], tl[0]:br[0]] = np.clip(np.uint8(canvas[tl[1]:br[1], tl[0]:br[0]]) * darken_background, 0, 255)

    # sort the text by width, if necessary:
    if sort == Sorting.NONE:
        dims_array = enumerate(dims_array)
    else:
        dims_array = sorted(enumerate(dims_array), key=lambda x: x[1][0], reverse=(sort == Sorting.DSCN))

    # iterate through text lines and draw them:
    for i, (j, dim) in enumerate(dims_array):
        s = string[j]

        pt = [outer_pad[0] + int(tl[0]), outer_pad[1] + int(tl[1]) + (i+1) * dim[1] + i * inner_pad]
        if align == Align.CENTER:
            pt[0] = org[0] - dim[0] // 2
        elif align == Align.RIGHT:
            pt[0] = br[0] - dim[0] - outer_pad[0]

        # draw the outline, when necessary
        if outline:
            cv2.putText(canvas, s, (pt[0],pt[1]), font_face, font_size, outline_color, font_width + outline_width, line_type)

        # draw the string
        cv2.putText(canvas, s, (pt[0],pt[1]), font_face, font_size, font_color[i] if isinstance(font_color, list) else font_color,
                    font_width, line_type)

    # return the total width and height of the text block
    if return_canvas:
        return canvas
    else:
        return (max_width, max_height)
## --------------------------------------------------------------------------------------------------------------------------
FLASHING_TEXT = {"text":"", "duration":0, "tick":0, "color":COLOR_WHITE}
BLACK_MASK = None

def flash_text(text, duration=1, color=COLOR_WHITE):
    global FLASHING_TEXT, BLACK_MASK

    FLASHING_TEXT = {"text":text, "duration":duration, "tick":duration, "color":color}
    BLACK_MASK = None


def flash_text_on_screen(image, time_start=None, font_size=3):
    global FLASHING_TEXT, BLACK_MASK

    if FLASHING_TEXT['tick'] <= 0:
        return

    if not is_valid_image(image):
        return

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_width = font_size * 2

    dims, _ = cv2.getTextSize(FLASHING_TEXT['text'], font_face, font_size, font_width)

    frame = np.zeros((dims[1]+100, dims[0]+50, 3), dtype=np.uint8)

    frame_alpha = frame.copy()
    frame_alpha[:] = (255,0,255)

    pt = (frame.shape[1]//2 - dims[0]//2, frame.shape[0]//2 + dims[1]//2)
    cv2.putText(frame, FLASHING_TEXT['text'], pt, font_face, font_size, COLOR_WHITE, font_width)
    cv2.putText(frame_alpha, FLASHING_TEXT['text'], pt, font_face, font_size, COLOR_BLACK, font_width*2)
    cv2.putText(frame_alpha, FLASHING_TEXT['text'], pt, font_face, font_size, FLASHING_TEXT['color'], font_width)

    (x, y) = (image.shape[1] // 2, image.shape[0] // 2)

    if frame.shape[1] >= image.shape[1]:
        fx = image.shape[1] / frame.shape[1]
        frame = cv2.resize(frame, None, fx=fx, fy=fx)
        frame_alpha = cv2.resize(frame_alpha, None, fx=fx, fy=fx)

    if BLACK_MASK is None:
        BLACK_MASK = (frame_alpha==[0,0,0]).all(-1)

    crop = image[y-frame.shape[0]//2:y-frame.shape[0]//2+frame.shape[0],
                 x-frame.shape[1]//2:x-frame.shape[1]//2+frame.shape[1]]

    try:
        alpha = np.clip(FLASHING_TEXT['tick'] / FLASHING_TEXT['duration'], 0, 1)
        crop[:] = cv2.addWeighted(frame, np.clip(FLASHING_TEXT['tick'] / FLASHING_TEXT['duration'], 0, 1), crop, 1.0, 1.0)

        crop[BLACK_MASK] = np.clip(crop[BLACK_MASK] * (1-alpha), 0, 255)

    except Exception:
        # traceback.print_exc()
        pass

    time_tick = np.clip(time.time() - time_start, 0.1, 10)

    FLASHING_TEXT['tick'] -= time_tick

