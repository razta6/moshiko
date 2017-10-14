import sys
import os
import logging
# DEVELOPMENT
import cv2

# Constants
DATA_DIR = '/usr/share/opencv/'
NET_WEIGHT, NET_HEIGHT = 224, 224

PROFILES = {
    'HAAR_FRONTALFACE_ALT2': 'haarcascades/haarcascade_frontalface_alt2.xml',
    'HAAR_FRONTALFACE_ALT': 'haarcascades/haarcascade_frontalface_alt.xml',
    'HAAR_FRONTALFACE_DEFAULT': 'haarcascades/haarcascade_frontalface_default.xml',
    'HAAR_PROFILEFACE': 'haarcascades/haarcascade_profileface.xml'
}

# CV compatibility stubs
if 'IMREAD_GRAYSCALE' not in dir(cv2):
    # <2.4
    cv2.IMREAD_GRAYSCALE = 0
if 'cv' in dir(cv2):
    # <3.0
    cv2.CASCADE_DO_CANNY_PRUNING = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
    cv2.CASCADE_FIND_BIGGEST_OBJECT = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
    cv2.FONT_HERSHEY_SIMPLEX = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
    cv2.LINE_AA = cv2.cv.CV_AA


    def getTextSize(buf, font, scale, thickness):
        return cv2.cv.GetTextSize(buf, font)


    def putText(im, line, pos, font, scale, color, thickness, lineType):
        return cv2.cv.PutText(cv2.cv.fromarray(im), line, pos, font, color)


    cv2.getTextSize = getTextSize
    cv2.putText = putText

# Set logger output, level and format
logging.basicConfig(filename='moshiko.log', format='%(asctime)s\t%(levelname)s\t%(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p', level=logging.DEBUG)


# Support functions
def message(msg):
    sys.stderr.write("INFO\t{}\n".format(msg))
    logging.info(msg)


def debug(msg):
    sys.stderr.write("DEBUG\t{}\n".format(msg))
    logging.debug(msg)


def warning(msg):
    sys.stderr.write("WARNING\t{}\n".format(msg))
    logging.warning(msg)


def error(msg):
    sys.stderr.write("ERROR\t{}\n".format(msg))
    logging.error(msg)


def fatal(msg):
    sys.stderr.write("FATAL\t{}\n".format(msg))
    logging.error("!!!FATAL!!!" + msg)
    sys.exit(1)


def crop_rect(im, rect):
    pad = 40
    if rect[1] - pad < 0 or rect[1] + rect[3] + pad > 224 \
            or rect[0] - pad < 0 or rect[0] + rect[2] + pad > 224:
        return im[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    else:
        return im[rect[1] - pad:rect[1] + rect[3] + pad, rect[0] - pad:rect[0] + rect[2] + pad]


def get_prediction_from_names(torch_output):
    classification = []
    try:
        names = open('names.txt')
        names = names.readlines()
    except:
        return classification
    for label in torch_output:
        classification.append(names[label])
    return classification
